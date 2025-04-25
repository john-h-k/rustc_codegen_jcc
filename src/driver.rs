use std::{cell::RefCell, fmt::Write, num::NonZeroUsize, ops::Range};

use rustc_abi::{
    BackendRepr, FieldsShape, Float, HasDataLayout, Integer, Primitive, Scalar, Variants,
    WrappingRange,
};
use rustc_codegen_ssa::traits::{
    AsmCodegenMethods, BackendTypes, BaseTypeCodegenMethods, ConstCodegenMethods,
    DebugInfoCodegenMethods, LayoutTypeCodegenMethods, MiscCodegenMethods, PreDefineCodegenMethods,
    StaticCodegenMethods, TypeMembershipCodegenMethods,
};
use rustc_const_eval::interpret::{
    self, Allocation, ConstAllocation, GlobalAlloc, InitChunk, Pointer, read_target_uint,
};
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::{
    bug,
    mir::mono::{CodegenUnit, Linkage, Visibility},
    ty::{
        self, CoroutineArgsExt, ExistentialTraitRef, Instance, Ty, TyCtxt, TyKind, TypingEnv,
        layout::{
            FnAbiOf, FnAbiOfHelpers, HasTyCtxt, HasTypingEnv, LayoutOf, LayoutOfHelpers,
            TyAndLayout,
        },
        print::with_no_trimmed_paths,
    },
};
use rustc_session::Session;
use rustc_span::{Symbol, def_id::DefId};
use rustc_target::callconv::{ArgAbi, CastTarget, FnAbi, PassMode};
use rustc_type_ir::{FloatTy, IntTy, TypeVisitableExt, UintTy};
use smallvec::{SmallVec, smallvec};

use crate::jcc::{
    alloc::ArenaAlloc,
    ir::{
        AddrOffset, AsIrRaw, IrBasicBlock, IrFloatTy, IrFunc, IrGlb, IrId, IrIntTy, IrLinkage,
        IrOp, IrUnit, IrVarTy, IrVarTyAggregate, IrVarTyAggregateTy, IrVarTyFuncFlags, IrVarValue,
        IrVarValueAddr, IrVarValueListEl, IrVarValueTy,
    },
};

use std::fmt::Debug;
use std::fmt::Formatter;
struct TypedDebugWrapper<'a, T: ?Sized>(&'a T);

impl<T: Debug> Debug for TypedDebugWrapper<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}::{:?}", core::any::type_name::<T>(), self.0)
    }
}

trait TypedDebug: Debug {
    fn typed_debug(&self) -> TypedDebugWrapper<'_, Self> {
        TypedDebugWrapper(self)
    }
}

impl<T: ?Sized + Debug> TypedDebug for T {}

pub fn integer_ty_to_jcc_ty<'tcx>(cx: &CodegenCx<'tcx>, integer: rustc_abi::Integer) -> IrVarTy {
    match integer {
        Integer::I8 => cx.type_i8(),
        Integer::I16 => cx.type_i16(),
        Integer::I32 => cx.type_i32(),
        Integer::I64 => cx.type_i64(),
        Integer::I128 => cx.type_i128(),
    }
}

pub fn float_ty_to_jcc_ty<'tcx>(cx: &CodegenCx<'tcx>, float: rustc_abi::Float) -> IrVarTy {
    match float {
        Float::F16 => cx.type_f16(),
        Float::F32 => cx.type_f32(),
        Float::F64 => cx.type_f64(),
        Float::F128 => cx.type_f128(),
    }
}

pub fn scalar_ty_to_jcc_ty<'tcx>(cx: &CodegenCx<'tcx>, scalar: &rustc_abi::Scalar) -> IrVarTy {
    match scalar.primitive() {
        Primitive::Int(integer, _) => integer_ty_to_jcc_ty(cx, integer),
        Primitive::Float(float) => float_ty_to_jcc_ty(cx, float),
        Primitive::Pointer(address_space) => cx.type_ptr_ext(address_space),
    }
}

fn struct_fields<'tcx>(cx: &CodegenCx<'tcx>, layout: &TyAndLayout<'tcx>) -> (Vec<IrVarTy>, bool) {
    let field_count = layout.fields.count();
    let mut packed = false;
    let mut offset = rustc_abi::Size::ZERO;
    let mut prev_effective_align = layout.align.abi;
    let mut result: Vec<_> = Vec::with_capacity(1 + field_count * 2);

    for i in layout.fields.index_by_increasing_offset() {
        let target_offset = layout.fields.offset(i);
        let field = layout.field(cx, i);
        let effective_field_align = layout
            .align
            .abi
            .min(field.align.abi)
            .restrict_for_offset(target_offset);

        packed |= effective_field_align < field.align.abi;
        assert!(target_offset >= offset);

        let padding = target_offset - offset;
        let padding_align = prev_effective_align.min(effective_field_align);

        assert_eq!(offset.align_to(padding_align) + padding, target_offset);

        result.push(cx.type_padding_filler(padding, padding_align));
        result.push(ty_to_jcc_ty(cx, &field));

        offset = target_offset + field.size;
        prev_effective_align = effective_field_align;
    }

    if layout.is_sized() && field_count > 0 {
        if offset > layout.size {
            bug!(
                "layout: {:#?} stride: {:?} offset: {:?}",
                layout,
                layout.size,
                offset
            );
        }

        let padding = layout.size - offset;
        let padding_align = prev_effective_align;

        assert_eq!(offset.align_to(padding_align) + padding, layout.size);

        result.push(cx.type_padding_filler(padding, padding_align));

        assert_eq!(result.len(), 1 + field_count * 2);
    }

    (result, packed)
}

pub fn ty_to_jcc_ty<'tcx>(cx: &CodegenCx<'tcx>, ty_layout: &TyAndLayout<'tcx>) -> IrVarTy {
    let TyAndLayout { ty, layout } = &ty_layout;

    // TODO: caching
    // read comments in gcc impl because there are useful details - `rust-lang/rustc_codegen_gcc src/type_of.rs gcc_type`

    if let BackendRepr::Scalar(ref scalar) = layout.backend_repr {
        let var_ty = match *ty.kind() {
            ty::FnPtr(sig_tys, hdr) => {
                cx.fn_ptr_backend_type(cx.fn_abi_of_fn_ptr(sig_tys.with(hdr), ty::List::empty()))
            }
            _ => scalar_ty_to_jcc_ty(cx, scalar),
        };

        return var_ty;
    }

    assert!(
        !ty.has_escaping_bound_vars(),
        "{:?} has escaping bound vars",
        ty
    );

    // Make sure lifetimes are erased, to avoid generating distinct LLVM
    // types for Rust types that only differ in the choice of lifetimes.
    let normal_ty = cx.tcx.erase_regions(*ty);

    if *ty != normal_ty {
        let mut layout = cx.layout_of(normal_ty);

        ty_to_jcc_ty(cx, &layout)
    } else {
        match layout.backend_repr {
            BackendRepr::Scalar(_) => bug!("handled elsewhere"),
            BackendRepr::SimdVector { ref element, count } => {
                todo!("SimdVector");
            }
            BackendRepr::ScalarPair(l, r) => {
                return cx.type_struct(
                    &[scalar_ty_to_jcc_ty(cx, &l), scalar_ty_to_jcc_ty(cx, &r)],
                    false,
                );
            }
            BackendRepr::Memory { .. } => {}
        }

        let name = match *ty.kind() {
            ty::Adt(..)
            | ty::Closure(..)
            | ty::CoroutineClosure(..)
            | ty::Foreign(..)
            | ty::Coroutine(..)
            | ty::Str
                if !cx.sess().fewer_names() =>
            {
                let mut name = with_no_trimmed_paths!(ty.to_string());

                if let (&ty::Adt(def, _), &Variants::Single { index }) =
                    (ty.kind(), &layout.variants)
                {
                    if def.is_enum() && !def.variants().is_empty() {
                        write!(&mut name, "::{}", def.variant(index).name).unwrap();
                    }
                }
                if let (&ty::Coroutine(_, _), &Variants::Single { index }) =
                    (ty.kind(), &layout.variants)
                {
                    write!(&mut name, "::{}", ty::CoroutineArgs::variant_name(index)).unwrap();
                }

                Some(name)
            }
            ty::Adt(..) => Some(String::new()),
            _ => None,
        };

        match layout.fields {
            FieldsShape::Primitive | FieldsShape::Union(_) => {
                let fill = cx.type_padding_filler(layout.size, layout.align.abi);
                let packed = false;
                match name {
                    None => cx.type_struct(&[fill], packed),
                    Some(ref name) => {
                        // TODO: can we use the name here to get more aliasing info?

                        cx.type_struct(&[fill], packed)
                    }
                }
            }
            FieldsShape::Array { count, .. } => {
                let el_ty = ty_to_jcc_ty(cx, &ty_layout.field(cx, 0));
                cx.type_array(el_ty, count)
            }
            FieldsShape::Arbitrary { .. } => match name {
                None => {
                    let (fields, packed) = struct_fields(cx, ty_layout);
                    cx.type_struct(&fields, packed)
                }
                Some(ref name) => {
                    // TODO: can we use the name here to get more aliasing info?

                    let (fields, packed) = struct_fields(cx, ty_layout);
                    cx.type_struct(&fields, packed)
                }
            },
        }
    }
}

pub(crate) struct JccModule {
    pub(crate) unit: IrUnit,
}

unsafe impl Send for JccModule {}
unsafe impl Sync for JccModule {}

pub struct CodegenCx<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    // not present for codegen_allocator
    pub cg_unit: Option<&'tcx CodegenUnit<'tcx>>,
    pub unit: IrUnit,
    pub arena: ArenaAlloc,

    pub glb_map: RefCell<FxHashMap<String, IrGlb>>,
    pub vtables: RefCell<FxHashMap<(Ty<'tcx>, Option<ExistentialTraitRef<'tcx>>), IrOp>>,

    pub cur_bb: RefCell<Option<IrBasicBlock>>,
}

impl<'tcx> CodegenCx<'tcx> {
    pub fn alloc_next_op(&self) -> IrOp {
        let bb = self.cur_bb.borrow().unwrap();
        eprintln!("bb {}", bb.id());
        eprintln!("bb {:?}", bb.as_mut_ptr());
        eprintln!("{} stmts", bb.stmts().len());
        let stmt = bb.alloc_stmt();
        eprintln!(
            "{} stmts new id {}",
            bb.stmts().len(),
            bb.last().unwrap().id()
        );
        stmt.alloc_op()
    }

    pub fn mk_next_op(&self, mk: impl FnOnce(IrOp)) -> IrOp {
        let op = self.alloc_next_op();

        mk(op);

        op
    }

    fn try_get_glb_by_name(&self, name: &str) -> Option<IrGlb> {
        self.glb_map.borrow_mut().get(name).copied()
    }

    fn try_get_glb(&self, instance: Instance<'tcx>) -> Option<IrGlb> {
        let sym = self.tcx.symbol_name(instance).name;
        self.try_get_glb_by_name(sym)
    }

    fn set_cur_bb(&self, bb: IrBasicBlock) {
        *self.cur_bb.borrow_mut() = Some(bb)
    }

    fn get_glb(&self, instance: Instance<'tcx>) -> IrGlb {
        // TODO: vis/link won't be updated properly in predefine_fn
        let sym = self.tcx.symbol_name(instance).name;
        let (_, params, ret) = self.abi_of(instance);
        let glb = self.declare_fn(
            instance,
            Linkage::External,
            Visibility::Default,
            sym,
            &params[..],
            &ret,
        );

        self.glb_map.borrow_mut().insert(sym.to_string(), glb);
        glb
    }

    fn alloc_bytes_var(&self, alloc: &ConstAllocation<'_>) -> IrGlb {
        let arena = self.arena.as_ref();
        let alloc = alloc.inner();

        let unit = self.unit;
        let ty = unit.var_ty_bytes(alloc.len());

        let glb = unit.add_global_def_var(ty, None, IrLinkage::Internal);
        let var = glb.var();

        let mut llvals = Vec::with_capacity(alloc.provenance().ptrs().len() + 1);

        let dl = self.data_layout();
        let pointer_size = dl.pointer_size.bytes() as usize;

        // Note: this function may call `inspect_with_uninit_and_ptr_outside_interpreter`, so `range`
        // must be within the bounds of `alloc` and not contain or overlap a pointer provenance.
        fn append_chunks_of_init_and_uninit_bytes<'a, 'b>(
            cx: &'a CodegenCx<'b>,
            llvals: &mut Vec<IrVarValueListEl>,
            alloc: &'a Allocation,
            range: Range<usize>,
        ) {
            let offset = range.start;
            let chunks = alloc.init_mask().range_as_init_chunks(range.clone().into());

            let chunk_to_llval = move |chunk| match chunk {
                InitChunk::Init(range) => {
                    let range = (range.start.bytes() as usize)..(range.end.bytes() as usize);
                    let bytes = alloc.inspect_with_uninit_and_ptr_outside_interpreter(range);

                    Some(IrVarValueListEl {
                        offset,
                        value: IrVarValue::from_bytes(cx.unit, offset, bytes),
                    })
                }
                InitChunk::Uninit(range) => {
                    // do nothing, all ranges outside of value lists are uninit
                    // (or they are zeroed currently i think? but they are _meant_ be uninit)
                    None
                }
            };

            // Generating partially-uninit consts is limited to small numbers of chunks,
            // to avoid the cost of generating large complex const expressions.
            // For example, `[(u32, u8); 1024 * 1024]` contains uninit padding in each element, and
            // would result in `{ [5 x i8] zeroinitializer, [3 x i8] undef, ...repeat 1M times... }`.
            let max = cx.sess().opts.unstable_opts.uninit_const_chunk_threshold;
            let allow_uninit_chunks = chunks.clone().take(max.saturating_add(1)).count() <= max;

            if allow_uninit_chunks {
                llvals.extend(chunks.filter_map(chunk_to_llval));
            } else {
                // If this allocation contains any uninit bytes, codegen as if it was initialized
                // (using some arbitrary value for uninit bytes).

                let bytes = alloc.inspect_with_uninit_and_ptr_outside_interpreter(range);

                llvals.push(IrVarValueListEl {
                    offset,
                    value: IrVarValue::from_bytes(cx.unit, offset, bytes),
                });
            }
        }

        let mut next_offset = 0;
        for &(offset, prov) in alloc.provenance().ptrs().iter() {
            let offset = offset.bytes();
            assert_eq!(offset as usize as u64, offset);
            let offset = offset as usize;
            if offset > next_offset {
                // This `inspect` is okay since we have checked that there is no provenance, it
                // is within the bounds of the allocation, and it doesn't affect interpreter execution
                // (we inspect the result after interpreter execution).
                append_chunks_of_init_and_uninit_bytes(
                    self,
                    &mut llvals,
                    alloc,
                    next_offset..offset,
                );
            }
            let ptr_offset = read_target_uint(
                dl.endian,
                // This `inspect` is okay since it is within the bounds of the allocation, it doesn't
                // affect interpreter execution (we inspect the result after interpreter execution),
                // and we properly interpret the provenance as a relocation pointer offset.
                alloc.inspect_with_uninit_and_ptr_outside_interpreter(
                    offset..(offset + pointer_size),
                ),
            )
            .expect("const_alloc_to_llvm: could not read relocation pointer")
                as u64;

            let address_space = self.tcx.global_alloc(prov.alloc_id()).address_space(self);

            self.scalar_to_var_value(
                interpret::Scalar::from_pointer(
                    Pointer::new(prov, rustc_abi::Size::from_bytes(ptr_offset)),
                    &self.tcx,
                ),
                Scalar::Initialized {
                    value: Primitive::Pointer(address_space),
                    valid_range: WrappingRange::full(dl.pointer_size),
                },
                self.type_ptr_ext(address_space),
            );

            next_offset = offset + pointer_size;
        }

        if alloc.len() >= next_offset {
            let range = next_offset..alloc.len();
            // This `inspect` is okay since we have check that it is after all provenance, it is
            // within the bounds of the allocation, and it doesn't affect interpreter execution (we
            // inspect the result after interpreter execution).
            append_chunks_of_init_and_uninit_bytes(self, &mut llvals, alloc, range);
        }

        let value = IrVarValue {
            ty: IrVarValueTy::List(llvals),
            var_ty: ty,
        };

        var.mk_value(&value);
        glb
    }
}

impl<'tcx> AsmCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn codegen_global_asm(
        &self,
        template: &[rustc_ast::InlineAsmTemplatePiece],
        operands: &[rustc_codegen_ssa::traits::GlobalAsmOperandRef<'tcx>],
        options: rustc_ast::InlineAsmOptions,
        line_spans: &[rustc_span::Span],
    ) {
        todo!()
    }

    fn mangled_name(&self, instance: Instance<'tcx>) -> String {
        todo!()
    }
}

impl<'tcx> BackendTypes for CodegenCx<'tcx> {
    type Value = IrOp;
    type Metadata = ();
    type Function = IrFunc;
    type BasicBlock = IrBasicBlock;
    type Type = IrVarTy;
    type Funclet = ();
    type DIScope = ();
    type DILocation = ();
    type DIVariable = ();
}

impl<'tcx> HasDataLayout for CodegenCx<'tcx> {
    fn data_layout(&self) -> &rustc_abi::TargetDataLayout {
        self.tcx.data_layout()
    }
}

impl<'tcx> HasTyCtxt<'tcx> for CodegenCx<'tcx> {
    fn tcx(&self) -> ty::TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx> HasTypingEnv<'tcx> for CodegenCx<'tcx> {
    fn typing_env(&self) -> ty::TypingEnv<'tcx> {
        TypingEnv::fully_monomorphized()
    }
}

impl<'tcx> LayoutOfHelpers<'tcx> for CodegenCx<'tcx> {
    type LayoutOfResult = TyAndLayout<'tcx>;

    fn handle_layout_err(
        &self,
        err: ty::layout::LayoutError<'tcx>,
        span: rustc_span::Span,
        ty: Ty<'tcx>,
    ) -> <Self::LayoutOfResult as ty::layout::MaybeResult<TyAndLayout<'tcx>>>::Error {
        todo!()
    }
}

impl<'tcx> FnAbiOfHelpers<'tcx> for CodegenCx<'tcx> {
    type FnAbiOfResult = &'tcx FnAbi<'tcx, Ty<'tcx>>;

    fn handle_fn_abi_err(
        &self,
        err: ty::layout::FnAbiError<'tcx>,
        span: rustc_span::Span,
        fn_abi_request: ty::layout::FnAbiRequest<'tcx>,
    ) -> <Self::FnAbiOfResult as ty::layout::MaybeResult<&'tcx FnAbi<'tcx, Ty<'tcx>>>>::Error {
        todo!()
    }
}

impl CodegenCx<'_> {
    pub fn type_none(&self) -> IrVarTy {
        IrVarTy::ty_none()
    }

    pub fn type_i1(&self) -> IrVarTy {
        IrVarTy::ty_i1()
    }

    pub fn type_struct(&self, fields: &[IrVarTy], packed: bool) -> IrVarTy {
        if packed {
            todo!("packed structs");
        }

        self.unit.var_ty_struct(fields)
    }

    pub fn type_union(&self, fields: &[IrVarTy]) -> IrVarTy {
        self.unit.var_ty_union(fields)
    }

    pub fn type_padding_filler(&self, size: rustc_abi::Size, align: rustc_abi::Align) -> IrVarTy {
        let unit = Integer::approximate_align(self, align);

        let size = size.bytes();
        let unit_size = unit.size().bytes();

        assert_eq!(size % unit_size, 0);

        let unit = integer_ty_to_jcc_ty(self, unit);

        self.type_array(unit, size / unit_size)
    }
}

impl<'tcx> BaseTypeCodegenMethods for CodegenCx<'tcx> {
    fn type_i8(&self) -> Self::Type {
        IrVarTy::ty_i8()
    }

    fn type_i16(&self) -> Self::Type {
        IrVarTy::ty_i16()
    }

    fn type_i32(&self) -> Self::Type {
        IrVarTy::ty_i32()
    }

    fn type_i64(&self) -> Self::Type {
        IrVarTy::ty_i64()
    }

    fn type_i128(&self) -> Self::Type {
        IrVarTy::ty_i128()
    }

    fn type_isize(&self) -> Self::Type {
        // TODO: pointer size
        IrVarTy::ty_i64()
    }

    fn type_f16(&self) -> Self::Type {
        IrVarTy::ty_f16()
    }

    fn type_f32(&self) -> Self::Type {
        IrVarTy::ty_f32()
    }

    fn type_f64(&self) -> Self::Type {
        IrVarTy::ty_f64()
    }

    fn type_f128(&self) -> Self::Type {
        todo!("f128")
    }

    fn type_array(&self, ty: Self::Type, len: u64) -> Self::Type {
        self.unit.var_ty_array(&ty, len as _)
    }

    fn type_func(&self, args: &[Self::Type], ret: Self::Type) -> Self::Type {
        self.unit.var_ty_func(args, &ret, IrVarTyFuncFlags::None)
    }

    fn type_kind(&self, ty: Self::Type) -> rustc_codegen_ssa::common::TypeKind {
        todo!()
    }

    fn type_ptr(&self) -> Self::Type {
        self.unit.var_ty_pointer()
    }

    fn type_ptr_ext(&self, _address_space: rustc_abi::AddressSpace) -> Self::Type {
        self.unit.var_ty_pointer()
    }

    fn element_type(&self, ty: Self::Type) -> Self::Type {
        todo!()
    }

    fn vector_length(&self, ty: Self::Type) -> usize {
        todo!()
    }

    fn float_width(&self, ty: Self::Type) -> usize {
        todo!()
    }

    fn int_width(&self, ty: Self::Type) -> u64 {
        todo!()
    }

    fn val_ty(&self, v: Self::Value) -> Self::Type {
        v.var_ty()
    }
}

impl<'tcx> MiscCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn vtables(
        &self,
    ) -> &std::cell::RefCell<
        rustc_data_structures::fx::FxHashMap<
            (Ty<'tcx>, Option<ty::ExistentialTraitRef<'tcx>>),
            Self::Value,
        >,
    > {
        &self.vtables
    }

    fn get_fn(&self, instance: Instance<'tcx>) -> Self::Function {
        self.get_glb(instance).func()
    }

    fn get_fn_addr(&self, instance: Instance<'tcx>) -> Self::Value {
        let glb = self.get_glb(instance);

        let op = self.alloc_next_op();
        op.mk_addr_glb(glb);
        op
    }

    fn eh_personality(&self) -> Self::Value {
        todo!()
    }

    fn sess(&self) -> &Session {
        self.tcx.sess
    }

    fn codegen_unit(&self) -> &'tcx CodegenUnit<'tcx> {
        self.cg_unit.unwrap()
    }

    fn set_frame_pointer_type(&self, llfn: Self::Function) {
        // TODO:
    }

    fn apply_target_cpu_attr(&self, llfn: Self::Function) {
        // TODO:
    }

    fn declare_c_main(&self, fn_type: Self::Type) -> Option<Self::Function> {
        let entry_name = self.sess().target.entry_name.as_ref();

        let argv_argc = [self.type_i32(), self.type_ptr()];
        let ret_code = self.type_i32();

        let func = self.declare_simple_fn(
            true,
            Linkage::External,
            Visibility::Hidden,
            entry_name,
            &argv_argc[..],
            &ret_code,
        );

        // NOTE: it is needed to set the current_func here as well, because get_fn() is not called
        // for the main function.

        // FIXME: have cur_fun + cur_bb not just bb (or something)
        let bb = func.func().last().unwrap();

        self.set_cur_bb(bb);

        Some(func.func())
    }
}

impl<'tcx> LayoutTypeCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn backend_type(&self, layout: TyAndLayout<'tcx>) -> Self::Type {
        ty_to_jcc_ty(self, &layout)
    }

    fn cast_backend_type(&self, ty: &CastTarget) -> Self::Type {
        todo!()
    }

    fn fn_decl_backend_type(&self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> Self::Type {
        let (params, ret) = self.abi_fn_ty(fn_abi);
        self.unit
            .var_ty_func(&params[..], &ret, IrVarTyFuncFlags::None)
    }

    fn fn_ptr_backend_type(&self, fn_abi: &FnAbi<'tcx, Ty<'tcx>>) -> Self::Type {
        self.type_ptr_ext(self.data_layout().instruction_address_space)
    }

    fn reg_backend_type(&self, ty: &rustc_abi::Reg) -> Self::Type {
        todo!()
    }

    fn immediate_backend_type(&self, layout: TyAndLayout<'tcx>) -> Self::Type {
        ty_to_jcc_ty(self, &layout)
    }

    fn is_backend_immediate(&self, layout: TyAndLayout<'tcx>) -> bool {
        matches!(
            layout.ty.kind(),
            TyKind::Bool
                | TyKind::Char
                | TyKind::Int(..)
                | TyKind::Uint(..)
                | TyKind::Float(..)
                | TyKind::RawPtr(..)
                | TyKind::FnPtr(..)
                | TyKind::Ref(..)
        )
    }

    fn is_backend_scalar_pair(&self, layout: TyAndLayout<'tcx>) -> bool {
        let kind = layout.ty.kind();
        matches!(kind, TyKind::RawPtr(pointee_ty, _) | TyKind::Ref(_, pointee_ty, _) if self.tcx.type_has_metadata(*pointee_ty, TypingEnv::fully_monomorphized()))
    }

    fn scalar_pair_element_backend_type(
        &self,
        layout: TyAndLayout<'tcx>,
        index: usize,
        immediate: bool,
    ) -> Self::Type {
        todo!()
    }
}

impl<'tcx> TypeMembershipCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn add_type_metadata(&self, _function: Self::Function, _typeid: String) {}

    fn set_type_metadata(&self, _function: Self::Function, _typeid: String) {}

    fn typeid_metadata(&self, _typeid: String) -> Option<Self::Metadata> {
        None
    }

    fn add_kcfi_type_metadata(&self, _function: Self::Function, _typeid: u32) {}

    fn set_kcfi_type_metadata(&self, _function: Self::Function, _typeid: u32) {}
}

impl<'tcx> CodegenCx<'tcx> {
    fn scalar_to_var_value(
        &self,
        cv: interpret::Scalar,
        layout: rustc_abi::Scalar,
        llty: IrVarTy,
    ) -> IrVarValue {
        match cv {
            interpret::Scalar::Int(scalar_int) => {
                // TODO: handle properly
                let value = scalar_int.to_bits_unchecked() as u64;

                IrVarValue {
                    ty: IrVarValueTy::Int(value.into()),
                    var_ty: llty,
                }
            }
            interpret::Scalar::Ptr(ptr, _size) => {
                let (prov, offset) = ptr.into_parts();
                let global_alloc = self.tcx.global_alloc(prov.alloc_id());

                match global_alloc {
                    GlobalAlloc::Function { instance } => {
                        let glb = self.get_glb(instance);

                        IrVarValue {
                            ty: IrVarValueTy::Addr(IrVarValueAddr { glb, offset: 0 }),
                            var_ty: llty,
                        }
                    }
                    GlobalAlloc::VTable(ty, raw_list) => todo!(),
                    GlobalAlloc::Static(def_id) => todo!(),
                    GlobalAlloc::Memory(const_allocation) => {
                        // FIXME: does this cause recursion if self ref (do we need to do "get or create")?
                        let glb = self.alloc_bytes_var(&const_allocation);

                        IrVarValue {
                            ty: IrVarValueTy::Addr(IrVarValueAddr { glb, offset: 0 }),
                            var_ty: llty,
                        }
                    }
                }
            }
        }
    }
}

impl<'tcx> ConstCodegenMethods for CodegenCx<'tcx> {
    fn const_null(&self, t: Self::Type) -> Self::Value {
        todo!()
    }

    fn const_undef(&self, t: Self::Type) -> Self::Value {
        self.mk_next_op(|op| op.mk_undf(t))
    }

    fn const_poison(&self, t: Self::Type) -> Self::Value {
        self.mk_next_op(|op| op.mk_undf(t))
    }

    fn const_bool(&self, val: bool) -> Self::Value {
        self.mk_next_op(|op| op.mk_cnst_int(self.type_i1(), if val { 1 } else { 0 }))
    }

    // TODO: check cast logic for all these

    fn const_i8(&self, i: i8) -> Self::Value {
        self.mk_next_op(|op| op.mk_cnst_int(self.type_i8(), i as _))
    }

    fn const_i16(&self, i: i16) -> Self::Value {
        self.mk_next_op(|op| op.mk_cnst_int(self.type_i16(), i as _))
    }

    fn const_i32(&self, i: i32) -> Self::Value {
        self.mk_next_op(|op| op.mk_cnst_int(self.type_i32(), i as _))
    }

    fn const_int(&self, t: Self::Type, i: i64) -> Self::Value {
        // what does int mean here?
        self.mk_next_op(|op| op.mk_cnst_int(t, i as _))
    }

    fn const_u8(&self, i: u8) -> Self::Value {
        self.mk_next_op(|op| op.mk_cnst_int(self.type_i8(), i as _))
    }

    fn const_u32(&self, i: u32) -> Self::Value {
        self.mk_next_op(|op| op.mk_cnst_int(self.type_i32(), i as _))
    }

    fn const_u64(&self, i: u64) -> Self::Value {
        self.mk_next_op(|op| op.mk_cnst_int(self.type_i64(), i as _))
    }

    fn const_u128(&self, i: u128) -> Self::Value {
        // TODO: broken cast
        self.mk_next_op(|op| op.mk_cnst_int(self.type_i128(), i as _))
    }

    fn const_usize(&self, i: u64) -> Self::Value {
        // TODO: ptr size
        self.mk_next_op(|op| op.mk_cnst_int(self.type_isize(), i as _))
    }

    fn const_uint(&self, t: Self::Type, i: u64) -> Self::Value {
        self.mk_next_op(|op| op.mk_cnst_int(t, i as _))
    }

    fn const_uint_big(&self, t: Self::Type, u: u128) -> Self::Value {
        self.mk_next_op(|op| op.mk_cnst_int(t, u as _))
    }

    // FIXME: more cast issues

    fn const_real(&self, t: Self::Type, val: f64) -> Self::Value {
        self.mk_next_op(|op| op.mk_cnst_float(t, val as _))
    }

    fn const_str(&self, s: &str) -> (Self::Value, Self::Value) {
        todo!()
    }

    fn const_struct(&self, elts: &[Self::Value], packed: bool) -> Self::Value {
        todo!()
    }

    fn const_vector(&self, elts: &[Self::Value]) -> Self::Value {
        todo!()
    }

    fn const_to_opt_uint(&self, v: Self::Value) -> Option<u64> {
        todo!()
    }

    fn const_to_opt_u128(&self, v: Self::Value, sign_ext: bool) -> Option<u128> {
        todo!()
    }

    fn const_data_from_alloc(&self, alloc: ConstAllocation<'_>) -> Self::Value {
        let var = self.alloc_bytes_var(&alloc);

        self.mk_next_op(|op| op.mk_addr_glb(var))
    }

    fn scalar_to_backend(
        &self,
        cv: interpret::Scalar,
        layout: rustc_abi::Scalar,
        llty: Self::Type,
    ) -> Self::Value {
        match cv {
            interpret::Scalar::Int(scalar_int) => {
                // TODO: handle properly
                let value = scalar_int.to_bits_unchecked() as u64;

                let op = self.alloc_next_op();
                op.mk_cnst_int(llty, value);
                op
            }
            interpret::Scalar::Ptr(ptr, _size) => {
                let (prov, offset) = ptr.into_parts();
                let global_alloc = self.tcx.global_alloc(prov.alloc_id());
                match global_alloc {
                    GlobalAlloc::Function { instance } => {
                        let glb = self.get_glb(instance);

                        let op = self.alloc_next_op();
                        op.mk_addr_glb(glb);
                        op
                    }
                    GlobalAlloc::VTable(ty, raw_list) => todo!(),
                    GlobalAlloc::Static(def_id) => todo!(),
                    GlobalAlloc::Memory(const_allocation) => {
                        let glb = self.alloc_bytes_var(&const_allocation);

                        let op = self.alloc_next_op();
                        op.mk_addr_glb(glb);
                        op
                    }
                }
            }
        }
    }

    fn const_ptr_byte_offset(&self, val: Self::Value, offset: rustc_abi::Size) -> Self::Value {
        let Some(offset) = NonZeroUsize::new(offset.bytes_usize()) else {
            return val;
        };

        let offset = AddrOffset::offset(val, offset);

        let op = val.stmt().alloc_op();
        op.mk_addr_offset(offset);

        op
    }
}

impl<'tcx> DebugInfoCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn create_vtable_debuginfo(
        &self,
        ty: Ty<'tcx>,
        trait_ref: Option<ty::ExistentialTraitRef<'tcx>>,
        vtable: Self::Value,
    ) {
        // TODO:
    }

    fn create_function_debug_context(
        &self,
        instance: Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        llfn: Self::Function,
        mir: &rustc_middle::mir::Body<'tcx>,
    ) -> Option<
        rustc_codegen_ssa::mir::debuginfo::FunctionDebugContext<
            'tcx,
            Self::DIScope,
            Self::DILocation,
        >,
    > {
        // TODO:
        None
    }

    fn dbg_scope_fn(
        &self,
        instance: Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        maybe_definition_llfn: Option<Self::Function>,
    ) -> Self::DIScope {
        todo!()
    }

    fn dbg_loc(
        &self,
        scope: Self::DIScope,
        inlined_at: Option<Self::DILocation>,
        span: rustc_span::Span,
    ) -> Self::DILocation {
        todo!()
    }

    fn extend_scope_to_file(
        &self,
        scope_metadata: Self::DIScope,
        file: &rustc_span::SourceFile,
    ) -> Self::DIScope {
        todo!()
    }

    fn debuginfo_finalize(&self) {
        todo!()
    }

    fn create_dbg_var(
        &self,
        variable_name: Symbol,
        variable_type: Ty<'tcx>,
        scope_metadata: Self::DIScope,
        variable_kind: rustc_codegen_ssa::mir::debuginfo::VariableKind,
        span: rustc_span::Span,
    ) -> Self::DIVariable {
        todo!()
    }
}

impl<'tcx> StaticCodegenMethods for CodegenCx<'tcx> {
    fn static_addr_of(
        &self,
        cv: Self::Value,
        align: rustc_abi::Align,
        kind: Option<&str>,
    ) -> Self::Value {
        // FIXME: we want this to deduplicate, ie search (using `cv`) if existing glb with this value exists

        // we are in awkard scenario where `Value` is an `IrOp` but this is about static constants
        // so we need to back it out into a value
        // _however_, i think this is only called after `const_data_from_alloc` or similar, so we can return the op from that
        cv
    }

    fn codegen_static(&self, def_id: DefId) {
        todo!()
    }

    fn add_used_global(&self, global: Self::Value) {
        todo!()
    }

    fn add_compiler_used_global(&self, global: Self::Value) {
        todo!()
    }
}

impl From<Linkage> for IrLinkage {
    fn from(value: Linkage) -> Self {
        match value {
            Linkage::External => IrLinkage::External,
            Linkage::Internal => IrLinkage::Internal,
            Linkage::AvailableExternally => todo!(),
            Linkage::LinkOnceAny => todo!(),
            Linkage::LinkOnceODR => todo!(),
            Linkage::WeakAny => todo!(),
            Linkage::WeakODR => todo!(),
            Linkage::ExternalWeak => todo!(),
            Linkage::Common => todo!(),
        }
    }
}

impl<'tcx> CodegenCx<'tcx> {
    // TODO: use smallvec
    fn abi_map_ty(&self, arg: &ArgAbi<'tcx, Ty<'tcx>>) -> SmallVec<[IrVarTy; 2]> {
        let ty = self.backend_type(arg.layout);

        match &arg.mode {
            PassMode::Ignore => SmallVec::default(),
            PassMode::Direct(arg_attributes) => smallvec![ty],
            PassMode::Pair(arg_attributes, arg_attributes1) => {
                let Some(IrVarTyAggregate {
                    ty: IrVarTyAggregateTy::Struct,
                    fields: fields @ [_, _],
                    ..
                }) = ty.aggregate()
                else {
                    bug!("expected Pair to be struct aggregate with two fields");
                };

                SmallVec::from_slice(fields)
            }
            PassMode::Cast { pad_i32, cast } => todo!(),
            PassMode::Indirect {
                attrs,
                meta_attrs,
                on_stack,
            } => todo!(),
        }
    }

    fn abi_fn_ty<'a: 'tcx>(&self, fn_abi: &FnAbi<'a, Ty<'tcx>>) -> (Vec<IrVarTy>, IrVarTy) {
        let params = &fn_abi.args;
        let ret = &fn_abi.ret;

        let params = params
            .iter()
            .flat_map(|arg| self.abi_map_ty(arg))
            .collect::<Vec<_>>();

        let ret = self.backend_type(ret.layout);

        (params, ret)
    }

    fn abi_of(&self, instance: Instance<'tcx>) -> (&FnAbi<'_, Ty<'tcx>>, Vec<IrVarTy>, IrVarTy) {
        let fn_abi = self.fn_abi_of_instance(instance, ty::List::empty());
        let (params, ret) = self.abi_fn_ty(fn_abi);
        (fn_abi, params, ret)
    }

    pub fn declare_simple_fn(
        &self,
        defined: bool,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str,
        params: &[IrVarTy],
        ret: &IrVarTy,
    ) -> IrGlb {
        // no `instance` param
        if let Some(glb) = self.try_get_glb_by_name(symbol_name) {
            return glb;
        }

        // FIXME: variadic flag for variadic fn
        let fun_ty = self.unit.var_ty_func(params, ret, IrVarTyFuncFlags::None);

        // TODO: visibility
        let glb = if defined {
            let glb = self
                .unit
                .add_global_def_func(fun_ty, symbol_name, linkage.into());
            self.build_fn_params(glb.func(), &params[..]);
            glb
        } else {
            self.unit
                .add_global_undef_func(fun_ty, symbol_name, linkage.into())
        };

        self.glb_map
            .borrow_mut()
            .insert(symbol_name.to_string(), glb);

        glb
    }

    pub fn declare_fn(
        &self,
        instance: Instance<'tcx>,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str,
        params: &[IrVarTy],
        ret: &IrVarTy,
    ) -> IrGlb {
        if let Some(glb) = self.try_get_glb_by_name(symbol_name) {
            return glb;
        }

        // FIXME: variadic flag for variadic fn
        let fun_ty = self.unit.var_ty_func(params, ret, IrVarTyFuncFlags::None);

        // HACK: these arent working properly
        let is_undef = symbol_name.contains("lang_internal");

        // TODO: visibility
        let glb = if is_undef || self.tcx.is_foreign_item(instance.def_id()) {
            self.unit
                .add_global_undef_func(fun_ty, symbol_name, linkage.into())
        } else {
            let glb = self
                .unit
                .add_global_def_func(fun_ty, symbol_name, linkage.into());
            self.build_fn_params(glb.func(), &params[..]);
            glb
        };

        self.glb_map
            .borrow_mut()
            .insert(symbol_name.to_string(), glb);

        glb
    }

    pub fn build_fn_params(&self, fun: IrFunc, params: &[IrVarTy]) -> IrBasicBlock {
        // FIXME: recompute params in `declare_fn` and here
        let bb = fun.alloc_basicblock();

        if !params.is_empty() {
            let param_stmt = fun.mk_param_stmt();

            for &param in params {
                if param.is_aggregate() {
                    let lcl = fun.add_local(param);

                    let op = param_stmt.alloc_op();
                    op.mk_lcl_param(param);
                } else {
                    let op = param_stmt.alloc_op();
                    op.mk_mov_param(param);
                }
            }
        }

        bb
    }
}

impl<'tcx> PreDefineCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn predefine_static(
        &self,
        def_id: DefId,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str,
    ) {
        todo!()
    }

    fn predefine_fn(
        &self,
        instance: Instance<'tcx>,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str,
    ) {
        let (_, params, ret) = self.abi_of(instance);

        self.declare_fn(
            instance,
            linkage,
            visibility,
            symbol_name,
            &params[..],
            &ret,
        );
    }
}

impl<'tcx> CodegenCx<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>, cg_unit: Option<&'tcx CodegenUnit<'tcx>>) -> Self {
        let arena = ArenaAlloc::new(b"codegen-cx");
        let unit = IrUnit::new(*arena.as_ref());
        let glb_map = RefCell::new(FxHashMap::default());
        let vtables = RefCell::new(FxHashMap::default());

        Self {
            tcx,
            arena,
            cg_unit,
            unit,
            glb_map,
            vtables,
            cur_bb: RefCell::new(None),
        }
    }

    pub(crate) fn module(&self) -> JccModule {
        JccModule { unit: self.unit }
    }
}
