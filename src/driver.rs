use std::{
    cell::RefCell,
    num::NonZeroUsize,
    ops::Range,
};

use rustc_abi::{FieldsShape, HasDataLayout};
use rustc_codegen_ssa::traits::{
        AsmCodegenMethods, BackendTypes, BaseTypeCodegenMethods,
        ConstCodegenMethods, DebugInfoCodegenMethods,
        LayoutTypeCodegenMethods, MiscCodegenMethods, PreDefineCodegenMethods,
        StaticCodegenMethods, TypeMembershipCodegenMethods,
    };
use rustc_const_eval::interpret::{
    self, Allocation, ConstAllocation, GlobalAlloc, InitChunk, read_target_uint,
};
use rustc_data_structures::fx::FxHashMap;
use rustc_middle::{
    bug,
    mir::mono::{CodegenUnit, Linkage, Visibility},
    ty::{
        self, ExistentialTraitRef, Instance, Ty, TyCtxt, TyKind, TypingEnv,
        layout::{FnAbiOf, FnAbiOfHelpers, HasTyCtxt, HasTypingEnv, LayoutOfHelpers, TyAndLayout},
    },
};
use rustc_session::Session;
use rustc_span::{Symbol, def_id::DefId};
use rustc_target::callconv::{ArgAbi, CastTarget, FnAbi, PassMode};
use rustc_type_ir::{FloatTy, IntTy, UintTy};
use smallvec::{SmallVec, smallvec};

use crate::jcc::{
        alloc::{ArenaAlloc, ArenaAllocRef},
        ir::{
            AddrOffset, IrBasicBlock, IrBytes, IrFloatTy, IrFunc, IrGlb, IrIntTy, IrOp,
            IrUnit, IrVarTy, IrVarTyAggregate, IrVarTyAggregateTy, IrVarTyFuncFlags,
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

fn ty_to_jcc_ty<'tcx>(tcx: TyCtxt<'tcx>, unit: &IrUnit, ty: &TyAndLayout<'tcx>) -> IrVarTy {
    let TyAndLayout { ty, layout } = ty;

    match ty.kind() {
        TyKind::Never => unit.var_ty_none(),
        TyKind::Tuple(tys) if tys.len() == 0 => unit.var_ty_none(),
        TyKind::Bool => unit.var_ty_integer(IrIntTy::I1),
        TyKind::Int(IntTy::I8) | TyKind::Uint(UintTy::U8) => unit.var_ty_integer(IrIntTy::I8),
        TyKind::Char => unit.var_ty_integer(IrIntTy::I32),
        TyKind::Int(IntTy::I16) | TyKind::Uint(UintTy::U16) => unit.var_ty_integer(IrIntTy::I16),
        TyKind::Int(IntTy::I32) | TyKind::Uint(UintTy::U32) => unit.var_ty_integer(IrIntTy::I32),
        TyKind::Int(IntTy::I64) | TyKind::Uint(UintTy::U64) => unit.var_ty_integer(IrIntTy::I64),
        // FIXME: should be ptr-sized int
        TyKind::Int(IntTy::Isize) | TyKind::Uint(UintTy::Usize) => unit.var_ty_pointer(),
        TyKind::Int(IntTy::I128) | TyKind::Uint(UintTy::U128) => unit.var_ty_integer(IrIntTy::I128),
        TyKind::Float(FloatTy::F16) => unit.var_ty_float(IrFloatTy::F16),
        TyKind::Float(FloatTy::F32) => unit.var_ty_float(IrFloatTy::F32),
        TyKind::Float(FloatTy::F64) => unit.var_ty_float(IrFloatTy::F64),
        TyKind::Float(FloatTy::F128) => todo!("f128"),
        TyKind::FnPtr(..) => unit.var_ty_pointer(),
        TyKind::RawPtr(pointee_ty, _) | TyKind::Ref(_, pointee_ty, _) => {
            if tcx.type_has_metadata(*pointee_ty, TypingEnv::fully_monomorphized()) {
                // is this correct? metadata type = { ptr, ptr }?

                let FieldsShape::Arbitrary {
                    offsets,
                    memory_index,
                } = layout.fields()
                else {
                    bug!()
                };

                dbg!(unit.var_ty_fat_pointer())
            } else {
                unit.var_ty_pointer()
            }
        }
        _ => panic!("bad ty {ty:?}, {:?}", ty.kind().typed_debug()),
    }
}

pub(crate) struct JccModule {
    pub(crate) unit: IrUnit,
}

unsafe impl Send for JccModule {}
unsafe impl Sync for JccModule {}

pub struct CodegenCx<'tcx> {
    pub tcx: TyCtxt<'tcx>,
    pub unit: IrUnit,
    pub arena: ArenaAlloc,

    pub fn_map: RefCell<FxHashMap<Instance<'tcx>, IrGlb>>,
    pub vtables: RefCell<FxHashMap<(Ty<'tcx>, Option<ExistentialTraitRef<'tcx>>), IrOp>>,

    pub cur_bb: RefCell<Option<IrBasicBlock>>,
}

impl<'tcx> CodegenCx<'tcx> {
    pub fn alloc_next_op(&self) -> IrOp {
        let bb = self.cur_bb.borrow().unwrap();
        let stmt = bb.alloc_stmt();
        stmt.alloc_op()
    }

    pub fn mk_next_op(&self, mk: impl FnOnce(IrOp)) -> IrOp {
        let bb = self.cur_bb.borrow().unwrap();
        let stmt = bb.alloc_stmt();
        let op = stmt.alloc_op();

        mk(op);

        op
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

        self.fn_map.borrow_mut().insert(instance, glb);
        glb
    }

    fn alloc_bytes_var(&self, alloc: &ConstAllocation<'_>) -> IrGlb {
        let arena = self.arena.as_ref();
        let alloc = alloc.inner();

        let unit = self.unit;
        let ty = unit.var_ty_bytes(alloc.len());

        let glb = unit.add_global_def_var(ty, None);
        let var = glb.var();

        let mut llvals = Vec::with_capacity_in(alloc.provenance().ptrs().len() + 1, arena);

        let dl = self.data_layout();
        let pointer_size = dl.pointer_size.bytes() as usize;

        // Note: this function may call `inspect_with_uninit_and_ptr_outside_interpreter`, so `range`
        // must be within the bounds of `alloc` and not contain or overlap a pointer provenance.
        fn append_chunks_of_init_and_uninit_bytes<'a, 'b>(
            cx: &'a CodegenCx<'b>,
            llvals: &mut Vec<IrBytes<'a>, &ArenaAllocRef>,
            alloc: &'a Allocation,
            range: Range<usize>,
        ) {
            let offset = range.start;
            let chunks = alloc.init_mask().range_as_init_chunks(range.clone().into());

            let chunk_to_llval = move |chunk| match chunk {
                InitChunk::Init(range) => {
                    let range = (range.start.bytes() as usize)..(range.end.bytes() as usize);
                    let bytes = alloc.inspect_with_uninit_and_ptr_outside_interpreter(range);

                    Some(IrBytes::new(offset, bytes))
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
                // llvals.push(cx.const_bytes(bytes));
                llvals.push(IrBytes::new(offset, bytes));
            }
        }

        let next_offset = 0;
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

            todo!("ptr relocs");
            // self.scalar_to_var_value(
            //     interpret::Scalar::from_pointer(
            //         Pointer::new(prov, rustc_abi::Size::from_bytes(ptr_offset)),
            //         &self.tcx,
            //     ),
            //     Scalar::Initialized {
            //         value: Primitive::Pointer(address_space),
            //         valid_range: WrappingRange::full(dl.pointer_size),
            //     },
            //     self.type_ptr_ext(address_space),
            // );

            next_offset = offset + pointer_size;
        }

        if alloc.len() >= next_offset {
            let range = next_offset..alloc.len();
            // This `inspect` is okay since we have check that it is after all provenance, it is
            // within the bounds of the allocation, and it doesn't affect interpreter execution (we
            // inspect the result after interpreter execution).
            append_chunks_of_init_and_uninit_bytes(self, &mut llvals, alloc, range);
        }

        var.mk_const_bytes(ty, &llvals[..]);
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
    pub fn type_i1(&self) -> IrVarTy {
        IrVarTy::ty_i1()
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
        &self.tcx.sess
    }

    fn codegen_unit(&self) -> &'tcx CodegenUnit<'tcx> {
        todo!()
    }

    fn set_frame_pointer_type(&self, llfn: Self::Function) {
        todo!()
    }

    fn apply_target_cpu_attr(&self, llfn: Self::Function) {
        todo!()
    }

    fn declare_c_main(&self, fn_type: Self::Type) -> Option<Self::Function> {
        todo!()
    }
}

impl<'tcx> LayoutTypeCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn backend_type(&self, layout: TyAndLayout<'tcx>) -> Self::Type {
        ty_to_jcc_ty(self.tcx, &self.unit, &layout)
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
        todo!()
    }

    fn reg_backend_type(&self, ty: &rustc_abi::Reg) -> Self::Type {
        todo!()
    }

    fn immediate_backend_type(&self, layout: TyAndLayout<'tcx>) -> Self::Type {
        ty_to_jcc_ty(self.tcx, &self.unit, &layout)
    }

    fn is_backend_immediate(&self, layout: TyAndLayout<'tcx>) -> bool {
        match layout.ty.kind() {
            TyKind::Bool
            | TyKind::Char
            | TyKind::Int(..)
            | TyKind::Uint(..)
            | TyKind::Float(..)
            | TyKind::RawPtr(..)
            | TyKind::Ref(..) => true,
            _ => false,
        }
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

// impl<'tcx> CodegenCx<'tcx> {
// fn scalar_to_var_value(
//     &self,
//     cv: interpret::Scalar,
//     layout: rustc_abi::Scalar,
//     llty: IrVarTy,
// ) -> () {
//     match cv {
//         interpret::Scalar::Int(scalar_int) => {
//             // TODO: handle properly
//             let value = scalar_int.to_bits_unchecked() as u64;

//             let op = self.alloc_next_op();
//             op.mk_cnst_int(llty, value);
//             op
//         }
//         interpret::Scalar::Ptr(ptr, _size) => {
//             let (prov, offset) = ptr.into_parts();
//             let global_alloc = self.tcx.global_alloc(prov.alloc_id());
//             match global_alloc {
//                 GlobalAlloc::Function { instance } => {
//                     let glb = self.get_glb(instance);

//                     let op = self.alloc_next_op();
//                     op.mk_addr_glb(glb);
//                     op
//                 }
//                 GlobalAlloc::VTable(ty, raw_list) => todo!(),
//                 GlobalAlloc::Static(def_id) => todo!(),
//                 GlobalAlloc::Memory(const_allocation) => {
//                     let glb = self.alloc_bytes_var(&const_allocation);
//                 }
//             }
//         }
//     }
// }
// }

impl<'tcx> ConstCodegenMethods for CodegenCx<'tcx> {
    fn const_null(&self, t: Self::Type) -> Self::Value {
        todo!()
    }

    fn const_undef(&self, t: Self::Type) -> Self::Value {
        todo!()
    }

    fn const_poison(&self, t: Self::Type) -> Self::Value {
        todo!()
    }

    fn const_bool(&self, val: bool) -> Self::Value {
        todo!()
    }

    fn const_i8(&self, i: i8) -> Self::Value {
        todo!()
    }

    fn const_i16(&self, i: i16) -> Self::Value {
        todo!()
    }

    fn const_i32(&self, i: i32) -> Self::Value {
        todo!()
    }

    fn const_int(&self, t: Self::Type, i: i64) -> Self::Value {
        todo!()
    }

    fn const_u8(&self, i: u8) -> Self::Value {
        todo!()
    }

    fn const_u32(&self, i: u32) -> Self::Value {
        todo!()
    }

    fn const_u64(&self, i: u64) -> Self::Value {
        todo!()
    }

    fn const_u128(&self, i: u128) -> Self::Value {
        todo!()
    }

    fn const_usize(&self, i: u64) -> Self::Value {
        // TODO: ptr size
        let ty = self.type_isize();
        let op = self.alloc_next_op();
        op.mk_cnst_int(ty, i);
        op
    }

    fn const_uint(&self, t: Self::Type, i: u64) -> Self::Value {
        todo!()
    }

    fn const_uint_big(&self, t: Self::Type, u: u128) -> Self::Value {
        todo!()
    }

    fn const_real(&self, t: Self::Type, val: f64) -> Self::Value {
        todo!()
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

    fn const_data_from_alloc(&self, alloc: interpret::ConstAllocation<'_>) -> Self::Value {
        todo!()
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
        todo!()
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
        todo!()
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
                    fields: &[_0, _1],
                    ..
                }) = ty.aggregate()
                else {
                    bug!("expected Pair to be struct aggregate with two fields");
                };

                smallvec![_0, _1]
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

    fn declare_fn(
        &self,
        instance: Instance<'tcx>,
        linkage: Linkage,
        visibility: Visibility,
        symbol_name: &str,
        params: &[IrVarTy],
        ret: &IrVarTy,
    ) -> IrGlb {
        if let Some(glb) = self.fn_map.borrow().get(&instance).copied() {
            return glb;
        }

        // FIXME: variadic flag for variadic fn
        let fun_ty = self
            .unit
            .var_ty_func(&params[..], &ret, IrVarTyFuncFlags::None);

        // TODO: visibility
        if self.tcx.is_foreign_item(instance.def_id()) {
            self.unit.add_global_undef_func(fun_ty, symbol_name)
        } else {
            self.unit.add_global_def_func(fun_ty, symbol_name)
        }
        // match linkage {
        //     Linkage::External
        //     | Linkage::AvailableExternally
        //     | Linkage::LinkOnceAny
        //     | Linkage::LinkOnceODR
        //     | Linkage::WeakAny
        //     | Linkage::WeakODR
        //     | Linkage::ExternalWeak => unit.add_global_undef_func(fun_ty, symbol_name),
        //     Linkage::Internal | Linkage::Common => unit.add_global_def_func(fun_ty, symbol_name),
        // }
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
        let glb = self.declare_fn(
            instance,
            linkage,
            visibility,
            symbol_name,
            &params[..],
            &ret,
        );
        let fun = glb.func();

        let unit = self.unit;

        if glb.is_def() {
            // FIXME: recompute params in `declare_fn` and here
            let bb = fun.alloc_basicblock();

            if params.len() > 0 {
                let param_stmt = fun.mk_param_stmt();

                for param in params {
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

            *self.cur_bb.borrow_mut() = Some(bb);
        }

        self.fn_map.borrow_mut().insert(instance, glb);
    }
}

impl<'tcx> CodegenCx<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>) -> Self {
        let arena = ArenaAlloc::new(b"codegen-cx");
        let unit = IrUnit::new(*arena.as_ref());
        let fn_map = RefCell::new(FxHashMap::default());
        let vtables = RefCell::new(FxHashMap::default());

        Self {
            tcx,
            arena,
            unit,
            fn_map,
            vtables,
            cur_bb: RefCell::new(None),
        }
    }

    pub(crate) fn module(&self) -> JccModule {
        JccModule { unit: self.unit }
    }
}
