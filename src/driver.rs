use std::{
    cell::RefCell,
    ffi::{self, CString},
    num::NonZeroUsize,
    ptr,
};

use rustc_abi::HasDataLayout;
use rustc_ast::token::NtPatKind::PatWithOr;
use rustc_codegen_ssa::{
    CrateInfo,
    mono_item::MonoItemExt,
    traits::{
        AsmCodegenMethods, BackendTypes, BaseTypeCodegenMethods, ConstCodegenMethods,
        DebugInfoCodegenMethods, ExtraBackendMethods, LayoutTypeCodegenMethods, MiscCodegenMethods,
        PreDefineCodegenMethods, StaticCodegenMethods, TypeMembershipCodegenMethods,
    },
};
use rustc_const_eval::interpret;
use rustc_data_structures::{fx::FxHashMap, profiling::SelfProfilerRef};
use rustc_metadata::EncodedMetadata;
use rustc_middle::{
    mir::{
        BasicBlock, BasicBlockData, Statement, StatementKind, TerminatorKind,
        mono::{CodegenUnit, MonoItem},
    },
    ty::{
        self, EarlyBinder, ExistentialTraitRef, Instance, Ty, TyCtxt, TyKind, TypingEnv,
        layout::{FnAbiOfHelpers, HasTyCtxt, HasTypingEnv, LayoutOfHelpers, TyAndLayout},
    },
};
use rustc_session::Session;
use rustc_span::{Symbol, def_id::DefId};
use rustc_target::callconv::CastTarget;
use rustc_type_ir::{FloatTy, IntTy, UintTy};

use crate::{
    jcc::{
        alloc::ArenaAllocRef,
        ir::{
            self, AddrOffset, IrBasicBlock, IrFloatTy, IrFunc, IrGlb, IrIntTy, IrOp, IrUnit,
            IrVarTy, IrVarTyFuncFlags,
        },
    },
    jcc_sys::{self, IR_FUNC_FLAG_NONE},
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

fn ty_to_jcc_ty<'tcx>(tcx: TyCtxt<'tcx>, unit: &IrUnit, ty: &Ty<'tcx>) -> IrVarTy {
    match ty.kind() {
        TyKind::Never => unit.none(),
        TyKind::Tuple(tys) if tys.len() == 0 => unit.none(),
        TyKind::Bool => unit.integer(IrIntTy::I1),
        TyKind::Int(IntTy::I8) | TyKind::Uint(UintTy::U8) => unit.integer(IrIntTy::I8),
        TyKind::Char => unit.integer(IrIntTy::I32),
        TyKind::Int(IntTy::I16) | TyKind::Uint(UintTy::U16) => unit.integer(IrIntTy::I16),
        TyKind::Int(IntTy::I32) | TyKind::Uint(UintTy::U32) => unit.integer(IrIntTy::I32),
        TyKind::Int(IntTy::I64) | TyKind::Uint(UintTy::U64) => unit.integer(IrIntTy::I64),
        // FIXME: should be ptr-sized int
        TyKind::Int(IntTy::Isize) | TyKind::Uint(UintTy::Usize) => unit.pointer(),
        TyKind::Int(IntTy::I128) | TyKind::Uint(UintTy::U128) => unit.integer(IrIntTy::I128),
        TyKind::Float(FloatTy::F16) => unit.float(IrFloatTy::F16),
        TyKind::Float(FloatTy::F32) => unit.float(IrFloatTy::F32),
        TyKind::Float(FloatTy::F64) => unit.float(IrFloatTy::F64),
        TyKind::Float(FloatTy::F128) => todo!("f128"),
        TyKind::FnPtr(..) => unit.pointer(),
        TyKind::RawPtr(pointee_ty, _) | TyKind::Ref(_, pointee_ty, _) => {
            if tcx.type_has_metadata(*pointee_ty, TypingEnv::fully_monomorphized()) {
                panic!("ptr with metadata");
            } else {
                unit.pointer()
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
        todo!()
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
    type FnAbiOfResult = &'tcx rustc_target::callconv::FnAbi<'tcx, Ty<'tcx>>;

    fn handle_fn_abi_err(
        &self,
        err: ty::layout::FnAbiError<'tcx>,
        span: rustc_span::Span,
        fn_abi_request: ty::layout::FnAbiRequest<'tcx>,
    ) -> <Self::FnAbiOfResult as ty::layout::MaybeResult<
        &'tcx rustc_target::callconv::FnAbi<'tcx, Ty<'tcx>>,
    >>::Error {
        todo!()
    }
}

impl<'tcx> BaseTypeCodegenMethods for CodegenCx<'tcx> {
    fn type_i8(&self) -> Self::Type {
        todo!()
    }

    fn type_i16(&self) -> Self::Type {
        todo!()
    }

    fn type_i32(&self) -> Self::Type {
        todo!()
    }

    fn type_i64(&self) -> Self::Type {
        todo!()
    }

    fn type_i128(&self) -> Self::Type {
        todo!()
    }

    fn type_isize(&self) -> Self::Type {
        todo!()
    }

    fn type_f16(&self) -> Self::Type {
        todo!()
    }

    fn type_f32(&self) -> Self::Type {
        todo!()
    }

    fn type_f64(&self) -> Self::Type {
        todo!()
    }

    fn type_f128(&self) -> Self::Type {
        todo!()
    }

    fn type_array(&self, ty: Self::Type, len: u64) -> Self::Type {
        todo!()
    }

    fn type_func(&self, args: &[Self::Type], ret: Self::Type) -> Self::Type {
        todo!()
    }

    fn type_kind(&self, ty: Self::Type) -> rustc_codegen_ssa::common::TypeKind {
        todo!()
    }

    fn type_ptr(&self) -> Self::Type {
        todo!()
    }

    fn type_ptr_ext(&self, address_space: rustc_abi::AddressSpace) -> Self::Type {
        todo!()
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
        todo!()
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
        self.fn_map.borrow()[&instance].func()
    }

    fn get_fn_addr(&self, instance: Instance<'tcx>) -> Self::Value {
        todo!()
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
        todo!()
    }

    fn cast_backend_type(&self, ty: &CastTarget) -> Self::Type {
        todo!()
    }

    fn fn_decl_backend_type(
        &self,
        fn_abi: &rustc_target::callconv::FnAbi<'tcx, Ty<'tcx>>,
    ) -> Self::Type {
        todo!()
    }

    fn fn_ptr_backend_type(
        &self,
        fn_abi: &rustc_target::callconv::FnAbi<'tcx, Ty<'tcx>>,
    ) -> Self::Type {
        todo!()
    }

    fn reg_backend_type(&self, ty: &rustc_abi::Reg) -> Self::Type {
        todo!()
    }

    fn immediate_backend_type(&self, layout: TyAndLayout<'tcx>) -> Self::Type {
        ty_to_jcc_ty(self.tcx, &self.unit, &layout.ty)
    }

    fn is_backend_immediate(&self, layout: TyAndLayout<'tcx>) -> bool {
        match layout.ty.kind() {
            TyKind::Bool
            | TyKind::Char
            | TyKind::Int(..)
            | TyKind::Uint(..)
            | TyKind::Float(..) => true,
            _ => false,
        }
    }

    fn is_backend_scalar_pair(&self, layout: TyAndLayout<'tcx>) -> bool {
        false
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
        todo!()
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
            interpret::Scalar::Ptr(pointer, _) => todo!(),
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
        fn_abi: &rustc_target::callconv::FnAbi<'tcx, Ty<'tcx>>,
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
        fn_abi: &rustc_target::callconv::FnAbi<'tcx, Ty<'tcx>>,
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

impl<'tcx> PreDefineCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn predefine_static(
        &self,
        def_id: DefId,
        linkage: rustc_middle::mir::mono::Linkage,
        visibility: rustc_middle::mir::mono::Visibility,
        symbol_name: &str,
    ) {
        todo!()
    }

    fn predefine_fn(
        &self,
        instance: Instance<'tcx>,
        linkage: rustc_middle::mir::mono::Linkage,
        visibility: rustc_middle::mir::mono::Visibility,
        symbol_name: &str,
    ) {
        let kind = instance
            .ty(self.tcx, TypingEnv::fully_monomorphized())
            .kind();

        let def_id = instance.def.def_id();
        let param_env = self.tcx.param_env(def_id);

        let sig = kind.fn_sig(self.tcx).skip_binder();

        let params = sig.inputs();
        let ret = sig.output();

        let unit = self.unit;

        let params = params
            .iter()
            .map(|ty| ty_to_jcc_ty(self.tcx, &unit, ty))
            .collect::<Vec<_>>();

        let ret = ty_to_jcc_ty(self.tcx, &unit, &ret);

        // FIXME: variadic flag for variadic fn
        let fun_ty = unit.func(&params[..], &ret, IrVarTyFuncFlags::None);

        let glb = unit.add_global_def_func(fun_ty, symbol_name);
        let fun = glb.func();

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
        self.fn_map.borrow_mut().insert(instance, glb);
    }
}

impl<'tcx> CodegenCx<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>) -> Self {
        let unit = IrUnit::new();
        let fn_map = RefCell::new(FxHashMap::default());
        let vtables = RefCell::new(FxHashMap::default());

        Self {
            tcx,
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
