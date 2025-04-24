use std::{
    ffi::{self, CString},
    ptr,
};

use rustc_abi::HasDataLayout;
use rustc_codegen_ssa::{
    CrateInfo,
    mono_item::MonoItemExt,
    traits::{
        AsmCodegenMethods, BackendTypes, BaseTypeCodegenMethods, ConstCodegenMethods,
        DebugInfoCodegenMethods, ExtraBackendMethods, LayoutTypeCodegenMethods, MiscCodegenMethods,
        PreDefineCodegenMethods, StaticCodegenMethods, TypeMembershipCodegenMethods,
    },
};
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_metadata::EncodedMetadata;
use rustc_middle::{
    mir::{BasicBlock, BasicBlockData, Statement, StatementKind, TerminatorKind, mono::MonoItem},
    ty::{
        EarlyBinder, Instance, Ty, TyCtxt, TyKind, TypingEnv,
        layout::{FnAbiOfHelpers, HasTyCtxt, HasTypingEnv, LayoutOfHelpers},
    },
};
use rustc_span::{Symbol, def_id::DefId};
use rustc_type_ir::{FloatTy, IntTy, UintTy};

use crate::{
    jcc::{self, ArenaAlloc, IrBasicBlock, IrFunc, IrOp, IrVarTy},
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

fn ty_to_jcc_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: &Ty<'tcx>) -> jcc_sys::ir_var_ty {
    unsafe {
        // TODO: ptr-size ty
        let pointer_ty = jcc_sys::IR_VAR_TY_I64;

        match ty.kind() {
            TyKind::Never => jcc_sys::IR_VAR_TY_NONE,
            TyKind::Tuple(tys) if tys.len() == 0 => jcc_sys::IR_VAR_TY_NONE,
            TyKind::Bool => jcc_sys::IR_VAR_TY_I1,
            TyKind::Int(IntTy::I8) | TyKind::Uint(UintTy::U8) => jcc_sys::IR_VAR_TY_I8,
            TyKind::Char => jcc_sys::IR_VAR_TY_I32,
            TyKind::Int(IntTy::I16) | TyKind::Uint(UintTy::U16) => jcc_sys::IR_VAR_TY_I16,
            TyKind::Int(IntTy::I32) | TyKind::Uint(UintTy::U32) => jcc_sys::IR_VAR_TY_I32,
            TyKind::Int(IntTy::I64) | TyKind::Uint(UintTy::U64) => jcc_sys::IR_VAR_TY_I64,
            TyKind::Int(IntTy::Isize) | TyKind::Uint(UintTy::Usize) => pointer_ty,
            TyKind::Int(IntTy::I128) | TyKind::Uint(UintTy::U128) => jcc_sys::IR_VAR_TY_I128,
            TyKind::Float(FloatTy::F16) => jcc_sys::IR_VAR_TY_F16,
            TyKind::Float(FloatTy::F32) => jcc_sys::IR_VAR_TY_F32,
            TyKind::Float(FloatTy::F64) => jcc_sys::IR_VAR_TY_F64,
            TyKind::Float(FloatTy::F128) => todo!("f128"),
            TyKind::FnPtr(..) => pointer_ty,
            TyKind::RawPtr(pointee_ty, _) | TyKind::Ref(_, pointee_ty, _) => {
                if tcx.type_has_metadata(*pointee_ty, TypingEnv::fully_monomorphized()) {
                    panic!("ptr with metadata");
                } else {
                    pointer_ty
                }
            }
            _ => panic!("bad ty {ty:?}, {:?}", ty.kind().typed_debug()),
        }
    }
}

pub(crate) struct JccModule {
    pub(crate) unit: *mut jcc_sys::ir_unit,
}

unsafe impl Send for JccModule {}
unsafe impl Sync for JccModule {}

pub struct CodegenCx<'tcx> {
    arena: ArenaAlloc,
    tcx: TyCtxt<'tcx>,
    unit: *mut jcc_sys::ir_unit,
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
    fn tcx(&self) -> rustc_middle::ty::TyCtxt<'tcx> {
        self.tcx
    }
}

impl<'tcx> HasTypingEnv<'tcx> for CodegenCx<'tcx> {
    fn typing_env(&self) -> rustc_middle::ty::TypingEnv<'tcx> {
        todo!()
    }
}

impl<'tcx> LayoutOfHelpers<'tcx> for CodegenCx<'tcx> {
    type LayoutOfResult = rustc_middle::ty::layout::TyAndLayout<'tcx>;

    fn handle_layout_err(
        &self,
        err: rustc_middle::ty::layout::LayoutError<'tcx>,
        span: rustc_span::Span,
        ty: Ty<'tcx>,
    ) -> <Self::LayoutOfResult as rustc_middle::ty::layout::MaybeResult<
        rustc_middle::ty::layout::TyAndLayout<'tcx>,
    >>::Error {
        todo!()
    }
}

impl<'tcx> FnAbiOfHelpers<'tcx> for CodegenCx<'tcx> {
    type FnAbiOfResult = &'tcx rustc_target::callconv::FnAbi<'tcx, Ty<'tcx>>;

    fn handle_fn_abi_err(
        &self,
        err: rustc_middle::ty::layout::FnAbiError<'tcx>,
        span: rustc_span::Span,
        fn_abi_request: rustc_middle::ty::layout::FnAbiRequest<'tcx>,
    ) -> <Self::FnAbiOfResult as rustc_middle::ty::layout::MaybeResult<
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
            (
                Ty<'tcx>,
                Option<rustc_middle::ty::ExistentialTraitRef<'tcx>>,
            ),
            Self::Value,
        >,
    > {
        todo!()
    }

    fn get_fn(&self, instance: Instance<'tcx>) -> Self::Function {
        todo!()
    }

    fn get_fn_addr(&self, instance: Instance<'tcx>) -> Self::Value {
        todo!()
    }

    fn eh_personality(&self) -> Self::Value {
        todo!()
    }

    fn sess(&self) -> &rustc_session::Session {
        todo!()
    }

    fn codegen_unit(&self) -> &'tcx rustc_middle::mir::mono::CodegenUnit<'tcx> {
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
    fn backend_type(&self, layout: rustc_middle::ty::layout::TyAndLayout<'tcx>) -> Self::Type {
        todo!()
    }

    fn cast_backend_type(&self, ty: &rustc_target::callconv::CastTarget) -> Self::Type {
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

    fn immediate_backend_type(
        &self,
        layout: rustc_middle::ty::layout::TyAndLayout<'tcx>,
    ) -> Self::Type {
        todo!()
    }

    fn is_backend_immediate(&self, layout: rustc_middle::ty::layout::TyAndLayout<'tcx>) -> bool {
        todo!()
    }

    fn is_backend_scalar_pair(&self, layout: rustc_middle::ty::layout::TyAndLayout<'tcx>) -> bool {
        todo!()
    }

    fn scalar_pair_element_backend_type(
        &self,
        layout: rustc_middle::ty::layout::TyAndLayout<'tcx>,
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

    fn const_data_from_alloc(
        &self,
        alloc: rustc_const_eval::interpret::ConstAllocation<'_>,
    ) -> Self::Value {
        todo!()
    }

    fn scalar_to_backend(
        &self,
        cv: rustc_const_eval::interpret::Scalar,
        layout: rustc_abi::Scalar,
        llty: Self::Type,
    ) -> Self::Value {
        todo!()
    }

    fn const_ptr_byte_offset(&self, val: Self::Value, offset: rustc_abi::Size) -> Self::Value {
        todo!()
    }
}

impl<'tcx> DebugInfoCodegenMethods<'tcx> for CodegenCx<'tcx> {
    fn create_vtable_debuginfo(
        &self,
        ty: Ty<'tcx>,
        trait_ref: Option<rustc_middle::ty::ExistentialTraitRef<'tcx>>,
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
        todo!()
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
        todo!()
    }
}

impl<'tcx> CodegenCx<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>) -> Self {
        let arena = ArenaAlloc::new();
        let unit = jcc_sys::ir_unit {
            arena: arena.as_ptr(),
            target: unsafe { &jcc_sys::AARCH64_MACOS_TARGET },
            ..Default::default()
        };
        let unit = Box::into_raw(Box::new(unit));

        Self { tcx, arena, unit }
    }

    pub(crate) fn codegen_stmt(&mut self, inst: &Instance<'tcx>, stmt: &Statement<'tcx>) {
        match &stmt.kind {
            StatementKind::Assign(assg) => {
                let (place, value) = assg.as_ref();
            }
            StatementKind::SetDiscriminant {
                place,
                variant_index,
            } => todo!(),
            StatementKind::Deinit(place) => todo!(),
            StatementKind::Intrinsic(non_diverging_intrinsic) => todo!(),
            _ => {}
        }
    }

    pub(crate) fn codegen_basic_block(
        &mut self,
        inst: &Instance<'tcx>,
        fun: *mut jcc_sys::ir_func,
        jcc_blocks: &[*mut jcc_sys::ir_basicblock],
        idx: usize,
        block: &BasicBlockData<'tcx>,
    ) {
        let jcc_bb = jcc_blocks[idx];

        for stmt in block.statements.iter() {
            self.codegen_stmt(inst, stmt);
        }

        let terminator = block.terminator();
        match &terminator.kind {
            TerminatorKind::Goto { target } => {
                let to = jcc_blocks[target.index()];
                unsafe {
                    jcc_sys::ir_make_basicblock_merge(fun, jcc_bb, to);

                    let stmt = jcc_sys::ir_alloc_stmt(fun, jcc_bb);
                    let op = jcc_sys::ir_append_op(
                        fun,
                        stmt,
                        jcc_sys::IR_OP_TY_BR,
                        jcc_sys::IR_VAR_TY_NONE,
                    );
                }
            }
            TerminatorKind::SwitchInt { discr, targets } => todo!(),
            TerminatorKind::Return => unsafe {
                (*jcc_bb).ty = jcc_sys::IR_BASICBLOCK_TY_RET;
                let stmt = jcc_sys::ir_alloc_stmt(fun, jcc_bb);
                let op = jcc_sys::ir_append_op(
                    fun,
                    stmt,
                    jcc_sys::IR_OP_TY_RET,
                    jcc_sys::IR_VAR_TY_NONE,
                );

                (*op)._1.ret = jcc_sys::ir_op_ret {
                    ..Default::default()
                };
            },
            TerminatorKind::Unreachable => todo!(),
            TerminatorKind::UnwindResume => todo!(),
            TerminatorKind::UnwindTerminate(unwind_terminate_reason) => todo!(),
            TerminatorKind::Drop {
                place,
                target,
                unwind,
                replace,
            } => todo!(),
            TerminatorKind::Call {
                func,
                args,
                destination,
                target,
                unwind,
                call_source,
                fn_span,
            } => {
                // FIXME: we create too many BBs as JCC does not have bb for this
                {
                    let target = func;
                    dbg!(&target);
                }

                if let Some(target) = target {
                    let to = jcc_blocks[target.index()];
                    unsafe {
                        jcc_sys::ir_make_basicblock_merge(fun, jcc_bb, to);

                        let stmt = jcc_sys::ir_alloc_stmt(fun, jcc_bb);
                        let op = jcc_sys::ir_append_op(
                            fun,
                            stmt,
                            jcc_sys::IR_OP_TY_BR,
                            jcc_sys::IR_VAR_TY_NONE,
                        );
                    }
                }
            }
            TerminatorKind::TailCall {
                func,
                args,
                fn_span,
            } => todo!(),
            TerminatorKind::Assert {
                cond,
                expected,
                msg,
                target,
                unwind,
            } => todo!(),
            TerminatorKind::InlineAsm {
                asm_macro,
                template,
                operands,
                options,
                line_spans,
                targets,
                unwind,
            } => todo!(),
            _ => unreachable!(),
        }
    }

    pub(crate) fn codegen_fn(&mut self, inst: &Instance<'tcx>) {
        let kind = inst.ty(self.tcx, TypingEnv::fully_monomorphized()).kind();

        match kind {
            TyKind::FnDef(_, _) | TyKind::Closure(_, _) | TyKind::Coroutine(_, _) => {}
            _ => return,
        }

        let mir = self.tcx.instance_mir(inst.def);
        let sym = self.tcx.symbol_name(*inst).name;

        let def_id = inst.def.def_id();
        let param_env = self.tcx.param_env(def_id);

        let sig = kind.fn_sig(self.tcx).skip_binder();

        let params = sig.inputs();
        let ret_ty = sig.output();

        let fun_ty = jcc_sys::ir_var_func_ty {
            ret_ty: Box::into_raw(Box::new(ty_to_jcc_ty(self.tcx, &ret_ty))),
            num_params: params.len(),
            params: params
                .iter()
                .map(|ty| ty_to_jcc_ty(self.tcx, ty))
                .collect::<Vec<_>>()
                .leak()
                .as_mut_ptr(),
            flags: jcc_sys::IR_VAR_FUNC_TY_FLAG_NONE,
        };

        let mut fun_var_ty = jcc_sys::ir_var_ty {
            ty: jcc_sys::IR_VAR_TY_TY_FUNC,
            _1: jcc_sys::ir_var_ty__bindgen_ty_1 { func: fun_ty },
        };

        let glb = unsafe {
            jcc_sys::ir_add_global(
                self.unit,
                jcc_sys::IR_GLB_TY_FUNC,
                &mut fun_var_ty,
                jcc_sys::IR_GLB_DEF_TY_DEFINED,
                CString::new(sym.to_string()).unwrap().into_raw(),
            )
        };

        unsafe {
            (*glb).linkage = jcc_sys::IR_LINKAGE_EXTERNAL;
            (*glb)._1.func = self.arena.alloc::<jcc_sys::ir_func>();

            *(*glb)._1.func = jcc_sys::ir_func {
                unit: self.unit,
                func_ty: fun_ty,
                name: CString::new(sym).unwrap().into_raw(),
                arena: self.arena.as_ptr(),
                flags: IR_FUNC_FLAG_NONE,
                ..Default::default()
            };
        }

        let fun = unsafe { (*glb)._1.func };

        let jcc_blocks = mir
            .basic_blocks
            .iter()
            .map(|_| unsafe { jcc_sys::ir_alloc_basicblock(fun) })
            .collect::<Vec<_>>();

        unsafe {
            let first = jcc_blocks[0];
            let param_stmt = if (*first).first.is_null() {
                jcc_sys::ir_alloc_stmt(fun, first)
            } else {
                jcc_sys::ir_insert_before_stmt(fun, (*first).first)
            };

            (*param_stmt).flags |= jcc_sys::IR_STMT_FLAG_PARAM;

            for param in params {
                let mut jcc_ty = ty_to_jcc_ty(self.tcx, param);

                if jcc_ty.ty == jcc_sys::IR_VAR_TY_TY_STRUCT
                    || jcc_ty.ty == jcc_sys::IR_VAR_TY_TY_UNION
                {
                    let lcl = jcc_sys::ir_add_local(fun, &jcc_ty);
                    (*lcl).flags |= jcc_sys::IR_LCL_FLAG_PARAM;

                    let addr = jcc_sys::ir_alloc_op(fun, param_stmt);
                    (*addr).ty = jcc_sys::IR_OP_TY_ADDR;
                    (*addr).var_ty = jcc_sys::IR_VAR_TY_POINTER;
                    (*addr).flags |= jcc_sys::IR_OP_FLAG_PARAM;
                    (*addr)._1.addr = jcc_sys::ir_op_addr {
                        ty: jcc_sys::IR_OP_ADDR_TY_LCL,
                        _1: jcc_sys::ir_op_addr__bindgen_ty_1 { lcl },
                    };
                } else {
                    if jcc_ty.ty == jcc_sys::IR_VAR_TY_TY_ARRAY {
                        // arrays/aggregates are actually pointers
                        jcc_ty = jcc_sys::IR_VAR_TY_POINTER;
                    }

                    let lcl = jcc_sys::ir_add_local(fun, &jcc_ty);
                    (*lcl).flags |= jcc_sys::IR_LCL_FLAG_PARAM;

                    let mov = jcc_sys::ir_alloc_op(fun, param_stmt);
                    (*mov).ty = jcc_sys::IR_OP_TY_MOV;
                    (*mov).var_ty = jcc_ty;
                    (*mov).flags |= jcc_sys::IR_OP_FLAG_PARAM;
                    (*mov)._1.mov.value = ptr::null_mut();

                    let store = jcc_sys::ir_alloc_op(fun, param_stmt);
                    (*store).ty = jcc_sys::IR_OP_TY_STORE;
                    (*store).var_ty = jcc_sys::IR_VAR_TY_NONE;
                    (*store)._1.store = jcc_sys::ir_op_store {
                        ty: jcc_sys::IR_OP_STORE_TY_LCL,
                        value: mov,
                        _1: jcc_sys::ir_op_store__bindgen_ty_1 { lcl },
                    };
                }
            }
        }

        for (idx, block) in mir.basic_blocks.iter_enumerated() {
            self.codegen_basic_block(inst, fun, &jcc_blocks, idx.index(), block);
        }
    }

    pub(crate) fn codegen_static(&mut self, def_id: &DefId) {
        let const_val = self
            .tcx
            .eval_static_initializer(def_id)
            .expect("eval_static_initializer failed");
    }

    pub(crate) fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    pub(crate) fn module(&self) -> JccModule {
        JccModule { unit: self.unit }
    }
}
