use std::{
    ffi::{self, CString},
    ptr,
};

use rustc_codegen_ssa::{CrateInfo, traits::ExtraBackendMethods};
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_metadata::EncodedMetadata;
use rustc_middle::{
    mir::{BasicBlock, BasicBlockData, Statement, StatementKind, TerminatorKind},
    ty::{EarlyBinder, Instance, Ty, TyCtxt, TyKind, TypingEnv},
};
use rustc_span::{Symbol, def_id::DefId};
use rustc_type_ir::{FloatTy, IntTy, UintTy};

use crate::{
    jcc::{self, ArenaAlloc},
    jcc_sys::{self, IR_FUNC_FLAG_NONE},
};

fn ty_to_jcc_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: &Ty<'tcx>) -> jcc_sys::ir_var_ty {
    unsafe {
        // TODO: ptr-size ty
        let pointer_ty = jcc_sys::IR_VAR_TY_I64;

        match ty.kind() {
            TyKind::Never => jcc_sys::IR_VAR_TY_NONE,
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
            _ => panic!("bad ty {ty:?}"),
        }
    }
}

pub(crate) struct JccModule {
    pub(crate) unit: *mut jcc_sys::ir_unit,
}

unsafe impl Send for JccModule {}
unsafe impl Sync for JccModule {}

pub(crate) struct CodegenCx<'tcx> {
    arena: ArenaAlloc,
    tcx: TyCtxt<'tcx>,
    unit: *mut jcc_sys::ir_unit,
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
        block: &BasicBlockData<'tcx>,
    ) {
        for stmt in block.statements.iter() {
            self.codegen_stmt(inst, stmt);
        }

        let terminator = block.terminator();
        match &terminator.kind {
            TerminatorKind::Goto { target } => todo!(),
            TerminatorKind::SwitchInt { discr, targets } => todo!(),
            TerminatorKind::Return => todo!(),
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
            } => todo!(),
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

        let fun = jcc_sys::ir_func {
            unit: self.unit,
            func_ty: fun_ty,
            name: CString::new(sym).unwrap().into_raw(),
            arena: self.arena.as_ptr(),
            flags: IR_FUNC_FLAG_NONE,
            ..Default::default()
        };

        for block in mir.basic_blocks.iter() {
            self.codegen_basic_block(inst, block);
        }
    }

    pub(crate) fn codegen_static(&mut self, def_id: &DefId) {
        let const_val = self
            .tcx
            .eval_static_initializer(def_id)
            .expect("eval_static_initializer failed");
    }

    pub(crate) fn emit_cgu(&mut self, name: Symbol) -> JccModule {
        JccModule { unit: self.unit }
    }
}
