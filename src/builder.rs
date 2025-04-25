use std::{
    cell::RefCell,
    num::NonZeroUsize,
    ops::{Deref, Range},
};

use rustc_abi::{BackendRepr, HasDataLayout, TargetDataLayout};
use rustc_codegen_ssa::{
    MemFlags,
    common::{IntPredicate, RealPredicate},
    mir::{
        operand::{OperandRef, OperandValue},
        place::PlaceRef,
    },
    traits::{
        AbiBuilderMethods, ArgAbiBuilderMethods, AsmBuilderMethods, BackendTypes,
        BaseTypeCodegenMethods, BuilderMethods, ConstCodegenMethods, CoverageInfoBuilderMethods,
        DebugInfoBuilderMethods, IntrinsicCallBuilderMethods, LayoutTypeCodegenMethods,
        StaticBuilderMethods,
    },
};
use rustc_middle::{
    bug,
    mir::coverage::CoverageKind,
    ty::{
        Instance, Ty, TyCtxt, TypingEnv,
        layout::{
            FnAbiError, FnAbiOfHelpers, FnAbiRequest, HasTyCtxt, HasTypingEnv, LayoutError,
            LayoutOf, LayoutOfHelpers, MaybeResult, TyAndLayout,
        },
    },
};
use rustc_span::sym;
use rustc_target::callconv::{ArgAbi, FnAbi, PassMode};
use rustc_type_ir::TyKind::FnDef;

use crate::{
    CodegenCx, JccModule,
    driver::{IrBuildValue, ty_to_jcc_ty},
    jcc::ir::{
        AddrOffset, AsIrRaw, HasNext, IrBasicBlock, IrBasicBlockTy, IrBinOpTy, IrCnstTy, IrFunc,
        IrId, IrOp, IrUnOpTy, IrVarTy,
    },
};

pub struct Builder<'a, 'tcx> {
    cx: &'a CodegenCx<'tcx>,
    cur_bb: RefCell<Option<IrBasicBlock>>,
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    pub fn with_cx(cx: &'a CodegenCx<'tcx>) -> Self {
        Self {
            cx,
            cur_bb: RefCell::new(None),
        }
    }

    pub fn module(&self) -> JccModule {
        self.cx.module()
    }

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

    fn set_cur_bb(&self, bb: IrBasicBlock) {
        *self.cur_bb.borrow_mut() = Some(bb)
    }

    fn val_to_u64(&self, val: IrBuildValue) -> Option<u64> {
        match val {
            IrBuildValue::Cnst(ir_cnst) => match ir_cnst.cnst {
                IrCnstTy::Int(val) => Some(val),
                IrCnstTy::Float(_) => None,
            },
            IrBuildValue::Op(ir_op) => ir_op.get_int_cnst().map(|c| c.val),
        }
    }
}

impl<'a, 'tcx> BackendTypes for Builder<'a, 'tcx> {
    type Value = IrBuildValue;

    type Metadata = ();
    type Function = IrFunc;
    type BasicBlock = IrBasicBlock;
    type Type = IrVarTy;
    type Funclet = ();
    type DIScope = ();
    type DILocation = ();
    type DIVariable = ();
}

impl<'a, 'tcx> StaticBuilderMethods for Builder<'a, 'tcx> {
    fn get_static(&mut self, def_id: rustc_hir::def_id::DefId) -> Self::Value {
        todo!()
    }
}

impl<'a, 'tcx> AsmBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn codegen_inline_asm(
        &mut self,
        template: &[rustc_ast::InlineAsmTemplatePiece],
        operands: &[rustc_codegen_ssa::traits::InlineAsmOperandRef<'tcx, Self>],
        options: rustc_ast::InlineAsmOptions,
        line_spans: &[rustc_span::Span],
        instance: Instance<'_>,
        dest: Option<Self::BasicBlock>,
        catch_funclet: Option<(Self::BasicBlock, Option<&Self::Funclet>)>,
    ) {
        todo!()
    }
}

impl<'a, 'tcx> IntrinsicCallBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn codegen_intrinsic_call(
        &mut self,
        instance: Instance<'tcx>,
        fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
        args: &[OperandRef<'tcx, Self::Value>],
        llresult: Self::Value,
        span: rustc_span::Span,
    ) -> Result<(), Instance<'tcx>> {
        let tcx = self.tcx;
        let callee_ty = instance.ty(tcx, self.typing_env());

        let (def_id, fn_args) = match *callee_ty.kind() {
            FnDef(def_id, fn_args) => (def_id, fn_args),
            _ => bug!("expected fn item type, found {}", callee_ty),
        };

        let sig = callee_ty.fn_sig(tcx);
        let sig = tcx.normalize_erasing_late_bound_regions(self.typing_env(), sig);

        let arg_tys = sig.inputs();
        let ret_ty = sig.output();

        let name = tcx.item_name(def_id);
        let name_str = name.as_str();

        let llret_ty = ty_to_jcc_ty(self, &self.layout_of(ret_ty));

        let result = PlaceRef::new_sized(llresult, fn_abi.ret.layout);

        match name {
            sym::black_box => {
                // FIXME: jcc needs black box (probably via volatile)
                return Ok(());
            }
            _ => todo!("intrinsic {name}"),
        }
    }

    fn abort(&mut self) {
        todo!()
    }

    fn assume(&mut self, val: Self::Value) {
        todo!()
    }

    fn expect(&mut self, cond: Self::Value, expected: bool) -> Self::Value {
        todo!()
    }

    fn type_test(&mut self, pointer: Self::Value, typeid: Self::Metadata) -> Self::Value {
        todo!()
    }

    fn type_checked_load(
        &mut self,
        llvtable: Self::Value,
        vtable_byte_offset: u64,
        typeid: Self::Metadata,
    ) -> Self::Value {
        todo!()
    }

    fn va_start(&mut self, val: Self::Value) -> Self::Value {
        todo!()
    }

    fn va_end(&mut self, val: Self::Value) -> Self::Value {
        todo!()
    }
}

impl<'a, 'tcx> AbiBuilderMethods for Builder<'a, 'tcx> {
    fn get_param(&mut self, index: usize) -> Self::Value {
        self.cur_bb
            .borrow()
            .unwrap()
            .func()
            .get_param_op(index)
            .into()
    }
}

impl<'a, 'tcx> ArgAbiBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn store_fn_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        idx: &mut usize,
        dst: PlaceRef<'tcx, Self::Value>,
    ) {
        // store argument passed to function, ie we are callee/receiver
        let bb = self.cur_bb.borrow().unwrap();
        let stmt = bb.alloc_stmt();

        let addr = dst.val.llval;

        let value = match arg_abi.mode {
            PassMode::Ignore => return,
            PassMode::Direct(arg_attributes) => {
                // need to get the nth argument
                bb.func().get_param_op(*idx)
            }
            PassMode::Pair(arg_attributes, arg_attributes1) => todo!(),
            PassMode::Cast { pad_i32, ref cast } => todo!(),
            PassMode::Indirect {
                attrs,
                meta_attrs,
                on_stack,
            } => todo!(),
        };

        let op = stmt.alloc_op();
        op.mk_store_addr(addr.op(), value);
    }

    fn store_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        val: Self::Value,
        dst: PlaceRef<'tcx, Self::Value>,
    ) {
        // store argument to pass to function, ie we are caller/sender
        // nop
    }

    fn arg_memory_ty(&self, arg_abi: &ArgAbi<'tcx, Ty<'tcx>>) -> Self::Type {
        todo!()
    }
}

impl<'a, 'tcx> DebugInfoBuilderMethods for Builder<'a, 'tcx> {
    fn dbg_var_addr(
        &mut self,
        dbg_var: Self::DIVariable,
        dbg_loc: Self::DILocation,
        variable_alloca: Self::Value,
        direct_offset: rustc_abi::Size,
        // NB: each offset implies a deref (i.e. they're steps in a pointer chain).
        indirect_offsets: &[rustc_abi::Size],
        // Byte range in the `dbg_var` covered by this fragment,
        // if this is a fragment of a composite `DIVariable`.
        fragment: Option<Range<rustc_abi::Size>>,
    ) {
        // TODO:
    }

    fn set_dbg_loc(&mut self, dbg_loc: Self::DILocation) {
        // TODO:
    }

    fn clear_dbg_loc(&mut self) {
        // TODO:
    }

    fn get_dbg_loc(&self) -> Option<Self::DILocation> {
        // TODO:
        None
    }

    fn insert_reference_to_gdb_debug_scripts_section_global(&mut self) {
        // TODO:
    }

    fn set_var_name(&mut self, value: Self::Value, name: &str) {
        // TODO:
    }
}

impl<'a, 'tcx> CoverageInfoBuilderMethods<'tcx> for Builder<'a, 'tcx> {
    fn add_coverage(&mut self, instance: Instance<'tcx>, kind: &CoverageKind) {
        todo!()
    }
}

impl<'a, 'tcx> Deref for Builder<'a, 'tcx> {
    type Target = CodegenCx<'tcx>;

    fn deref(&self) -> &Self::Target {
        self.cx
    }
}

impl<'a, 'tcx> LayoutOfHelpers<'tcx> for Builder<'a, 'tcx> {
    type LayoutOfResult = TyAndLayout<'tcx>;

    fn handle_layout_err(
        &self,
        err: LayoutError<'tcx>,
        span: rustc_span::Span,
        ty: Ty<'tcx>,
    ) -> <Self::LayoutOfResult as MaybeResult<TyAndLayout<'tcx>>>::Error {
        todo!()
    }
}

impl<'a, 'tcx> HasDataLayout for Builder<'a, 'tcx> {
    fn data_layout(&self) -> &TargetDataLayout {
        self.cx.data_layout()
    }
}

impl<'a, 'tcx> HasTyCtxt<'tcx> for Builder<'a, 'tcx> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.cx.tcx
    }
}

impl<'a, 'tcx> HasTypingEnv<'tcx> for Builder<'a, 'tcx> {
    fn typing_env(&self) -> TypingEnv<'tcx> {
        TypingEnv::fully_monomorphized()
    }
}

impl<'a, 'tcx> FnAbiOfHelpers<'tcx> for Builder<'a, 'tcx> {
    type FnAbiOfResult = &'tcx FnAbi<'tcx, Ty<'tcx>>;

    fn handle_fn_abi_err(
        &self,
        err: FnAbiError<'tcx>,
        span: rustc_span::Span,
        fn_abi_request: FnAbiRequest<'tcx>,
    ) -> <Self::FnAbiOfResult as MaybeResult<&'tcx FnAbi<'tcx, Ty<'tcx>>>>::Error {
        todo!()
    }
}

impl<'a, 'tcx> BuilderMethods<'a, 'tcx> for Builder<'a, 'tcx> {
    type CodegenCx = CodegenCx<'tcx>;

    fn build(cx: &'a Self::CodegenCx, llbb: Self::BasicBlock) -> Self {
        Builder::with_cx(cx)
    }

    fn cx(&self) -> &Self::CodegenCx {
        self.cx
    }

    fn llbb(&self) -> Self::BasicBlock {
        self.cur_bb.borrow().unwrap()
    }

    fn set_span(&mut self, span: rustc_span::Span) {
        // TODO: debug info
    }

    fn append_block(cx: &'a Self::CodegenCx, llfn: Self::Function, name: &str) -> Self::BasicBlock {
        // if only one bb, it is param block, keep using it
        let first = llfn.first();
        if let Some(first) = first
            && first.next().is_none()
        {
            return first;
        }

        let bb = llfn.alloc_basicblock();

        bb.comment(name.as_bytes());
        bb
    }

    fn append_sibling_block(&mut self, name: &str) -> Self::BasicBlock {
        let cur = &self.cur_bb.borrow().unwrap();
        let bb = cur.func().alloc_basicblock();

        bb.comment(name.as_bytes());
        bb
    }

    fn switch_to_block(&mut self, llbb: Self::BasicBlock) {
        *self.cur_bb.borrow_mut() = Some(llbb);
    }

    fn ret_void(&mut self) {
        let op = self.alloc_next_op();
        op.mk_ret(None);
    }

    fn ret(&mut self, v: Self::Value) {
        let bb = self.cur_bb.borrow().unwrap();
        bb.mk_ty(IrBasicBlockTy::Ret);

        let op = self.alloc_next_op();
        op.mk_ret(Some(v.op()));
    }

    fn br(&mut self, dest: Self::BasicBlock) {
        let bb = self.cur_bb.borrow().unwrap();
        bb.mk_ty(IrBasicBlockTy::Merge { target: dest });

        let op = self.alloc_next_op();
        op.mk_br();
    }

    fn cond_br(
        &mut self,
        cond: Self::Value,
        then_llbb: Self::BasicBlock,
        else_llbb: Self::BasicBlock,
    ) {
        let bb = self.cur_bb.borrow().unwrap();
        bb.mk_ty(IrBasicBlockTy::Split {
            true_target: then_llbb,
            false_target: else_llbb,
        });

        let op = self.alloc_next_op();
        op.mk_cond_br(cond.op());
    }

    fn switch(
        &mut self,
        v: Self::Value,
        else_llbb: Self::BasicBlock,
        cases: impl ExactSizeIterator<Item = (u128, Self::BasicBlock)>,
    ) {
        todo!()
    }

    fn invoke(
        &mut self,
        llty: Self::Type,
        fn_attrs: Option<&rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        llfn: Self::Value,
        args: &[Self::Value],
        then: Self::BasicBlock,
        catch: Self::BasicBlock,
        funclet: Option<&Self::Funclet>,
        instance: Option<Instance<'tcx>>,
    ) -> Self::Value {
        todo!()
    }

    fn unreachable(&mut self) {
        todo!()
    }

    fn add(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Add, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn fadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fadd, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn fadd_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fadd, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn fadd_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fadd, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn sub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Sub, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn fsub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fsub, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn fsub_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fsub, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn fsub_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fsub, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn mul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Mul, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn fmul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fmul, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn fmul_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fmul, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn fmul_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fmul, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn udiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Udiv, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn exactudiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        // FIXME: what is this op?
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Udiv, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn sdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Sdiv, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn exactsdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        // FIXME: what is this op?
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Sdiv, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn fdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fdiv, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn fdiv_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fdiv, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn fdiv_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fdiv, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn urem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Umod, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn srem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Smod, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn frem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!("frem");
    }

    fn frem_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!("frem");
    }

    fn frem_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        todo!("frem");
    }

    fn shl(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Lshift, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn lshr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Urshift, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn ashr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Srshift, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn and(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::And, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn or(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Or, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn xor(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Xor, var_ty, lhs.op(), rhs.op()))
            .into()
    }

    fn neg(&mut self, v: Self::Value) -> Self::Value {
        let var_ty = v.var_ty();
        self.mk_next_op(|op| op.mk_unnop(IrUnOpTy::Neg, var_ty, v.op()))
            .into()
    }

    fn fneg(&mut self, v: Self::Value) -> Self::Value {
        let var_ty = v.var_ty();
        self.mk_next_op(|op| op.mk_unnop(IrUnOpTy::Fneg, var_ty, v.op()))
            .into()
    }

    fn not(&mut self, v: Self::Value) -> Self::Value {
        let var_ty = v.var_ty();
        self.mk_next_op(|op| op.mk_unnop(IrUnOpTy::Not, var_ty, v.op()))
            .into()
    }

    fn checked_binop(
        &mut self,
        oop: rustc_codegen_ssa::traits::OverflowOp,
        ty: Ty<'_>,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> (Self::Value, Self::Value) {
        todo!()
    }

    fn from_immediate(&mut self, val: Self::Value) -> Self::Value {
        if self.cx().val_ty(val) == self.cx().type_i1() {
            self.zext(val, self.cx().type_i8())
        } else {
            val
        }
    }

    fn to_immediate_scalar(&mut self, val: Self::Value, scalar: rustc_abi::Scalar) -> Self::Value {
        if scalar.is_bool() {
            return self.unchecked_utrunc(val, self.cx().type_i1());
        }

        val
    }

    fn alloca(&mut self, size: rustc_abi::Size, align: rustc_abi::Align) -> Self::Value {
        // HACK: we don't have "alloc size"
        // only "alloc local"

        let sz = size.bytes_usize();
        let ty = self.unit.var_ty_bytes(sz);

        let bb = self.cur_bb.borrow().unwrap();
        let lcl = bb.func().add_local(ty);

        // TODO: should we alloc new stmt here?
        let op = self.alloc_next_op();
        op.mk_addr_lcl(lcl);

        op.into()
    }

    fn dynamic_alloca(&mut self, size: Self::Value, align: rustc_abi::Align) -> Self::Value {
        todo!("dynamic alloca not supported")
    }

    fn load(&mut self, ty: Self::Type, ptr: Self::Value, align: rustc_abi::Align) -> Self::Value {
        self.mk_next_op(|op| op.mk_load_addr(ty, ptr.op())).into()
    }

    fn volatile_load(&mut self, ty: Self::Type, ptr: Self::Value) -> Self::Value {
        todo!()
    }

    fn atomic_load(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        order: rustc_codegen_ssa::common::AtomicOrdering,
        size: rustc_abi::Size,
    ) -> Self::Value {
        todo!()
    }

    fn load_operand(
        &mut self,
        place: PlaceRef<'tcx, Self::Value>,
    ) -> OperandRef<'tcx, Self::Value> {
        if place.layout.is_zst() {
            todo!("load zst");
        }

        // let op = self.alloc_next_op();
        // op.mk_store_addr(ptr, val);
        // op

        let val = if place.val.llextra.is_some() {
            // FIXME: Merge with the `else` below?
            OperandValue::Ref(place.val)
        } else if self.is_backend_immediate(place.layout) {
            let load = self.load(
                self.backend_type(place.layout),
                place.val.llval,
                place.val.align,
            );

            OperandValue::Immediate(
                if let BackendRepr::Scalar(ref scalar) = place.layout.backend_repr {
                    self.to_immediate_scalar(load, *scalar)
                } else {
                    load
                },
            )
        } else if let BackendRepr::ScalarPair(ref a, ref b) = place.layout.backend_repr {
            let b_offset = a.size(self).align_to(b.align(self).abi);
            let load = |i: i32, scalar: &rustc_abi::Scalar, align: usize| {
                let llptr = if i == 0 {
                    place.val.llval
                } else {
                    self.inbounds_ptradd(place.val.llval, self.const_usize(b_offset.bytes()))
                };

                // let llty = place.layout.scalar_pair_element_gcc_type(self, i);
                // let load = self.load(llty, llptr, align);
                // scalar_load_metadata(self, load, scalar);
                // if scalar.is_bool() {
                //     self.trunc(load, self.type_i1())
                // } else {
                //     load
                // }
            };
            // OperandValue::Pair(
            //     load(0, a, place.val.align),
            //     load(1, b, place.val.align.restrict_for_offset(b_offset))
            // );
            todo!("pairs")
        } else {
            OperandValue::Ref(place.val)
        };

        OperandRef {
            val,
            layout: place.layout,
        }
    }

    fn write_operand_repeatedly(
        &mut self,
        elem: OperandRef<'tcx, Self::Value>,
        count: u64,
        dest: PlaceRef<'tcx, Self::Value>,
    ) {
        todo!()
    }

    fn range_metadata(&mut self, load: Self::Value, range: rustc_abi::WrappingRange) {
        todo!()
    }

    fn nonnull_metadata(&mut self, load: Self::Value) {
        todo!()
    }

    fn store(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        align: rustc_abi::Align,
    ) -> Self::Value {
        self.mk_next_op(|op| op.mk_store_addr(ptr.op(), val.op()))
            .into()
    }

    fn store_with_flags(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        align: rustc_abi::Align,
        flags: MemFlags,
    ) -> Self::Value {
        // FIXME: flags
        if !flags.is_empty() {
            todo!("mem flags");
        }

        self.store(val, ptr, align)
    }

    fn atomic_store(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        order: rustc_codegen_ssa::common::AtomicOrdering,
        size: rustc_abi::Size,
    ) {
        todo!()
    }

    fn gep(&mut self, ty: Self::Type, ptr: Self::Value, indices: &[Self::Value]) -> Self::Value {
        debug_assert!(
            indices.len() == 1,
            "multi-index gep (how do we calculate the other types scales?)",
        );

        let index = indices[0];
        let base = ptr;
        let scale = self.unit.var_ty_info(ty).size;
        let scale = NonZeroUsize::new(scale).unwrap();

        let op = self.alloc_next_op();
        op.mk_addr_offset(AddrOffset::index(base.op(), index.op(), scale));
        op.into()
    }

    fn inbounds_gep(
        &mut self,
        ty: Self::Type,
        ptr: Self::Value,
        indices: &[Self::Value],
    ) -> Self::Value {
        self.gep(ty, ptr, indices)
    }

    fn trunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.mk_next_op(|op| op.mk_trunc(dest_ty, val.op())).into()
    }

    fn sext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.mk_next_op(|op| op.mk_sext(dest_ty, val.op())).into()
    }

    fn zext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.mk_next_op(|op| op.mk_zext(dest_ty, val.op())).into()
    }

    fn fptoui_sat(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn fptosi_sat(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn fptoui(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.mk_next_op(|op| op.mk_uconv(dest_ty, val.op())).into()
    }

    fn fptosi(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.mk_next_op(|op| op.mk_sconv(dest_ty, val.op())).into()
    }

    fn uitofp(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.mk_next_op(|op| op.mk_uconv(dest_ty, val.op())).into()
    }

    fn sitofp(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.mk_next_op(|op| op.mk_sconv(dest_ty, val.op())).into()
    }

    fn fptrunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.mk_next_op(|op| op.mk_conv(dest_ty, val.op())).into()
    }

    fn fpext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.mk_next_op(|op| op.mk_conv(dest_ty, val.op())).into()
    }

    fn ptrtoint(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn inttoptr(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn bitcast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn intcast(&mut self, val: Self::Value, dest_ty: Self::Type, is_signed: bool) -> Self::Value {
        let src_ty = val.var_ty();
        debug_assert!(src_ty.is_int() && dest_ty.is_int());

        match (dest_ty.is_int_larger(&src_ty), is_signed) {
            (true, true) => self.sext(val, dest_ty),
            (true, false) => self.zext(val, dest_ty),
            (false, _) => self.trunc(val, dest_ty),
        }
    }

    fn pointercast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn icmp(&mut self, op: IntPredicate, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let ty = match op {
            IntPredicate::IntEQ => IrBinOpTy::Eq,
            IntPredicate::IntNE => IrBinOpTy::Neq,
            IntPredicate::IntUGT => IrBinOpTy::Ugt,
            IntPredicate::IntUGE => IrBinOpTy::Ugteq,
            IntPredicate::IntULT => IrBinOpTy::Ult,
            IntPredicate::IntULE => IrBinOpTy::Ulteq,
            IntPredicate::IntSGT => IrBinOpTy::Sgt,
            IntPredicate::IntSGE => IrBinOpTy::Sgteq,
            IntPredicate::IntSLT => IrBinOpTy::Slt,
            IntPredicate::IntSLE => IrBinOpTy::Slteq,
        };

        self.mk_next_op(|op| op.mk_binop(ty, self.type_i1(), lhs.op(), rhs.op()))
            .into()
    }

    fn fcmp(&mut self, op: RealPredicate, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let ty = match op {
            // FIXME: this does not properly respect ordered/unorderded
            RealPredicate::RealPredicateFalse => todo!(),
            RealPredicate::RealPredicateTrue => todo!(),
            RealPredicate::RealORD => todo!(),
            RealPredicate::RealUNO => todo!(),
            RealPredicate::RealOEQ => IrBinOpTy::Feq,
            RealPredicate::RealOGT => IrBinOpTy::Fgt,
            RealPredicate::RealOGE => IrBinOpTy::Fgteq,
            RealPredicate::RealOLT => IrBinOpTy::Flt,
            RealPredicate::RealOLE => IrBinOpTy::Flteq,
            RealPredicate::RealONE => IrBinOpTy::Fneq,
            RealPredicate::RealUEQ => IrBinOpTy::Feq,
            RealPredicate::RealUGT => IrBinOpTy::Fgt,
            RealPredicate::RealUGE => IrBinOpTy::Fgteq,
            RealPredicate::RealULT => IrBinOpTy::Flt,
            RealPredicate::RealULE => IrBinOpTy::Flteq,
            RealPredicate::RealUNE => IrBinOpTy::Fneq,
        };

        self.mk_next_op(|op| op.mk_binop(ty, self.type_i1(), lhs.op(), rhs.op()))
            .into()
    }

    fn memcpy(
        &mut self,
        dst: Self::Value,
        dst_align: rustc_abi::Align,
        src: Self::Value,
        src_align: rustc_abi::Align,
        size: Self::Value,
        flags: MemFlags,
    ) {
        let Some(size) = self.val_to_u64(size) else {
            todo!("non cnst size for memcpy");
        };

        let size = size.try_into().expect("u64 -> usize fail");

        let op = self.alloc_next_op();
        op.mk_memcpy(src.op(), dst.op(), size);
    }

    fn memmove(
        &mut self,
        dst: Self::Value,
        dst_align: rustc_abi::Align,
        src: Self::Value,
        src_align: rustc_abi::Align,
        size: Self::Value,
        flags: MemFlags,
    ) {
        todo!()
    }

    fn memset(
        &mut self,
        ptr: Self::Value,
        fill_byte: Self::Value,
        size: Self::Value,
        align: rustc_abi::Align,
        flags: MemFlags,
    ) {
        let Some(fill_byte) = self.val_to_u64(fill_byte).and_then(|f| f.try_into().ok()) else {
            todo!("non cnst fill byte for memset");
        };

        let Some(size) = self.val_to_u64(size).and_then(|f| f.try_into().ok()) else {
            todo!("non cnst size for memset");
        };

        let op = self.alloc_next_op();
        op.mk_memset(ptr.op(), fill_byte, size);
    }

    fn select(
        &mut self,
        cond: Self::Value,
        then_val: Self::Value,
        else_val: Self::Value,
    ) -> Self::Value {
        todo!()
    }

    fn va_arg(&mut self, list: Self::Value, ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn extract_element(&mut self, vec: Self::Value, idx: Self::Value) -> Self::Value {
        todo!()
    }

    fn vector_splat(&mut self, num_elts: usize, elt: Self::Value) -> Self::Value {
        todo!()
    }

    fn extract_value(&mut self, agg_val: Self::Value, idx: u64) -> Self::Value {
        todo!()
    }

    fn insert_value(&mut self, agg_val: Self::Value, elt: Self::Value, idx: u64) -> Self::Value {
        todo!()
    }

    fn set_personality_fn(&mut self, personality: Self::Value) {
        todo!()
    }

    fn cleanup_landing_pad(&mut self, pers_fn: Self::Value) -> (Self::Value, Self::Value) {
        todo!()
    }

    fn filter_landing_pad(&mut self, pers_fn: Self::Value) -> (Self::Value, Self::Value) {
        todo!()
    }

    fn resume(&mut self, exn0: Self::Value, exn1: Self::Value) {
        todo!()
    }

    fn cleanup_pad(&mut self, parent: Option<Self::Value>, args: &[Self::Value]) -> Self::Funclet {
        todo!()
    }

    fn cleanup_ret(&mut self, funclet: &Self::Funclet, unwind: Option<Self::BasicBlock>) {
        todo!()
    }

    fn catch_pad(&mut self, parent: Self::Value, args: &[Self::Value]) -> Self::Funclet {
        todo!()
    }

    fn catch_switch(
        &mut self,
        parent: Option<Self::Value>,
        unwind: Option<Self::BasicBlock>,
        handlers: &[Self::BasicBlock],
    ) -> Self::Value {
        todo!()
    }

    fn atomic_cmpxchg(
        &mut self,
        dst: Self::Value,
        cmp: Self::Value,
        src: Self::Value,
        order: rustc_codegen_ssa::common::AtomicOrdering,
        failure_order: rustc_codegen_ssa::common::AtomicOrdering,
        weak: bool,
    ) -> (Self::Value, Self::Value) {
        todo!()
    }

    fn atomic_rmw(
        &mut self,
        op: rustc_codegen_ssa::common::AtomicRmwBinOp,
        dst: Self::Value,
        src: Self::Value,
        order: rustc_codegen_ssa::common::AtomicOrdering,
    ) -> Self::Value {
        todo!()
    }

    fn atomic_fence(
        &mut self,
        order: rustc_codegen_ssa::common::AtomicOrdering,
        scope: rustc_codegen_ssa::common::SynchronizationScope,
    ) {
        todo!()
    }

    fn set_invariant_load(&mut self, load: Self::Value) {
        todo!()
    }

    fn lifetime_start(&mut self, ptr: Self::Value, size: rustc_abi::Size) {}

    fn lifetime_end(&mut self, ptr: Self::Value, size: rustc_abi::Size) {}

    fn call(
        &mut self,
        llty: Self::Type,
        fn_attrs: Option<&rustc_middle::middle::codegen_fn_attrs::CodegenFnAttrs>,
        fn_abi: Option<&FnAbi<'tcx, Ty<'tcx>>>,
        llfn: Self::Value,
        args: &[Self::Value],
        funclet: Option<&Self::Funclet>,
        instance: Option<Instance<'tcx>>,
    ) -> Self::Value {
        // TODO: fn ptr calls (not addr-glb)
        // let Some(fn_glb) = llfn.get_addr_glb() else {
        //     todo!("non direct calls");
        // };

        let ret = match fn_abi {
            Some(f) => self.backend_type(f.ret.layout),
            None => {
                let Some(fun) = llty.fun() else {
                    bug!("could not deduce return type (fn_abi None, and llty was not fun ty)");
                };

                fun.ret
            }
        };

        let args = args.into_iter().map(|a| a.op()).collect::<Vec<_>>();

        let op = self.mk_next_op(|op| op.mk_call(llty, llfn.op(), &args, ret));
        op.into()
    }

    fn apply_attrs_to_cleanup_callsite(&mut self, llret: Self::Value) {
        todo!()
    }
}
