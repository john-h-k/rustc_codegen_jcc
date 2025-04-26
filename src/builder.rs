use std::{
    cell::RefCell,
    cmp::{self, Ordering},
    num::NonZeroUsize,
    ops::{Deref, Range},
};

use rustc_abi::{HasDataLayout, TargetDataLayout};
use rustc_codegen_ssa::{
    MemFlags,
    common::{IntPredicate, RealPredicate},
    mir::{
        operand::{OperandRef, OperandValue},
        place::{PlaceRef, PlaceValue},
    },
    traits::{
        AbiBuilderMethods, ArgAbiBuilderMethods, AsmBuilderMethods, BackendTypes,
        BaseTypeCodegenMethods, BuilderMethods, ConstCodegenMethods, CoverageInfoBuilderMethods,
        DebugInfoBuilderMethods, IntrinsicCallBuilderMethods, LayoutTypeCodegenMethods,
        MiscCodegenMethods, OverflowOp, StaticBuilderMethods,
    },
};
use rustc_middle::{
    bug,
    middle::codegen_fn_attrs::CodegenFnAttrs,
    mir::coverage::CoverageKind,
    ty::{
        Instance, Ty, TyCtxt, TyKind, TypingEnv,
        layout::{
            FnAbiError, FnAbiOfHelpers, FnAbiRequest, HasTyCtxt, HasTypingEnv, LayoutError,
            LayoutOf, LayoutOfHelpers, MaybeResult, TyAndLayout,
        },
    },
};
use rustc_session::config::OptLevel;
use rustc_span::sym;
use rustc_target::callconv::{ArgAbi, FnAbi, PassMode};
use rustc_type_ir::TyKind::FnDef;

use crate::{
    CodegenCx, JccModule,
    driver::{IrBuildValue, scalar_pair_element_to_jcc_ty, ty_to_jcc_ty},
    jcc::ir::{
        AddrOffset, HasNext, IrBasicBlock, IrBasicBlockTy, IrBinOpTy, IrComment, IrFunc, IrOp,
        IrUnOpTy, IrVarTy,
    },
};

pub struct Builder<'a, 'tcx> {
    cx: &'a CodegenCx<'tcx>,
    cur_bb: RefCell<Option<IrBasicBlock>>,
    func: IrFunc,
}

impl<'a, 'tcx> Builder<'a, 'tcx> {
    pub fn with_cx(cx: &'a CodegenCx<'tcx>, cur_bb: IrBasicBlock) -> Self {
        Self {
            cx,
            func: cur_bb.func(),
            cur_bb: RefCell::new(Some(cur_bb)),
        }
    }

    fn nop_cast(&mut self, value: IrBuildValue, dest_ty: IrVarTy) -> IrBuildValue {
        let value = self.mk_op(value);
        self.mk_next_op(|op| op.mk_mov(dest_ty, value)).into()
    }

    fn ptr_size(&self) -> NonZeroUsize {
        // FIXME: ptr size
        const { NonZeroUsize::new(8).unwrap() }
    }

    fn get_block(&self) -> IrBasicBlock {
        self.cur_bb.borrow_mut().unwrap()
    }

    fn end_block(&self) {
        // *self.cur_bb.borrow_mut() = None;
    }

    pub fn module(&self) -> JccModule {
        self.cx.module()
    }

    // marked 'mut' because 'mk_next_op' is _not_ and calling this within that closure will cause incorrect op orderings
    pub fn mk_op(&mut self, value: IrBuildValue) -> IrOp {
        match value {
            IrBuildValue::Undf(var_ty) | IrBuildValue::Poison(var_ty) => {
                if var_ty.is_primitive() {
                    self.mk_next_op(|op| op.mk_undf(var_ty))
                } else {
                    let lcl = self.func.add_local(var_ty);
                    let op = self.mk_next_op(|op| op.mk_addr_lcl(lcl));
                    op.comment(b"undf/poison");
                    op
                }
            }

            IrBuildValue::Op(ir_op) => {
                // clone addr-lcl for nicer IR
                if let Some(lcl) = ir_op.get_addr_lcl() {
                    self.mk_next_op(|op| op.mk_addr_lcl(lcl))
                } else {
                    ir_op
                }
            }
            IrBuildValue::Cnst(ir_cnst) => self.mk_next_op(|op| op.mk_cnst(ir_cnst)),
            IrBuildValue::GlbAddr { glb, offset } => {
                let base = self.mk_next_op(|op| op.mk_addr_glb(glb));

                match offset {
                    0 => base,
                    offset => {
                        self.mk_next_op(|op| op.mk_addr_offset(AddrOffset::offset(base, offset)))
                    }
                }
            }
        }
    }

    pub fn alloc_next_op(&self) -> IrOp {
        let bb = self.get_block();

        let stmt = bb.alloc_stmt();

        stmt.alloc_op()
    }

    pub fn mk_next_op(&self, mk: impl Fn(IrOp)) -> IrOp {
        let op = self.alloc_next_op();

        mk(op);

        op
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
                Ok(())
            }
            _ => todo!("intrinsic {name}"),
        }
    }

    fn abort(&mut self) {
        todo!()
    }

    fn assume(&mut self, val: Self::Value) {
        // TODO:
    }

    fn expect(&mut self, cond: Self::Value, expected: bool) -> Self::Value {
        // TODO:
        cond
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

        let addr = dst.val.llval;

        let mut next = || {
            let val = self.get_param(*idx);
            *idx += 1;
            val
        };

        match arg_abi.mode {
            PassMode::Ignore => {}
            PassMode::Pair(..) => {
                OperandValue::Pair(next(), next()).store(self, dst);
            }
            PassMode::Indirect {
                attrs: _,
                meta_attrs: Some(_),
                on_stack: _,
            } => {
                let place_val = PlaceValue {
                    llval: next(),
                    llextra: Some(next()),
                    align: arg_abi.layout.align.abi,
                };
                OperandValue::Ref(place_val).store(self, dst);
            }
            PassMode::Direct(_)
            | PassMode::Indirect {
                attrs: _,
                meta_attrs: None,
                on_stack: _,
            }
            | PassMode::Cast { .. } => {
                let next_arg = next();
                self.store_arg(arg_abi, next_arg, dst);
            }
        }
    }

    fn store_arg(
        &mut self,
        arg_abi: &ArgAbi<'tcx, Ty<'tcx>>,
        val: Self::Value,
        dst: PlaceRef<'tcx, Self::Value>,
    ) {
        match &arg_abi.mode {
            PassMode::Ignore => {}
            // Sized indirect arguments
            PassMode::Indirect {
                attrs,
                meta_attrs: None,
                on_stack: _,
            } => {
                let align = attrs.pointee_align.unwrap_or(arg_abi.layout.align.abi);
                OperandValue::Ref(PlaceValue::new_sized(val, align)).store(self, dst);
            }
            // Unsized indirect qrguments
            PassMode::Indirect {
                attrs: _,
                meta_attrs: Some(_),
                on_stack: _,
            } => {
                bug!("unsized `ArgAbi` must be handled through `store_fn_arg`");
            }
            PassMode::Cast { cast, pad_i32: _ } => {
                // The ABI mandates that the value is passed as a different struct representation.
                // Spill and reload it from the stack to convert from the ABI representation to
                // the Rust representation.
                let scratch_size = cast.size(self);
                let scratch_align = cast.align(self);
                // Note that the ABI type may be either larger or smaller than the Rust type,
                // due to the presence or absence of trailing padding. For example:
                // - On some ABIs, the Rust layout { f64, f32, <f32 padding> } may omit padding
                //   when passed by value, making it smaller.
                // - On some ABIs, the Rust layout { u16, u16, u16 } may be padded up to 8 bytes
                //   when passed by value, making it larger.
                let copy_bytes = cmp::min(
                    cast.unaligned_size(self).bytes(),
                    arg_abi.layout.size.bytes(),
                );
                // Allocate some scratch space...
                let llscratch = self.alloca(scratch_size, scratch_align);
                self.lifetime_start(llscratch, scratch_size);
                // ...store the value...
                self.store(val, llscratch, scratch_align);
                // ... and then memcpy it to the intended destination.
                self.memcpy(
                    dst.val.llval,
                    arg_abi.layout.align.abi,
                    llscratch,
                    scratch_align,
                    self.const_usize(copy_bytes),
                    MemFlags::empty(),
                );
                self.lifetime_end(llscratch, scratch_size);
            }
            _ => {
                OperandRef::from_immediate_or_packed_pair(self, val, arg_abi.layout)
                    .val
                    .store(self, dst);
            }
        }
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
        Builder::with_cx(cx, llbb)
    }

    fn cx(&self) -> &Self::CodegenCx {
        self.cx
    }

    fn llbb(&self) -> Self::BasicBlock {
        // TODO: is clone proper
        self.cur_bb.borrow().unwrap()
    }

    fn set_span(&mut self, span: rustc_span::Span) {
        // TODO: debug info
    }

    fn append_block(cx: &'a Self::CodegenCx, llfn: Self::Function, name: &str) -> Self::BasicBlock {
        // if only one bb, it is param block,
        let first = llfn.first();
        if let Some(first) = first
            && first.first().and_then(|b| b.next()).is_none()
        {
            first.comment(name.as_bytes());
            return first;
        }

        let bb = llfn.alloc_basicblock();

        bb.comment(name.as_bytes());
        bb
    }

    fn append_sibling_block(&mut self, name: &str) -> Self::BasicBlock {
        Self::append_block(self.cx, self.func, name)
    }

    fn switch_to_block(&mut self, llbb: Self::BasicBlock) {
        *self.cur_bb.borrow_mut() = Some(llbb);
    }

    fn ret_void(&mut self) {
        let bb = self.get_block();
        bb.mk_ty(IrBasicBlockTy::Ret);

        self.mk_next_op(|op| op.mk_ret(None));
    }

    fn ret(&mut self, v: Self::Value) {
        let bb = self.get_block();
        bb.mk_ty(IrBasicBlockTy::Ret);

        // HACK: we should make it easier for consumers to build this (so change jcc)
        let fun_ret_var_ty = self.func.ret_var_ty();

        let v = self.mk_op(v);
        let v = if fun_ret_var_ty.is_aggregate() && v.var_ty().is_pointer() {
            self.mk_next_op(|op| op.mk_load_addr(fun_ret_var_ty, v))
        } else {
            v
        };

        self.mk_next_op(|op| op.mk_ret(Some(v)));
    }

    fn br(&mut self, dest: Self::BasicBlock) {
        let bb = self.get_block();
        bb.mk_ty(IrBasicBlockTy::Merge { target: dest });

        self.mk_next_op(|op| op.mk_br());
    }

    fn cond_br(
        &mut self,
        cond: Self::Value,
        then_llbb: Self::BasicBlock,
        else_llbb: Self::BasicBlock,
    ) {
        let bb = self.get_block();
        bb.mk_ty(IrBasicBlockTy::Split {
            true_target: then_llbb,
            false_target: else_llbb,
        });

        let cond = self.mk_op(cond);
        let op = self.mk_next_op(|op| op.mk_br_cond(cond));
    }

    fn switch(
        &mut self,
        v: Self::Value,
        else_llbb: Self::BasicBlock,
        cases: impl ExactSizeIterator<Item = (u128, Self::BasicBlock)>,
    ) {
        let bb = self.get_block();
        bb.mk_ty(IrBasicBlockTy::Switch {
            default_target: else_llbb,
            cases: &cases.collect::<Vec<_>>(),
        });

        let v = self.mk_op(v);
        let op = self.mk_next_op(|op| op.mk_br_switch(v));
    }

    fn invoke(
        &mut self,
        llty: Self::Type,
        fn_attrs: Option<&CodegenFnAttrs>,
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
        // TODO: make JCC unreachable BB type
        self.ret_void();
    }

    fn add(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Add, var_ty, lhs, rhs))
            .into()
    }

    fn fadd(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fadd, var_ty, lhs, rhs))
            .into()
    }

    fn fadd_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fadd, var_ty, lhs, rhs))
            .into()
    }

    fn fadd_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fadd, var_ty, lhs, rhs))
            .into()
    }

    fn sub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Sub, var_ty, lhs, rhs))
            .into()
    }

    fn fsub(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fsub, var_ty, lhs, rhs))
            .into()
    }

    fn fsub_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fsub, var_ty, lhs, rhs))
            .into()
    }

    fn fsub_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fsub, var_ty, lhs, rhs))
            .into()
    }

    fn mul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Mul, var_ty, lhs, rhs))
            .into()
    }

    fn fmul(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fmul, var_ty, lhs, rhs))
            .into()
    }

    fn fmul_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fmul, var_ty, lhs, rhs))
            .into()
    }

    fn fmul_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fmul, var_ty, lhs, rhs))
            .into()
    }

    fn udiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Udiv, var_ty, lhs, rhs))
            .into()
    }

    fn exactudiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        // FIXME: what is this op?
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Udiv, var_ty, lhs, rhs))
            .into()
    }

    fn sdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Sdiv, var_ty, lhs, rhs))
            .into()
    }

    fn exactsdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        // FIXME: what is this op?
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Sdiv, var_ty, lhs, rhs))
            .into()
    }

    fn fdiv(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fdiv, var_ty, lhs, rhs))
            .into()
    }

    fn fdiv_fast(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fdiv, var_ty, lhs, rhs))
            .into()
    }

    fn fdiv_algebraic(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Fdiv, var_ty, lhs, rhs))
            .into()
    }

    fn urem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Umod, var_ty, lhs, rhs))
            .into()
    }

    fn srem(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Smod, var_ty, lhs, rhs))
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
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Lshift, var_ty, lhs, rhs))
            .into()
    }

    fn lshr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Urshift, var_ty, lhs, rhs))
            .into()
    }

    fn ashr(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Srshift, var_ty, lhs, rhs))
            .into()
    }

    fn and(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::And, var_ty, lhs, rhs))
            .into()
    }

    fn or(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Or, var_ty, lhs, rhs))
            .into()
    }

    fn xor(&mut self, lhs: Self::Value, rhs: Self::Value) -> Self::Value {
        let var_ty = lhs.var_ty();
        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(IrBinOpTy::Xor, var_ty, lhs, rhs))
            .into()
    }

    fn neg(&mut self, v: Self::Value) -> Self::Value {
        let var_ty = v.var_ty();
        let v = self.mk_op(v);
        self.mk_next_op(|op| op.mk_unnop(IrUnOpTy::Neg, var_ty, v))
            .into()
    }

    fn fneg(&mut self, v: Self::Value) -> Self::Value {
        let var_ty = v.var_ty();
        let v = self.mk_op(v);
        self.mk_next_op(|op| op.mk_unnop(IrUnOpTy::Fneg, var_ty, v))
            .into()
    }

    fn not(&mut self, v: Self::Value) -> Self::Value {
        let var_ty = v.var_ty();

        // TODO: jcc should correctly handle ~ on i1 as !
        let ty = if var_ty.is_i1() {
            IrUnOpTy::LogNot
        } else {
            IrUnOpTy::Not
        };

        let v = self.mk_op(v);
        self.mk_next_op(|op| op.mk_unnop(ty, var_ty, v)).into()
    }

    fn checked_binop(
        &mut self,
        oop: OverflowOp,
        ty: Ty<'_>,
        lhs: Self::Value,
        rhs: Self::Value,
    ) -> (Self::Value, Self::Value) {
        // FIXME: actually check
        let op = match oop {
            OverflowOp::Add => self.add(lhs, rhs),
            OverflowOp::Sub => self.sub(lhs, rhs),
            OverflowOp::Mul => self.mul(lhs, rhs),
        };

        (op, IrBuildValue::cnst_int(self.type_i1(), 0))
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

        let bb = self.get_block();
        let lcl = bb.func().add_local(ty);

        // TODO: should we alloc new stmt here?
        let op = self.mk_next_op(|op| op.mk_addr_lcl(lcl));

        op.into()
    }

    fn dynamic_alloca(&mut self, size: Self::Value, align: rustc_abi::Align) -> Self::Value {
        todo!("dynamic alloca not supported")
    }

    fn load(&mut self, ty: Self::Type, ptr: Self::Value, align: rustc_abi::Align) -> Self::Value {
        let ptr = self.mk_op(ptr);
        self.mk_next_op(|op| op.mk_load_addr(ty, ptr)).into()
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
        // FIXME: faked
        self.load(ty, ptr, rustc_abi::Align::EIGHT)
    }

    fn load_operand(
        &mut self,
        place: PlaceRef<'tcx, Self::Value>,
    ) -> OperandRef<'tcx, Self::Value> {
        if place.layout.is_unsized() {
            let tail = self
                .tcx
                .struct_tail_for_codegen(place.layout.ty, self.typing_env());
            if matches!(tail.kind(), TyKind::Foreign(..)) {
                // Unsized locals and, at least conceptually, even unsized arguments must be copied
                // around, which requires dynamically determining their size. Therefore, we cannot
                // allow `extern` types here. Consult t-opsem before removing this check.
                panic!("unsized locals must not be `extern` types");
            }
        }
        assert_eq!(place.val.llextra.is_some(), place.layout.is_unsized());
        if place.layout.is_zst() {
            return OperandRef::zero_sized(place.layout);
        }

        fn scalar_load_metadata<'a, 'll, 'tcx>(
            bx: &mut Builder<'a, 'tcx>,
            load: IrBuildValue,
            scalar: rustc_abi::Scalar,
            layout: TyAndLayout<'tcx>,
            offset: rustc_abi::Size,
        ) {
            if bx.cx.sess().opts.optimize == OptLevel::No {
                // Don't emit metadata we're not going to use
                return;
            }

            if !scalar.is_uninit_valid() {
                // bx.noundef_metadata(load);
            }

            match scalar.primitive() {
                rustc_abi::Primitive::Int(..) => {
                    if !scalar.is_always_valid(bx) {
                        bx.range_metadata(load, scalar.valid_range(bx));
                    }
                }
                rustc_abi::Primitive::Pointer(_) => {
                    if !scalar.valid_range(bx).contains(0) {
                        bx.nonnull_metadata(load);
                    }

                    if let Some(pointee) = layout.pointee_info_at(bx, offset) {
                        if let Some(_) = pointee.safe {
                            // bx.align_metadata(load, pointee.align);
                        }
                    }
                }
                rustc_abi::Primitive::Float(_) => {}
            }
        }

        let val = if let Some(_) = place.val.llextra {
            // FIXME: Merge with the `else` below?
            OperandValue::Ref(place.val)
        } else if self.is_backend_immediate(place.layout) {
            let llty = ty_to_jcc_ty(self.cx, &place.layout);

            let const_llval = None;
            // TODO: try read constant
            // unsafe {
            //     if let IrBuildValue::GlbAddr { glb, .. } = place.val.llval {
            //         if llvm::LLVMIsGlobalConstant(global) == llvm::True {
            //             if let Some(init) = llvm::LLVMGetInitializer(global) {
            //                 if self.val_ty(init) == llty {
            //                     const_llval = Some(init);
            //                 }
            //             }
            //         }
            //     }
            // }

            let llval = const_llval.unwrap_or_else(|| {
                let load = self.load(llty, place.val.llval, place.val.align);

                if let rustc_abi::BackendRepr::Scalar(scalar) = place.layout.backend_repr {
                    scalar_load_metadata(self, load, scalar, place.layout, rustc_abi::Size::ZERO);
                    self.to_immediate_scalar(load, scalar)
                } else {
                    load
                }
            });
            OperandValue::Immediate(llval)
        } else if let rustc_abi::BackendRepr::ScalarPair(a, b) = place.layout.backend_repr {
            let b_offset = a.size(self).align_to(b.align(self).abi);
            let mut load = |i, scalar: rustc_abi::Scalar, layout, align, offset| {
                let llptr = if i == 0 {
                    place.val.llval
                } else {
                    self.inbounds_ptradd(place.val.llval, self.const_usize(b_offset.bytes()))
                };

                let llty = scalar_pair_element_to_jcc_ty(self, &place.layout, i, false);
                let load = self.load(llty, llptr, align);

                scalar_load_metadata(self, load, scalar, layout, offset);
                self.to_immediate_scalar(load, scalar)
            };

            OperandValue::Pair(
                load(0, a, place.layout, place.val.align, rustc_abi::Size::ZERO),
                load(
                    1,
                    b,
                    place.layout,
                    place.val.align.restrict_for_offset(b_offset),
                    b_offset,
                ),
            )
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
        // nop
    }

    fn nonnull_metadata(&mut self, load: Self::Value) {
        // nop
    }

    fn store(
        &mut self,
        val: Self::Value,
        ptr: Self::Value,
        align: rustc_abi::Align,
    ) -> Self::Value {
        let ptr = self.mk_op(ptr);
        let val = self.mk_op(val);

        self.mk_next_op(|op| op.mk_store_addr(ptr, val)).into()
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
        // okay we are gonna fake this for noe
        // FIXME: impl here + jcc
        self.store(val, ptr, rustc_abi::Align::EIGHT);
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

        let base = self.mk_op(base);
        let index = self.mk_op(index);

        self.mk_next_op(|op| op.mk_addr_offset(AddrOffset::index(base, index, scale)))
            .into()
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
        let src_ty = val.var_ty();
        if dest_ty.is_i1() && src_ty.is_i8() {
            // same size, but we need to generate an AND
            // TODO: make jcc support `i1 = trunc i8`
            // FIXME: this AND will not have right type!
            let mask = self.mk_next_op(|op| op.mk_cnst_int(src_ty, 1));
            return self.and(val, mask.into());
        }

        let val = self.mk_op(val);
        self.mk_next_op(|op| op.mk_trunc(dest_ty, val)).into()
    }

    fn sext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        if val.var_ty().is_i1() {
            bug!("sext i1 is meaninglesss");
        }

        let val = self.mk_op(val);
        self.mk_next_op(|op| op.mk_sext(dest_ty, val)).into()
    }

    fn zext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let src_ty = val.var_ty();
        if dest_ty.is_i8() && src_ty.is_i1() {
            return self.nop_cast(val, dest_ty);
        }

        let val = self.mk_op(val);
        self.mk_next_op(|op| op.mk_zext(dest_ty, val)).into()
    }

    fn fptoui_sat(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn fptosi_sat(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        todo!()
    }

    fn fptoui(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let val = self.mk_op(val);
        self.mk_next_op(|op| op.mk_uconv(dest_ty, val)).into()
    }

    fn fptosi(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let val = self.mk_op(val);
        self.mk_next_op(|op| op.mk_sconv(dest_ty, val)).into()
    }

    fn uitofp(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let val = self.mk_op(val);
        self.mk_next_op(|op| op.mk_uconv(dest_ty, val)).into()
    }

    fn sitofp(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let val = self.mk_op(val);
        self.mk_next_op(|op| op.mk_sconv(dest_ty, val)).into()
    }

    fn fptrunc(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let val = self.mk_op(val);
        self.mk_next_op(|op| op.mk_conv(dest_ty, val)).into()
    }

    fn fpext(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let val = self.mk_op(val);
        self.mk_next_op(|op| op.mk_conv(dest_ty, val)).into()
    }

    fn ptrtoint(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let val = self.mk_op(val);
        self.mk_next_op(|op| op.mk_mov(val.var_ty(), val)).into()
    }

    fn inttoptr(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let val = self.mk_op(val);
        self.mk_next_op(|op| op.mk_mov(val.var_ty(), val)).into()
    }

    fn bitcast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        let val = self.mk_op(val);
        self.mk_next_op(|op| op.mk_mov(val.var_ty(), val)).into()
    }

    fn intcast(&mut self, val: Self::Value, dest_ty: Self::Type, is_signed: bool) -> Self::Value {
        let src_ty = val.var_ty();
        debug_assert!(src_ty.is_int() && dest_ty.is_int());

        match (dest_ty.int_cmp(&src_ty), is_signed) {
            (Ordering::Greater, true) => self.sext(val, dest_ty),
            (Ordering::Greater, false) => self.zext(val, dest_ty),
            (Ordering::Less, _) => self.trunc(val, dest_ty),
            _ => val,
        }
    }

    fn pointercast(&mut self, val: Self::Value, dest_ty: Self::Type) -> Self::Value {
        self.nop_cast(val, dest_ty)
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

        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(ty, self.type_i1(), lhs, rhs))
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

        let lhs = self.mk_op(lhs);
        let rhs = self.mk_op(rhs);
        self.mk_next_op(|op| op.mk_binop(ty, self.type_i1(), lhs, rhs))
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

        let src = self.mk_op(src);
        let dst = self.mk_op(dst);
        let op = self.mk_next_op(|op| op.mk_memcpy(src, dst, size));
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

        let ptr = self.mk_op(ptr);
        let op = self.mk_next_op(|op| op.mk_memset(ptr, fill_byte, size));
    }

    fn select(
        &mut self,
        cond: Self::Value,
        then_val: Self::Value,
        else_val: Self::Value,
    ) -> Self::Value {
        let cond = self.mk_op(cond);
        let then_val = self.mk_op(then_val);
        let else_val = self.mk_op(else_val);

        if cond.var_ty().is_i1()
            && then_val.get_int_cnst().is_some_and(|c| c.val == 1)
            && else_val.get_int_cnst().is_some_and(|c| c.val == 0)
        {
            return self.zext(cond.into(), then_val.var_ty());
        }

        if cond.var_ty().is_i1()
            && then_val.get_int_cnst().is_some_and(|c| c.val == 0)
            && else_val.get_int_cnst().is_some_and(|c| c.val == 1)
        {
            let not = self.not(cond.into());
            return self.zext(not, then_val.var_ty());
        }

        let var_ty = then_val.var_ty();

        self.mk_next_op(|op| op.mk_select(var_ty, cond, then_val, else_val))
            .into()
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
        // pairs are a lie, just a load
        // we _know_ offset is ptr-size

        let agg_val = self.mk_op(agg_val);
        let addr = {
            match idx {
                0 => agg_val,
                1 => self.mk_next_op(|op| {
                    op.mk_addr_offset(AddrOffset::offset(agg_val, self.ptr_size().get()))
                }),
                _ => bug!("expected pair"),
            }
        };

        self.load(self.type_ptr(), addr.into(), rustc_abi::Align::EIGHT)
    }

    fn insert_value(&mut self, agg_val: Self::Value, elt: Self::Value, idx: u64) -> Self::Value {
        let agg_val = self.mk_op(agg_val);
        let addr = {
            match idx {
                0 => agg_val,
                1 => self.mk_next_op(|op| {
                    op.mk_addr_offset(AddrOffset::offset(agg_val, self.ptr_size().get()))
                }),
                _ => bug!("expected pair"),
            }
        };

        self.store(elt, addr.into(), rustc_abi::Align::EIGHT);
        agg_val.into()
    }

    fn set_personality_fn(&mut self, personality: Self::Value) {
        // NOP
    }

    fn cleanup_landing_pad(&mut self, pers_fn: Self::Value) -> (Self::Value, Self::Value) {
        todo!()
    }

    fn filter_landing_pad(&mut self, pers_fn: Self::Value) -> (Self::Value, Self::Value) {
        todo!()
    }

    fn resume(&mut self, exn0: Self::Value, exn1: Self::Value) {
        self.unreachable();
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
        // TODO:
    }

    fn lifetime_start(&mut self, ptr: Self::Value, size: rustc_abi::Size) {}

    fn lifetime_end(&mut self, ptr: Self::Value, size: rustc_abi::Size) {}

    fn call(
        &mut self,
        llty: Self::Type,
        fn_attrs: Option<&CodegenFnAttrs>,
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

        let ret = match (fn_abi, instance) {
            (Some(f), Some(instance)) => self.abi_of(instance).2,
            _ => {
                let Some(fun) = llty.fun() else {
                    bug!("could not deduce return type (fn_abi None, and llty was not fun ty)");
                };

                fun.ret
            }
        };

        let target = self.mk_op(llfn);
        let args = args.iter().map(|&a| self.mk_op(a)).collect::<Vec<_>>();

        let op = self.mk_next_op(|op| op.mk_call(llty, target, &args, ret));

        if ret.is_aggregate() {
            // need to spill
            let lcl = self.func.add_local(ret);
            let addr = self.mk_next_op(|op| op.mk_addr_lcl(lcl));
            self.store(op.into(), addr.into(), rustc_abi::Align::EIGHT);

            addr.into()
        } else {
            op.into()
        }
    }

    fn apply_attrs_to_cleanup_callsite(&mut self, llret: Self::Value) {
        todo!()
    }
}
