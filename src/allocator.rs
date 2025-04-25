use rustc_ast::expand::allocator::{
    ALLOCATOR_METHODS, AllocatorKind, AllocatorTy, NO_ALLOC_SHIM_IS_UNSTABLE,
    alloc_error_handler_name, default_fn_name, global_fn_name,
};
use rustc_codegen_ssa::traits::BaseTypeCodegenMethods;
use rustc_middle::bug;
use rustc_middle::mir::mono::{Linkage, Visibility};
use rustc_session::config::OomStrategy;
use rustc_symbol_mangling::mangle_internal_symbol;
use rustc_target::spec::SymbolVisibility;

use crate::driver::CodegenCx;
use crate::jcc::ir::{IrIntCnst, IrLinkage, IrVarTy, IrVarTyFuncFlags};

pub(crate) fn codegen(
    cx: &mut CodegenCx<'_>,
    _module_name: &str,
    kind: AllocatorKind,
    alloc_error_handler_kind: AllocatorKind,
) {
    let usize = match cx.tcx.sess.target.pointer_width {
        16 => cx.type_i16(),
        32 => cx.type_i32(),
        64 => cx.type_i64(),
        tws => bug!("Unsupported target word size for int: {}", tws),
    };

    let ptr_ty = cx.type_ptr();

    if kind == AllocatorKind::Default {
        for method in ALLOCATOR_METHODS {
            let mut args = Vec::with_capacity(method.inputs.len());

            for input in method.inputs.iter() {
                match input.ty {
                    AllocatorTy::Layout => {
                        args.push(usize);
                        args.push(usize);
                    }
                    AllocatorTy::Ptr => args.push(ptr_ty),
                    AllocatorTy::Usize => args.push(usize),

                    AllocatorTy::ResultPtr | AllocatorTy::Unit => panic!("invalid allocator arg"),
                }
            }

            let ret = match method.output {
                AllocatorTy::ResultPtr => cx.type_ptr(),
                AllocatorTy::Unit => cx.type_none(),

                AllocatorTy::Layout | AllocatorTy::Usize | AllocatorTy::Ptr => {
                    panic!("invalid allocator output")
                }
            };

            let from_name = mangle_internal_symbol(cx.tcx, &global_fn_name(method.name));
            let to_name = mangle_internal_symbol(cx.tcx, &default_fn_name(method.name));

            create_wrapper_function(cx, &from_name, &to_name, &args, &ret);
        }
    }

    let params = &[usize, usize];
    let ret = &cx.unit.var_ty_none();

    let err_handler_ty = cx.unit.var_ty_func(params, ret, IrVarTyFuncFlags::None);

    create_wrapper_function(
        cx,
        &mangle_internal_symbol(cx.tcx, "__rust_alloc_error_handler"),
        &mangle_internal_symbol(cx.tcx, alloc_error_handler_name(alloc_error_handler_kind)),
        params,
        ret,
    );

    let ty_i8 = cx.type_i8();
    let name = mangle_internal_symbol(cx.tcx, OomStrategy::SYMBOL);

    let global = cx
        .unit
        .add_global_def_var(err_handler_ty, Some(&name), IrLinkage::External);
    let var = global.var();

    let value = cx.tcx.sess.opts.unstable_opts.oom.should_panic();
    var.mk_int(ty_i8, IrIntCnst::from(value));

    let name = mangle_internal_symbol(cx.tcx, NO_ALLOC_SHIM_IS_UNSTABLE);
    let global = cx
        .unit
        .add_global_def_var(ty_i8, Some(&name), IrLinkage::External);

    let var = global.var();
    var.mk_int(ty_i8, IrIntCnst::from(0u32));
}

pub fn to_visibility(visibility: SymbolVisibility) -> Visibility {
    match visibility {
        SymbolVisibility::Hidden => Visibility::Hidden,
        SymbolVisibility::Protected => Visibility::Protected,
        SymbolVisibility::Interposable => Visibility::Default,
    }
}

fn create_wrapper_function(
    cx: &mut CodegenCx<'_>,
    from_name: &str,
    to_name: &str,
    params: &[IrVarTy],
    ret: &IrVarTy,
) {
    let void = cx.type_none();

    let caller = cx.declare_simple_fn(
        true,
        Linkage::External,
        to_visibility(cx.tcx.sess.default_visibility()),
        from_name,
        params,
        ret,
    );

    let fun = caller.func();

    if cx.tcx.sess.must_emit_unwind_tables() {
        todo!("unwind tables");
    }

    let callee = cx.declare_simple_fn(
        false,
        Linkage::External,
        Visibility::Hidden,
        to_name,
        params,
        ret,
    );

    let fun_ty = cx.unit.var_ty_func(params, ret, IrVarTyFuncFlags::None);

    let bb = fun.last().unwrap();

    let stmt = bb.alloc_stmt();

    let args = fun.mk_param_stmt().ops().collect::<Vec<_>>();

    let target = stmt.alloc_op();
    target.mk_addr_glb(callee);

    let call = stmt.alloc_op();
    call.mk_call(fun_ty, target, &args[..], *ret);

    let ret_op = stmt.alloc_op();
    if ret.is_none() {
        ret_op.mk_ret(None);
    } else {
        ret_op.mk_ret(Some(call));
    }
}
