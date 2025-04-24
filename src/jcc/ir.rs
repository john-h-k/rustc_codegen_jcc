use std::{
    alloc::{AllocError, Allocator, Layout},
    ffi::{CString, c_char, c_int},
    fmt::{Debug, Formatter},
    fs::File,
    io::{self, Stderr},
    mem::{self, MaybeUninit},
    num::NonZeroUsize,
    os::{
        fd::{AsFd, AsRawFd, IntoRawFd},
        raw::c_void,
    },
    ptr::{self, NonNull},
    slice,
};

use bitflags::bitflags;

use crate::jcc_sys::*;

use super::alloc::ArenaAllocRef;

unsafe extern "C" {
    fn fdopen(fd: i32, mode: *const c_char) -> *mut FILE;
    fn fflush(fp: *mut FILE) -> c_int;
}

struct OwnedCFile<T> {
    file: Option<T>,
    ptr: *mut FILE,
}

fn as_c_file(file: &impl AsRawFd) -> *mut FILE {
    let fd = file.as_raw_fd();
    let mode = CString::new("w").unwrap();
    unsafe { fdopen(fd, mode.as_ptr().cast()) }
}

impl<T: AsRawFd> OwnedCFile<T> {
    // e.g for use with `stderr` which you do _not_ close
    fn borrow(from: T) -> Self {
        Self {
            file: None,
            ptr: as_c_file(&from),
        }
    }
}

impl OwnedCFile<Stderr> {
    fn stderr() -> Self {
        Self::borrow(io::stderr())
    }
}

impl<T> Drop for OwnedCFile<T> {
    fn drop(&mut self) {
        unsafe {
            fflush(self.ptr);
        }
    }
}

impl Debug for ir_object {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "")?;
        let stderr = OwnedCFile::stderr();
        unsafe {
            debug_print_ir_object(stderr.ptr, self);
        }

        Ok(())
    }
}

pub enum IrIntTy {
    I1,
    I8,
    I16,
    I32,
    I64,
    I128,
}

pub enum IrFloatTy {
    F16,
    F32,
    F64,
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct IrVarTyFuncFlags: u32 {
        const None = IR_VAR_FUNC_TY_FLAG_NONE;
        const Variadic = IR_VAR_FUNC_TY_FLAG_VARIADIC;
    }
}

#[derive(Clone, Copy)]
pub struct IrUnit {
    ptr: NonNull<ir_unit>,
}

unsafe impl Sync for ir_var_ty {}

impl IrUnit {
    pub fn new(arena: ArenaAllocRef) -> Self {
        let unit = ir_unit {
            arena: arena.as_ptr(),
            // FIXME: target
            target: unsafe { &AARCH64_MACOS_TARGET },
            ..Default::default()
        };

        let ptr = unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(unit))) };

        Self { ptr }
    }

    pub fn var_ty_info(&self, var_ty: IrVarTy) -> IrVarTyInfo {
        let info = unsafe { ir_var_ty_info(self.as_mut_ptr(), &var_ty.0) };

        IrVarTyInfo {
            size: info.size,
            alignment: info.alignment,
        }
    }

    fn mk_arena(&self) -> ArenaAllocRef {
        ArenaAllocRef::from_raw(unsafe { self.ptr.as_ref().arena })
    }

    pub fn var_ty_none(&self) -> IrVarTy {
        IrVarTy(unsafe { IR_VAR_TY_NONE })
    }

    pub fn var_ty_pointer(&self) -> IrVarTy {
        IrVarTy(unsafe { IR_VAR_TY_POINTER })
    }

    pub fn var_ty_fat_pointer(&self) -> IrVarTy {
        // TODO: cache
        let fields = unsafe { [IR_VAR_TY_POINTER, IR_VAR_TY_POINTER] };
        let num_fields = fields.len();
        let fields = self.mk_arena().alloc_slice_copy(&fields);

        IrVarTy(ir_var_ty {
            ty: IR_VAR_TY_TY_STRUCT,
            _1: ir_var_ty__bindgen_ty_1 {
                aggregate: ir_var_aggregate_ty { fields, num_fields },
            },
        })
    }

    pub fn var_ty_bytes(&self, size: usize) -> IrVarTy {
        self.var_ty_array(&self.var_ty_integer(IrIntTy::I8), size)
    }

    pub fn var_ty_array(&self, IrVarTy(el): &IrVarTy, len: usize) -> IrVarTy {
        IrVarTy(unsafe { ir_var_ty_make_array(self.ptr.as_ref(), el, len) })
    }

    pub fn var_ty_func(
        &self,
        params: &[IrVarTy],
        ret: &IrVarTy,
        flags: IrVarTyFuncFlags,
    ) -> IrVarTy {
        let arena = self.mk_arena();
        let params_ptr = arena.alloc_slice_copy(params);
        let ret_ptr = arena.alloc_copy(ret);

        // SAFETY: IrVarTy is repr(transparent) over ir_var_ty
        let params_ptr = params_ptr.cast::<ir_var_ty>();
        let ret_ptr = ret_ptr.cast::<ir_var_ty>();

        IrVarTy(ir_var_ty {
            ty: IR_VAR_TY_TY_FUNC,
            _1: ir_var_ty__bindgen_ty_1 {
                func: ir_var_func_ty {
                    ret_ty: ret_ptr,
                    num_params: params.len(),
                    params: params_ptr,
                    flags: flags.bits(),
                },
            },
        })
    }

    pub fn var_ty_integer(&self, ty: IrIntTy) -> IrVarTy {
        IrVarTy(unsafe {
            match ty {
                IrIntTy::I1 => IR_VAR_TY_I1,
                IrIntTy::I8 => IR_VAR_TY_I8,
                IrIntTy::I16 => IR_VAR_TY_I16,
                IrIntTy::I32 => IR_VAR_TY_I32,
                IrIntTy::I64 => IR_VAR_TY_I64,
                IrIntTy::I128 => IR_VAR_TY_I128,
            }
        })
    }

    pub fn var_ty_float(&self, ty: IrFloatTy) -> IrVarTy {
        IrVarTy(unsafe {
            match ty {
                IrFloatTy::F16 => IR_VAR_TY_F16,
                IrFloatTy::F32 => IR_VAR_TY_F32,
                IrFloatTy::F64 => IR_VAR_TY_F64,
            }
        })
    }
}

impl AsIrRaw for IrUnit {
    type Raw = ir_unit;

    fn as_ir_object(&self) -> ir_object {
        panic!("ir_object has no ir_unit entry (todo?)")
    }

    fn as_mut_ptr(&self) -> *mut Self::Raw {
        self.ptr.as_ptr()
    }
}

impl FromIrRaw for IrUnit {
    fn from_non_null(ptr: NonNull<Self::Raw>) -> Self {
        Self { ptr }
    }
}

// TODO: make NonNull so Option<T> gets niche opt

pub trait AsIrRaw: Sized {
    type Raw;

    fn as_ir_object(&self) -> ir_object;
    fn as_mut_ptr(&self) -> *mut Self::Raw;
}

pub trait FromIrRaw: AsIrRaw {
    fn from_raw(ptr: *mut Self::Raw) -> Self {
        Self::from_non_null(NonNull::new(ptr).unwrap())
    }

    fn new(ptr: *mut Self::Raw) -> Option<Self> {
        NonNull::new(ptr).map(Self::from_non_null)
    }

    fn from_non_null(ptr: NonNull<Self::Raw>) -> Self;
}

macro_rules! ir_object_newtype {
    ($t:ident, $ty:ty, $obj:ident, $name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq)]
        #[repr(transparent)]
        pub struct $t(NonNull<$ty>);

        impl AsIrRaw for $t {
            type Raw = $ty;

            fn as_ir_object(&self) -> ir_object {
                ir_object {
                    ty: $obj,
                    _1: ir_object__bindgen_ty_1 {
                        $name: self.0.as_ptr(),
                    },
                }
            }

            fn as_mut_ptr(&self) -> *mut Self::Raw {
                self.0.as_ptr()
            }
        }

        impl AsIrRaw for Option<$t> {
            type Raw = $ty;

            fn as_ir_object(&self) -> ir_object {
                match self {
                    Some(obj) => obj.as_ir_object(),
                    None => panic!("none!"),
                }
            }

            fn as_mut_ptr(&self) -> *mut Self::Raw {
                match self {
                    Some($t(p)) => p.as_ptr(),
                    None => ptr::null_mut(),
                }
            }
        }

        impl FromIrRaw for $t {
            fn from_non_null(ptr: NonNull<Self::Raw>) -> Self {
                Self(ptr)
            }
        }

        impl std::fmt::Debug for $t {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                Debug::fmt(&self.as_ir_object(), f)
            }
        }
    };
}

ir_object_newtype!(IrGlb, ir_glb, IR_OBJECT_TY_GLB, glb);
ir_object_newtype!(IrFunc, ir_func, IR_OBJECT_TY_FUNC, func);
ir_object_newtype!(IrVar, ir_var, IR_OBJECT_TY_VAR, var);
ir_object_newtype!(
    IrBasicBlock,
    ir_basicblock,
    IR_OBJECT_TY_BASICBLOCK,
    basicblock
);
ir_object_newtype!(IrLcl, ir_lcl, IR_OBJECT_TY_LCL, lcl);
ir_object_newtype!(IrStmt, ir_stmt, IR_OBJECT_TY_STMT, stmt);
ir_object_newtype!(IrOp, ir_op, IR_OBJECT_TY_OP, op);

impl IrGlb {
    pub fn is_def(&self) -> bool {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        p.def_ty == IR_GLB_DEF_TY_DEFINED
    }

    pub fn var(&self) -> IrVar {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        unsafe {
            assert_eq!(p.ty, IR_GLB_TY_DATA, "expected ir_glb to be of ty data");

            assert!(!p._1.var.is_null(), "var was null (undef symbol)");

            IrVar::from_raw(p._1.var)
        }
    }

    pub fn func(&self) -> IrFunc {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        unsafe {
            assert_eq!(p.ty, IR_GLB_TY_FUNC, "expected ir_glb to be of ty func");

            assert!(!p._1.func.is_null(), "func was null (undef symbol)");

            IrFunc::from_raw(p._1.func)
        }
    }
}

pub trait HasNext: Sized {
    fn next(&self) -> Option<Self>;
}

impl HasNext for IrBasicBlock {
    fn next(&self) -> Option<Self> {
        Self::new(unsafe { self.0.as_ref().succ })
    }
}

impl HasNext for IrStmt {
    fn next(&self) -> Option<Self> {
        Self::new(unsafe { self.0.as_ref().succ })
    }
}

impl HasNext for IrOp {
    fn next(&self) -> Option<Self> {
        Self::new(unsafe { self.0.as_ref().succ })
    }
}

pub struct IterFunc<'a, Parent, T: FromIrRaw + HasNext> {
    func: &'a Parent,
    cur: Option<T>,
}

impl<'a, Parent, T: Copy + FromIrRaw + HasNext> Iterator for IterFunc<'a, Parent, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let Some(cur) = self.cur else {
            return None;
        };

        self.cur = cur.next();
        Some(cur)
    }
}

pub struct IrBytes<'a> {
    offset: usize,
    bytes: &'a [u8],
}

impl<'a> IrBytes<'a> {
    pub fn new(offset: usize, bytes: &'a [u8]) -> Self {
        Self { offset, bytes }
    }
}

pub fn mk_bytes_list(unit: IrUnit, bytes: &[u8]) -> ir_var_value_list {
    let arena = unit.mk_arena();

    let mut offsets = Vec::new_in(arena);
    let mut values = Vec::new_in(arena);
    let num_values = bytes.len();

    for (i, &byte) in bytes.iter().enumerate() {
        offsets.push(i);

        let value = ir_var_value {
            ty: IR_VAR_VALUE_TY_INT,
            var_ty: unsafe { IR_VAR_TY_I8 },
            _1: ir_var_value__bindgen_ty_1 {
                int_value: byte as _,
            },
        };

        values.push(value);
    }

    let values = values.leak().as_mut_ptr();
    let offsets = offsets.leak().as_mut_ptr();

    ir_var_value_list {
        values,
        offsets,
        num_values,
    }
}

impl IrVar {
    pub fn unit(&self) -> IrUnit {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrUnit::from_non_null(unsafe { NonNull::new_unchecked(p.unit) })
    }

    pub fn mk_const_bytes(&self, IrVarTy(var_ty): IrVarTy, bytes: &[IrBytes]) {
        let unit = self.unit();
        let arena = unit.mk_arena();

        let mut offsets = Vec::new_in(arena);
        let mut values = Vec::new_in(arena);
        let num_values = bytes.len();

        for chunk in bytes {
            let ty = unit.var_ty_bytes(chunk.bytes.len());

            offsets.push(chunk.offset);
            values.push(ir_var_value {
                ty: IR_VAR_VALUE_TY_VALUE_LIST,
                var_ty,
                _1: ir_var_value__bindgen_ty_1 {
                    value_list: mk_bytes_list(unit, chunk.bytes),
                },
            });
        }

        let values = values.leak().as_mut_ptr();
        let offsets = offsets.leak().as_mut_ptr();

        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        p.value = ir_var_value {
            ty: IR_VAR_VALUE_TY_VALUE_LIST,
            var_ty,
            _1: ir_var_value__bindgen_ty_1 {
                value_list: ir_var_value_list {
                    values,
                    offsets,
                    num_values,
                },
            },
        };
    }
}

impl IrFunc {
    pub fn arena(&self) -> ArenaAllocRef {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        ArenaAllocRef::from_raw(p.arena)
    }

    pub fn unit(&self) -> IrUnit {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrUnit::from_non_null(unsafe { NonNull::new_unchecked(p.unit) })
    }

    pub fn mk_param_stmt(&self) -> IrStmt {
        // TODO: ensure not called twice
        let bb = self.first().unwrap_or_else(|| self.alloc_basicblock());

        match bb.first() {
            Some(stmt) => {
                debug_assert!(stmt.is_params(), "stmt present but not params");
                stmt
            }
            None => {
                let stmt = bb.alloc_stmt();
                unsafe { stmt.0.as_ptr().as_mut_unchecked().flags |= IR_STMT_FLAG_PARAM };
                stmt
            }
        }
    }

    pub fn basicblocks<'a>(&'a self) -> IterFunc<'a, Self, IrBasicBlock> {
        IterFunc {
            func: self,
            cur: self.first(),
        }
    }

    pub fn first(&self) -> Option<IrBasicBlock> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrBasicBlock::new(p.first)
    }

    pub fn get_param_op(&self, idx: usize) -> IrOp {
        dbg!(&self);
        // FIXME: more efficient way to do this (JCC should probably have better UX here, probably a `ir_get_param_stmt` fn
        let stmt = self
            .first()
            .and_then(|b| b.first())
            .expect("get_param_op but no bb present");

        debug_assert!(stmt.is_params(), "no params stmt!");
        stmt.ops().nth(idx).expect("could not get param op")
    }

    pub fn alloc_basicblock(&self) -> IrBasicBlock {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        unsafe { IrBasicBlock::from_raw(ir_alloc_basicblock(p)) }
    }

    pub fn add_local(&self, IrVarTy(var_ty): IrVarTy) -> IrLcl {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrLcl::from_raw(unsafe { ir_add_local(p, &var_ty) })
    }

    pub fn add_param_local(&self, IrVarTy(var_ty): IrVarTy) -> IrLcl {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        let lcl = unsafe { ir_add_local(p, &var_ty) };
        (unsafe { *lcl }).flags |= IR_LCL_FLAG_PARAM;
        IrLcl::from_raw(lcl)
    }
}

pub enum IrBasicBlockTy {
    Ret,
    Merge {
        target: IrBasicBlock,
    },
    Split {
        true_target: IrBasicBlock,
        false_target: IrBasicBlock,
    },
}

impl IrBasicBlock {
    pub fn mk_ty(&self, ty: IrBasicBlockTy) {
        let f = self.func().as_mut_ptr();
        let p = self.as_mut_ptr();

        unsafe {
            match ty {
                IrBasicBlockTy::Ret => ir_make_basicblock_ret(f, p),
                IrBasicBlockTy::Merge { target } => {
                    ir_make_basicblock_merge(f, p, target.as_mut_ptr())
                }
                IrBasicBlockTy::Split {
                    true_target,
                    false_target,
                } => ir_make_basicblock_split(
                    f,
                    p,
                    true_target.as_mut_ptr(),
                    false_target.as_mut_ptr(),
                ),
            }
        }
    }

    pub fn comment(&self, comment: &[u8]) {
        let comment = self.func().arena().alloc_str(comment);

        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        p.comment = comment;
    }

    pub fn func(&self) -> IrFunc {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrFunc::from_raw(p.func)
    }

    pub fn first(&self) -> Option<IrStmt> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrStmt::new(p.first)
    }

    pub fn stmts<'a>(&'a self) -> IterFunc<'a, Self, IrStmt> {
        IterFunc {
            func: self,
            cur: self.first(),
        }
    }

    pub fn alloc_stmt(&self) -> IrStmt {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        unsafe {
            // FIXME: inefficient
            let fun = p.func;
            IrStmt::from_raw(ir_alloc_stmt(fun, p))
        }
    }
}

impl IrStmt {
    pub fn basicblock(&self) -> IrBasicBlock {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrBasicBlock::from_raw(p.basicblock)
    }

    pub fn func(&self) -> IrFunc {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        unsafe { IrFunc::from_raw((*p.basicblock).func) }
    }

    pub fn is_params(&self) -> bool {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        (p.flags & IR_STMT_FLAG_PARAM) != 0
    }

    pub fn is_phis(&self) -> bool {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        (p.flags & IR_STMT_FLAG_PHI) != 0
    }

    pub fn first(&self) -> Option<IrOp> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrOp::new(p.first)
    }

    pub fn ops<'a>(&'a self) -> IterFunc<'a, Self, IrOp> {
        IterFunc {
            func: self,
            cur: self.first(),
        }
    }

    pub fn alloc_op(&self) -> IrOp {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        unsafe {
            // FIXME: inefficient
            let fun = (*p.basicblock).func;
            IrOp::from_raw(ir_alloc_op(fun, p))
        }
    }
}

pub struct AddrOffset {
    base: IrOp,
    index: Option<IrOp>,
    scale: usize,
    offset: Option<NonZeroUsize>,
}

impl AddrOffset {
    pub fn offset(base: IrOp, offset: NonZeroUsize) -> Self {
        Self {
            base,
            offset: Some(offset),
            index: None,
            scale: 0,
        }
    }

    pub fn index(base: IrOp, index: IrOp, scale: NonZeroUsize) -> Self {
        Self {
            base,
            index: Some(index),
            scale: 0,
            offset: None,
        }
    }

    pub fn index_offset(
        base: IrOp,
        index: IrOp,
        scale: NonZeroUsize,
        offset: NonZeroUsize,
    ) -> Self {
        Self {
            base,
            index: Some(index),
            scale: scale.into(),
            offset: Some(offset),
        }
    }
}

pub struct IrIntCnst {
    pub val: u64,
}

impl IrOp {
    pub fn var_ty(&self) -> IrVarTy {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrVarTy(p.var_ty)
    }

    pub fn stmt(&self) -> IrStmt {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrStmt::from_raw(p.stmt)
    }

    pub fn basicblock(&self) -> IrBasicBlock {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        unsafe { IrBasicBlock::from_raw((*p.stmt).basicblock) }
    }

    pub fn func(&self) -> IrFunc {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        unsafe { IrFunc::from_raw((*(*p.stmt).basicblock).func) }
    }

    pub fn mk_call(&self, func_ty: IrVarTy, target: IrOp, args: &[IrOp], ret: IrVarTy) {
        unsafe {
            (*self.func().as_mut_ptr()).flags |= IR_FUNC_FLAG_MAKES_CALL;
        }

        let arena = self.func().unit().mk_arena();
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        let num_args = args.len();
        let args = arena.alloc_slice_copy(args);
        // SAFETY: repr(transparent)
        let args = args.cast::<*mut ir_op>();

        p.ty = IR_OP_TY_CALL;
        p.var_ty = ret.0;
        p._1.call = ir_op_call {
            func_ty: func_ty.0,
            target: target.as_mut_ptr(),
            num_args,
            args,
            arg_var_tys: ptr::null_mut(),

            ..Default::default()
        };
    }

    pub fn mk_ret(&self, value: Option<IrOp>) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        unsafe {
            let value = value.as_mut_ptr();

            p.ty = IR_OP_TY_RET;
            p.var_ty = IR_VAR_TY_NONE;
            p._1.ret = ir_op_ret { value };
        }
    }

    pub fn mk_cond_br(&self, cond: IrOp) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        unsafe {
            p.ty = IR_OP_TY_BR_COND;
            p.var_ty = IR_VAR_TY_NONE;
            p._1.br_cond = ir_op_br_cond {
                cond: cond.as_mut_ptr(),
            };
        }
    }

    pub fn mk_br(&self) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        unsafe {
            p.ty = IR_OP_TY_BR;
            p.var_ty = IR_VAR_TY_NONE;
        }
    }

    pub fn mk_lcl_param(&self, var_ty: IrVarTy) {
        debug_assert!(
            self.stmt().is_params(),
            "mk_lcl_param must be in param stmt"
        );

        let fun = self.func();
        let lcl = fun.add_param_local(var_ty);

        self.mk_addr_lcl(lcl);

        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        p.flags |= IR_OP_FLAG_PARAM;
    }

    pub fn mk_mov_param(&self, IrVarTy(var_ty): IrVarTy) {
        debug_assert!(
            self.stmt().is_params(),
            "mk_mov_param must be in param stmt"
        );

        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        p.ty = IR_OP_TY_MOV;
        p.var_ty = var_ty;
        p.flags |= IR_OP_FLAG_PARAM;
        p._1.mov = ir_op_mov {
            value: ptr::null_mut(),
        };
    }

    pub fn mk_cnst_int(&self, IrVarTy(var_ty): IrVarTy, value: u64) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        p.ty = IR_OP_TY_CNST;
        p.var_ty = var_ty;
        p._1.cnst = ir_op_cnst {
            ty: IR_OP_CNST_TY_INT,
            _1: ir_op_cnst__bindgen_ty_1 { int_value: value },
        };
    }

    pub fn mk_cnst_float(&self, IrVarTy(var_ty): IrVarTy, value: f64) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        p.ty = IR_OP_TY_CNST;
        p.var_ty = var_ty;
        p._1.cnst = ir_op_cnst {
            ty: IR_OP_CNST_TY_FLT,
            _1: ir_op_cnst__bindgen_ty_1 { flt_value: value },
        };
    }

    pub fn mk_store_addr(&self, addr: IrOp, value: IrOp) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        unsafe {
            p.ty = IR_OP_TY_STORE;
            p.var_ty = IR_VAR_TY_NONE;
            p._1.store = ir_op_store {
                ty: IR_OP_STORE_TY_ADDR,
                value: value.0.as_ptr(),
                _1: ir_op_store__bindgen_ty_1 {
                    addr: addr.0.as_ptr(),
                },
            };
        }
    }

    pub fn mk_addr_glb(&self, glb: IrGlb) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        unsafe {
            p.ty = IR_OP_TY_ADDR;
            p.var_ty = IR_VAR_TY_POINTER;
            p._1.addr = ir_op_addr {
                ty: IR_OP_ADDR_TY_GLB,
                _1: ir_op_addr__bindgen_ty_1 {
                    glb: glb.0.as_ptr(),
                },
            };
        }
    }

    pub fn mk_memset(&self, addr: IrOp, value: u8, length: usize) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        p.ty = IR_OP_TY_MEM_SET;
        unsafe {
            p.var_ty = IR_VAR_TY_NONE;
        }
        p._1.mem_set = ir_op_mem_set {
            addr: addr.as_mut_ptr(),
            value: value.into(),
            length,
        };
    }

    fn mk_cast<const CAST_TY: u32>(&self, IrVarTy(var_ty): IrVarTy, value: IrOp) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        p.ty = IR_OP_TY_CAST_OP;
        p.var_ty = var_ty;
        p._1.cast_op = ir_op_cast_op {
            ty: CAST_TY,
            value: value.as_mut_ptr(),
        };
    }

    pub fn mk_trunc(&self, var_ty: IrVarTy, value: IrOp) {
        self.mk_cast::<IR_OP_CAST_OP_TY_TRUNC>(var_ty, value)
    }

    pub fn mk_sext(&self, var_ty: IrVarTy, value: IrOp) {
        self.mk_cast::<IR_OP_CAST_OP_TY_SEXT>(var_ty, value)
    }

    pub fn mk_zext(&self, var_ty: IrVarTy, value: IrOp) {
        self.mk_cast::<IR_OP_CAST_OP_TY_ZEXT>(var_ty, value)
    }

    pub fn mk_conv(&self, var_ty: IrVarTy, value: IrOp) {
        self.mk_cast::<IR_OP_CAST_OP_TY_CONV>(var_ty, value)
    }

    pub fn mk_sconv(&self, var_ty: IrVarTy, value: IrOp) {
        self.mk_cast::<IR_OP_CAST_OP_TY_SCONV>(var_ty, value)
    }

    pub fn mk_uconv(&self, var_ty: IrVarTy, value: IrOp) {
        self.mk_cast::<IR_OP_CAST_OP_TY_UCONV>(var_ty, value)
    }

    fn chk_ty(&self) -> u32 {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        p.ty
    }

    pub fn get_int_cnst(&self) -> Option<IrIntCnst> {
        let IR_OP_CNST_TY_INT = self.chk_ty() else {
            return None;
        };

        unsafe {
            let p = self.0.as_ptr().as_mut_unchecked();
            let cnst = p._1.cnst;

            match cnst.ty {
                IR_OP_CNST_TY_INT => Some(IrIntCnst {
                    val: cnst._1.int_value,
                }),
                _ => None,
            }
        }
    }

    pub fn get_addr_glb(&self) -> Option<IrGlb> {
        let IR_OP_TY_ADDR = self.chk_ty() else {
            return None;
        };

        unsafe {
            let p = self.0.as_ptr().as_mut_unchecked();
            let addr = p._1.addr;

            match addr.ty {
                IR_OP_ADDR_TY_GLB => Some(IrGlb::from_raw(addr._1.glb)),
                _ => None,
            }
        }
    }

    pub fn mk_addr_lcl(&self, lcl: IrLcl) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        unsafe {
            p.ty = IR_OP_TY_ADDR;
            p.var_ty = IR_VAR_TY_POINTER;
            p._1.addr = ir_op_addr {
                ty: IR_OP_ADDR_TY_LCL,
                _1: ir_op_addr__bindgen_ty_1 {
                    lcl: lcl.0.as_ptr(),
                },
            };
        }
    }

    pub fn mk_addr_offset(&self, addr_offset: AddrOffset) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        unsafe {
            p.ty = IR_OP_TY_ADDR_OFFSET;
            p.var_ty = IR_VAR_TY_POINTER;
            p._1.addr_offset = ir_op_addr_offset {
                base: addr_offset.base.0.as_ptr(),
                index: addr_offset
                    .index
                    .map(|p| p.as_mut_ptr())
                    .unwrap_or(ptr::null_mut()),
                scale: addr_offset.scale,
                offset: addr_offset.offset.map(|o| o.into()).unwrap_or(0usize),
            };
        }
    }
}

impl IrUnit {
    pub fn add_global_def_var(&self, IrVarTy(var_ty): IrVarTy, name: Option<&str>) -> IrGlb {
        unsafe {
            let arena = self.mk_arena();
            let name = name.map(|n| arena.alloc_str(n)).unwrap_or(ptr::null());

            let unit = self.ptr.as_ptr().as_mut_unchecked();

            let glb = ir_add_global(unit, IR_GLB_TY_DATA, &var_ty, IR_GLB_DEF_TY_DEFINED, name);

            // TODO: linkage
            (*glb).linkage = IR_LINKAGE_EXTERNAL;
            (*glb)._1.var = arena.alloc::<ir_var>();

            // FIXME: can sometimes be const data/string
            let ty = IR_VAR_TY_DATA;

            *(*glb)._1.var = ir_var {
                unit,
                ty,
                var_ty,
                // TODO: jcc should have raw "bytes" rather than long list type
                value: ir_var_value {
                    ty: IR_VAR_VALUE_TY_ZERO,
                    var_ty,
                    _1: ir_var_value__bindgen_ty_1 { int_value: 0 },
                },
            };

            IrGlb(NonNull::new(glb).unwrap())
        }
    }

    pub fn add_global_undef_func(&self, IrVarTy(var_ty): IrVarTy, name: &str) -> IrGlb {
        unsafe {
            let arena = self.mk_arena();
            let name = arena.alloc_str(name);

            let unit = self.ptr.as_ptr().as_mut_unchecked();

            debug_assert!(var_ty.ty == IR_VAR_TY_TY_FUNC, "expected func ty");

            let glb = ir_add_global(unit, IR_GLB_TY_FUNC, &var_ty, IR_GLB_DEF_TY_UNDEFINED, name);

            // TODO: linkage
            (*glb).linkage = IR_LINKAGE_EXTERNAL;
            (*glb)._1.func = ptr::null_mut();

            IrGlb(NonNull::new(glb).unwrap())
        }
    }

    pub fn add_global_def_func(&self, IrVarTy(var_ty): IrVarTy, name: &str) -> IrGlb {
        unsafe {
            let arena = self.mk_arena();
            let name = arena.alloc_str(name);

            let unit = self.ptr.as_ptr().as_mut_unchecked();

            debug_assert!(var_ty.ty == IR_VAR_TY_TY_FUNC, "expected func ty");

            let glb = ir_add_global(unit, IR_GLB_TY_FUNC, &var_ty, IR_GLB_DEF_TY_DEFINED, name);

            // TODO: linkage
            (*glb).linkage = IR_LINKAGE_EXTERNAL;
            (*glb)._1.func = arena.alloc::<ir_func>();

            *(*glb)._1.func = ir_func {
                unit,
                func_ty: var_ty._1.func,
                name,
                arena: arena.as_ptr(),
                flags: IR_FUNC_FLAG_NONE,
                ..Default::default()
            };

            IrGlb(NonNull::new(glb).unwrap())
        }
    }
}

// TODO: expensive, take ref in more places
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct IrVarTy(ir_var_ty);

pub enum IrVarTyAggregateTy {
    Struct,
    Union,
}

pub struct IrVarTyAggregate<'a> {
    pub ty: IrVarTyAggregateTy,
    pub fields: &'a [IrVarTy],
}

pub struct IrVarTyInfo {
    pub size: usize,
    pub alignment: usize,
}

impl IrVarTy {
    pub fn ty_none() -> IrVarTy {
        IrVarTy(unsafe { IR_VAR_TY_NONE })
    }
    pub fn ty_pointer() -> IrVarTy {
        IrVarTy(unsafe { IR_VAR_TY_POINTER })
    }

    pub fn ty_i1() -> IrVarTy {
        IrVarTy(unsafe { IR_VAR_TY_I1 })
    }
    pub fn ty_i8() -> IrVarTy {
        IrVarTy(unsafe { IR_VAR_TY_I8 })
    }
    pub fn ty_i16() -> IrVarTy {
        IrVarTy(unsafe { IR_VAR_TY_I16 })
    }
    pub fn ty_i32() -> IrVarTy {
        IrVarTy(unsafe { IR_VAR_TY_I32 })
    }
    pub fn ty_i64() -> IrVarTy {
        IrVarTy(unsafe { IR_VAR_TY_I64 })
    }
    pub fn ty_i128() -> IrVarTy {
        IrVarTy(unsafe { IR_VAR_TY_I128 })
    }

    pub fn ty_f16() -> IrVarTy {
        IrVarTy(unsafe { IR_VAR_TY_F16 })
    }
    pub fn ty_f32() -> IrVarTy {
        IrVarTy(unsafe { IR_VAR_TY_F32 })
    }
    pub fn ty_f64() -> IrVarTy {
        IrVarTy(unsafe { IR_VAR_TY_F64 })
    }
    // const F128: Self = Self(unsafe { IR_VAR_TY_F128 });

    pub fn aggregate(&self) -> Option<IrVarTyAggregate> {
        let ty;
        match self.0.ty {
            IR_VAR_TY_TY_STRUCT => ty = IrVarTyAggregateTy::Struct,
            IR_VAR_TY_TY_UNION => ty = IrVarTyAggregateTy::Union,
            _ => return None,
        };

        // SAFETY: repr(transparent)
        let fields = unsafe {
            slice::from_raw_parts(
                self.0._1.aggregate.fields.cast::<IrVarTy>(),
                self.0._1.aggregate.num_fields,
            )
        };

        Some(IrVarTyAggregate { ty, fields })
    }

    pub fn is_aggregate(&self) -> bool {
        match self.0.ty {
            IR_VAR_TY_TY_STRUCT | IR_VAR_TY_TY_UNION => true,
            _ => false,
        }
    }

    pub fn is_array(&self) -> bool {
        match self.0.ty {
            IR_VAR_TY_TY_ARRAY => true,
            _ => false,
        }
    }
}

impl Debug for IrVarTy {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "")?;
        let stderr = OwnedCFile::stderr();
        unsafe {
            debug_print_var_ty_string(stderr.ptr, &self.0);
        }

        Ok(())
    }
}

impl PartialEq for IrVarTy {
    fn eq(&self, other: &Self) -> bool {
        unsafe { ir_var_ty_eq(&self.0, &other.0) }
    }
}

impl Eq for IrVarTy {}
