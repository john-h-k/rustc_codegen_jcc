use std::{
    cmp,
    ffi::{CString, c_char, c_int},
    fmt::{Debug, Formatter},
    io::{self, Stderr},
    marker::PhantomData,
    mem,
    num::NonZeroUsize,
    os::fd::AsRawFd,
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

struct FlushGuard<T> {
    file: Option<T>,
    ptr: *mut FILE,
}

fn as_c_file(file: &impl AsRawFd) -> *mut FILE {
    let fd = file.as_raw_fd();
    let mode = CString::new("w").unwrap();
    unsafe { fdopen(fd, mode.as_ptr().cast()) }
}

impl<T: AsRawFd> FlushGuard<T> {
    // e.g for use with `stderr` which you do _not_ close
    fn borrow(from: T) -> Self {
        Self {
            file: None,
            ptr: as_c_file(&from),
        }
    }
}

impl FlushGuard<Stderr> {
    fn stderr() -> Self {
        Self::borrow(io::stderr())
    }
}

impl<T> Drop for FlushGuard<T> {
    fn drop(&mut self) {
        unsafe {
            fflush(self.ptr);
        }
    }
}

impl Debug for ir_object {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        let stderr = FlushGuard::stderr();
        unsafe {
            debug_print_ir_object(stderr.ptr, self, &debug_print_ir_opts::default());
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
pub struct IrUnit<'jcc> {
    ptr: NonNull<ir_unit>,
    _data: PhantomData<&'jcc ir_unit>,
}

unsafe impl Sync for ir_var_ty {}

impl<'jcc> IrUnit<'jcc> {
    pub fn new(arena: ArenaAllocRef) -> Self {
        let unit = ir_unit {
            arena: arena.as_ptr(),
            // FIXME: target
            target: unsafe { &AARCH64_MACOS_TARGET },
            ..Default::default()
        };

        let ptr = unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(unit))) };

        Self {
            ptr,
            _data: PhantomData,
        }
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

    pub fn var_ty_none(&self) -> IrVarTy<'jcc> {
        IrVarTy::from_raw(unsafe { IR_VAR_TY_NONE })
    }

    pub fn var_ty_pointer(&self) -> IrVarTy<'jcc> {
        IrVarTy::from_raw(unsafe { IR_VAR_TY_POINTER })
    }

    pub fn var_ty_fat_pointer(&self) -> IrVarTy<'jcc> {
        // TODO: cache
        let fields = unsafe { [IR_VAR_TY_POINTER, IR_VAR_TY_POINTER] };
        let num_fields = fields.len();
        let fields = self.mk_arena().alloc_slice_copy(&fields);

        IrVarTy::from_raw(ir_var_ty {
            ty: IR_VAR_TY_TY_STRUCT,
            _1: ir_var_ty__bindgen_ty_1 {
                aggregate: ir_var_aggregate_ty { fields, num_fields },
            },
        })
    }

    pub fn var_ty_bytes(&self, size: usize) -> IrVarTy<'jcc> {
        self.var_ty_array(&self.var_ty_integer(IrIntTy::I8), size)
    }

    pub fn var_ty_array(&self, IrVarTy(el, ..): &IrVarTy, len: usize) -> IrVarTy<'jcc> {
        IrVarTy::from_raw(unsafe { ir_var_ty_mk_array(self.ptr.as_ref(), el, len) })
    }

    pub fn var_ty_union(&self, fields: &[IrVarTy]) -> IrVarTy<'jcc> {
        let arena = self.mk_arena();

        let num_fields = fields.len();

        let fields = arena.alloc_slice_copy(fields);
        // SAFETY: IrVarTy is repr(transparent) over ir_var_ty
        let fields = fields.cast::<ir_var_ty>();

        IrVarTy::from_raw(ir_var_ty {
            ty: IR_VAR_TY_TY_UNION,
            _1: ir_var_ty__bindgen_ty_1 {
                aggregate: ir_var_aggregate_ty { num_fields, fields },
            },
        })
    }

    pub fn var_ty_struct(&self, fields: &[IrVarTy]) -> IrVarTy<'jcc> {
        let arena = self.mk_arena();

        let num_fields = fields.len();

        let fields = arena.alloc_slice_copy(fields);
        // SAFETY: IrVarTy is repr(transparent) over ir_var_ty
        let fields = fields.cast::<ir_var_ty>();

        IrVarTy::from_raw(ir_var_ty {
            ty: IR_VAR_TY_TY_STRUCT,
            _1: ir_var_ty__bindgen_ty_1 {
                aggregate: ir_var_aggregate_ty { num_fields, fields },
            },
        })
    }

    pub fn var_ty_func(
        &self,
        params: &[IrVarTy],
        ret: &IrVarTy,
        flags: IrVarTyFuncFlags,
    ) -> IrVarTy<'jcc> {
        let arena = self.mk_arena();
        let params_ptr = arena.alloc_slice_copy(params);
        let ret_ptr = arena.alloc_copy(ret);

        // SAFETY: IrVarTy is repr(transparent) over ir_var_ty
        let params_ptr = params_ptr.cast::<ir_var_ty>();
        let ret_ptr = ret_ptr.cast::<ir_var_ty>();

        IrVarTy::from_raw(ir_var_ty {
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

    pub fn var_ty_integer(&self, ty: IrIntTy) -> IrVarTy<'jcc> {
        IrVarTy::from_raw(unsafe {
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

    pub fn var_ty_float(&self, ty: IrFloatTy) -> IrVarTy<'jcc> {
        IrVarTy::from_raw(unsafe {
            match ty {
                IrFloatTy::F16 => IR_VAR_TY_F16,
                IrFloatTy::F32 => IR_VAR_TY_F32,
                IrFloatTy::F64 => IR_VAR_TY_F64,
            }
        })
    }
}

impl AsIrRaw for IrUnit<'_> {
    type Raw = ir_unit;

    fn as_ir_object(&self) -> ir_object {
        panic!("ir_object has no ir_unit entry (todo?)")
    }

    fn as_mut_ptr(&self) -> *mut Self::Raw {
        self.ptr.as_ptr()
    }
}

impl FromIrRaw for IrUnit<'_> {
    type Raw = ir_unit;

    fn from_non_null(ptr: NonNull<Self::Raw>) -> Self {
        Self {
            ptr,
            _data: PhantomData,
        }
    }
}

// TODO: make NonNull so Option<T> gets niche opt

pub trait AsIrRaw: Sized {
    type Raw;

    fn as_ir_object(&self) -> ir_object;
    fn as_mut_ptr(&self) -> *mut Self::Raw;
}

pub trait IrId {
    fn id(&self) -> usize;
}

pub trait IrComment {
    fn comment(&self, comment: &[u8]);
}

pub trait FromIrRaw: Sized {
    type Raw;

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
        #[derive(Clone, Copy)]
        #[repr(transparent)]
        pub struct $t<'jcc>(NonNull<$ty>, PhantomData<&'jcc $ty>);

        impl PartialEq for $t<'_> {
            fn eq(&self, other: &Self) -> bool {
                ptr::eq(self.as_mut_ptr(), other.as_mut_ptr())
            }
        }

        impl Eq for $t<'_> {}

        impl AsIrRaw for $t<'_> {
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

        impl AsIrRaw for Option<$t<'_>> {
            type Raw = $ty;

            fn as_ir_object(&self) -> ir_object {
                match self {
                    Some(obj) => obj.as_ir_object(),
                    None => panic!("none!"),
                }
            }

            fn as_mut_ptr(&self) -> *mut Self::Raw {
                match self {
                    Some($t(p, ..)) => p.as_ptr(),
                    None => ptr::null_mut(),
                }
            }
        }

        impl FromIrRaw for $t<'_> {
            type Raw = $ty;

            fn from_non_null(ptr: NonNull<Self::Raw>) -> Self {
                Self(ptr, PhantomData)
            }
        }

        impl std::fmt::Debug for $t<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                Debug::fmt(&self.as_ir_object(), f)
            }
        }
    };
}

macro_rules! ir_object_id {
    ($t:ident) => {
        impl IrId for $t<'_> {
            fn id(&self) -> usize {
                unsafe { (*self.0.as_ptr()).id }
            }
        }
    };
}

macro_rules! ir_object_comment {
    ($t:ident) => {
        impl IrComment for $t<'_> {
            fn comment(&self, comment: &[u8]) {
                let comment = self.func().arena().alloc_str(comment);

                let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
                // debug_assert!(p.comment.is_null(), "comment already set!");
                p.comment = comment;
            }
        }
    };
}

ir_object_comment!(IrBasicBlock);
ir_object_comment!(IrStmt);
ir_object_comment!(IrOp);

ir_object_id!(IrGlb);
ir_object_id!(IrLcl);
ir_object_id!(IrBasicBlock);
ir_object_id!(IrStmt);
ir_object_id!(IrOp);

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

impl<'jcc> IrGlb<'jcc> {
    pub fn is_def(&self) -> bool {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        p.def_ty == IR_GLB_DEF_TY_DEFINED
    }

    pub fn mk_def(&self) {
        unsafe {
            let p = self.0.as_ptr().as_mut_unchecked();
            let unit = self.unit();
            let arena = unit.mk_arena();

            let var_ty = p.var_ty;
            let name = p.name;

            // TODO: linkage
            (*p).def_ty = IR_GLB_DEF_TY_DEFINED;
            (*p)._1.func = arena.alloc::<ir_func>();

            *(*p)._1.func = ir_func {
                unit: unit.as_mut_ptr(),
                func_ty: var_ty._1.func,
                name,
                arena: arena.as_ptr(),
                flags: IR_FUNC_FLAG_NONE,
                ..Default::default()
            };
        }
    }

    pub fn unit(&self) -> IrUnit {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrUnit::from_raw(p.unit)
    }

    pub fn var(&self) -> IrVar<'jcc> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        unsafe {
            assert_eq!(p.ty, IR_GLB_TY_DATA, "expected ir_glb to be of ty data");

            assert!(!p._1.var.is_null(), "var was null (undef symbol)");

            IrVar::from_raw(p._1.var)
        }
    }

    pub fn is_func(&self) -> bool {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        p.ty == IR_GLB_TY_FUNC
    }

    pub fn func(&self) -> IrFunc<'jcc> {
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

impl HasNext for IrBasicBlock<'_> {
    fn next(&self) -> Option<Self> {
        Self::new(unsafe { self.0.as_ref().succ })
    }
}

impl HasNext for IrStmt<'_> {
    fn next(&self) -> Option<Self> {
        Self::new(unsafe { self.0.as_ref().succ })
    }
}

impl HasNext for IrOp<'_> {
    fn next(&self) -> Option<Self> {
        Self::new(unsafe { self.0.as_ref().succ })
    }
}

// NOTE: dangerous to iterate this while mutating
pub struct IterFunc<T: FromIrRaw + HasNext> {
    cur: Option<T>,
    sz: usize,
}

impl<T: Copy + FromIrRaw + HasNext> ExactSizeIterator for IterFunc<T> {}

impl<T: Copy + FromIrRaw + HasNext> Iterator for IterFunc<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let cur = self.cur?;

        self.cur = cur.next();
        Some(cur)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.sz, Some(self.sz))
    }
}

pub struct IrVarValueListEl<'jcc> {
    pub value: IrVarValue<'jcc>,
    pub offset: usize,
}

pub struct IrVarValueAddr<'jcc> {
    pub glb: IrGlb<'jcc>,
    pub offset: usize,
}

pub enum IrVarValueTy<'jcc> {
    Zero,
    Addr(IrVarValueAddr<'jcc>),
    Int(IrIntCnst),
    List(Vec<IrVarValueListEl<'jcc>>),
}

pub struct IrVarValue<'jcc> {
    pub ty: IrVarValueTy<'jcc>,
    pub var_ty: IrVarTy<'jcc>,
}

impl<'jcc> IrVarValue<'jcc> {
    // TODO: we need unit (for `var_ty_bytes` and in turn `ir_var_ty_mk_array`) but kinda ugly
    pub fn from_bytes(unit: IrUnit<'jcc>, offset: usize, bytes: &[u8]) -> Self {
        let lst = bytes
            .iter()
            .enumerate()
            .map(|(i, &b)| {
                let offset = offset + i;
                let value = IrVarValue {
                    ty: IrVarValueTy::Int(b.into()),
                    var_ty: unsafe { IrVarTy::from_raw(IR_VAR_TY_I8) },
                };

                IrVarValueListEl { offset, value }
            })
            .collect::<Vec<_>>();

        let var_ty = unit.var_ty_bytes(bytes.len());

        IrVarValue {
            ty: IrVarValueTy::List(lst),
            var_ty,
        }
    }
}

pub fn mk_value(
    unit: IrUnit,
    IrVarValue {
        ty,
        var_ty: IrVarTy(var_ty, ..),
    }: &IrVarValue,
    base: usize,
) -> ir_var_value {
    let arena = unit.mk_arena();

    match ty {
        IrVarValueTy::Zero => ir_var_value {
            ty: IR_VAR_VALUE_TY_ZERO,
            var_ty: *var_ty,
            _1: ir_var_value__bindgen_ty_1 {
                // dummy field to satisfy rust
                int_value: 0,
            },
        },
        IrVarValueTy::Int(IrIntCnst { val }) => ir_var_value {
            ty: IR_VAR_VALUE_TY_INT,
            var_ty: *var_ty,
            _1: ir_var_value__bindgen_ty_1 { int_value: *val },
        },
        IrVarValueTy::Addr(IrVarValueAddr { glb, offset }) => ir_var_value {
            ty: IR_VAR_VALUE_TY_ADDR,
            var_ty: *var_ty,
            _1: ir_var_value__bindgen_ty_1 {
                // dummy field to satisfy rust
                addr: ir_var_addr {
                    glb: glb.as_mut_ptr(),
                    offset: *offset as _,
                },
            },
        },
        IrVarValueTy::List(ir_var_values) => {
            let num_values = ir_var_values.len();

            let values = arena.alloc_slice::<ir_var_value>(num_values);
            let offsets = arena.alloc_slice::<usize>(num_values);
            for (i, IrVarValueListEl { offset, value }) in ir_var_values.iter().enumerate() {
                unsafe {
                    let offset = base + offset;
                    let value = mk_value(unit, value, offset);

                    values.add(i).write(value);
                    offsets.add(i).write(offset)
                }
            }

            ir_var_value {
                ty: IR_VAR_VALUE_TY_VALUE_LIST,
                var_ty: *var_ty,
                _1: ir_var_value__bindgen_ty_1 {
                    value_list: ir_var_value_list {
                        values,
                        offsets,
                        num_values,
                    },
                },
            }
        }
    }
}

impl IrVar<'_> {
    pub fn unit(&self) -> IrUnit {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrUnit::from_non_null(unsafe { NonNull::new_unchecked(p.unit) })
    }

    pub fn mk_int(&self, IrVarTy(var_ty, ..): IrVarTy, value: IrIntCnst) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        p.value = ir_var_value {
            ty: IR_VAR_VALUE_TY_INT,
            var_ty,
            _1: ir_var_value__bindgen_ty_1 {
                int_value: value.val,
            },
        };
    }

    pub fn mk_value(&self, value: &IrVarValue) {
        let unit = self.unit();
        let arena = unit.mk_arena();

        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        p.value = mk_value(unit, value, 0);
    }
}

impl<'jcc> IrFunc<'jcc> {
    pub fn arena(&self) -> ArenaAllocRef {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        ArenaAllocRef::from_raw(p.arena)
    }

    pub fn unit(&self) -> IrUnit<'jcc> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrUnit::from_non_null(unsafe { NonNull::new_unchecked(p.unit) })
    }

    pub fn ret_var_ty(&self) -> IrVarTy<'jcc> {
        unsafe {
            let p = self.0.as_ptr().as_mut_unchecked();
            IrVarTy::from_raw(*p.func_ty.ret_ty)
        }
    }

    pub fn mk_param_stmt(&'jcc self) -> IrStmt<'jcc> {
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

    pub fn basicblocks(&self) -> IterFunc<IrBasicBlock<'jcc>> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IterFunc {
            cur: self.first(),
            sz: p.basicblock_count,
        }
    }

    pub fn first(&self) -> Option<IrBasicBlock<'jcc>> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrBasicBlock::new(p.first)
    }

    pub fn last(&self) -> Option<IrBasicBlock<'jcc>> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrBasicBlock::new(p.last)
    }

    pub fn get_param_op(&self, idx: usize) -> IrOp<'jcc> {
        // FIXME: more efficient way to do this (JCC should probably have better UX here, probably a `ir_get_param_stmt` fn
        let stmt = self
            .first()
            .and_then(|b| b.first())
            .expect("get_param_op but no bb present");

        debug_assert!(stmt.is_params(), "no params stmt!");
        let op = stmt.ops().nth(idx).expect("could not get param op");

        unsafe {
            debug_assert!(
                ((*op.as_mut_ptr()).flags & IR_OP_FLAG_PARAM) != 0,
                "expected param op"
            );
        }

        op
    }

    pub fn alloc_basicblock(&self) -> IrBasicBlock<'jcc> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        unsafe { IrBasicBlock::from_raw(ir_alloc_basicblock(p)) }
    }

    pub fn add_local(&self, IrVarTy(var_ty, ..): IrVarTy) -> IrLcl<'jcc> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrLcl::from_raw(unsafe { ir_add_local(p, &var_ty) })
    }

    pub fn add_param_local(&self, IrVarTy(var_ty, ..): IrVarTy) -> IrLcl<'jcc> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        let lcl = unsafe { ir_add_local(p, &var_ty) };
        (unsafe { *lcl }).flags |= IR_LCL_FLAG_PARAM;
        IrLcl::from_raw(lcl)
    }
}

pub enum IrBasicBlockTy<'a, 'jcc> {
    Ret,
    Merge {
        target: IrBasicBlock<'jcc>,
    },
    Split {
        true_target: IrBasicBlock<'jcc>,
        false_target: IrBasicBlock<'jcc>,
    },
    Switch {
        default_target: IrBasicBlock<'jcc>,
        cases: &'a [(u128, IrBasicBlock<'jcc>)],
    },
}

impl<'jcc> IrBasicBlock<'jcc> {
    pub fn mk_ty(&self, ty: IrBasicBlockTy) {
        let func = self.func();
        let arena = func.unit().mk_arena();
        let f = func.as_mut_ptr();
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
                IrBasicBlockTy::Switch {
                    default_target,
                    cases,
                } => {
                    let num_cases = cases.len();
                    let split_cases = arena.alloc_slice::<ir_split_case>(num_cases);

                    for (i, &(value, target)) in cases.iter().enumerate() {
                        split_cases.add(i).write(ir_split_case {
                            value: value.try_into().expect("u128"),
                            target: target.as_mut_ptr(),
                        })
                    }

                    ir_make_basicblock_switch(
                        f,
                        p,
                        num_cases,
                        split_cases,
                        default_target.as_mut_ptr(),
                    )
                }
            }
        }
    }

    pub fn func(&self) -> IrFunc<'jcc> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrFunc::from_raw(p.func)
    }

    pub fn first(&self) -> Option<IrStmt<'jcc>> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrStmt::new(p.first)
    }

    pub fn last(&self) -> Option<IrStmt<'jcc>> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrStmt::new(p.last)
    }

    pub fn stmts(&self) -> IterFunc<IrStmt<'jcc>> {
        let p = unsafe { self.func().as_mut_ptr().as_mut_unchecked() };
        IterFunc {
            cur: self.first(),
            sz: p.stmt_count,
        }
    }

    pub fn alloc_stmt(&self) -> IrStmt<'jcc> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        unsafe {
            // FIXME: inefficient
            let fun = p.func;
            IrStmt::from_raw(ir_alloc_stmt(fun, p))
        }
    }
}

impl<'jcc> IrStmt<'jcc> {
    pub fn basicblock(&self) -> IrBasicBlock<'jcc> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrBasicBlock::from_raw(p.basicblock)
    }

    pub fn func(&self) -> IrFunc<'jcc> {
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

    pub fn first(&self) -> Option<IrOp<'jcc>> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrOp::new(p.first)
    }

    pub fn last(&self) -> Option<IrOp<'jcc>> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrOp::new(p.last)
    }

    pub fn ops(&self) -> IterFunc<IrOp<'jcc>> {
        let p = unsafe { self.func().as_mut_ptr().as_mut_unchecked() };
        IterFunc {
            cur: self.first(),
            sz: p.op_count,
        }
    }

    pub fn alloc_op(&self) -> IrOp<'jcc> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        unsafe {
            // FIXME: inefficient
            let fun = (*p.basicblock).func;
            IrOp::from_raw(ir_alloc_op(fun, p))
        }
    }
}

pub struct AddrOffset<'jcc> {
    base: IrOp<'jcc>,
    index: Option<IrOp<'jcc>>,
    scale: usize,
    offset: usize,
}

impl<'jcc> AddrOffset<'jcc> {
    pub fn offset(base: IrOp<'jcc>, offset: usize) -> Self {
        Self {
            base,
            offset,
            index: None,
            scale: 0,
        }
    }

    pub fn index(base: IrOp<'jcc>, index: IrOp<'jcc>, scale: NonZeroUsize) -> Self {
        Self {
            base,
            index: Some(index),
            scale: scale.get(),
            offset: 0,
        }
    }

    pub fn index_offset(
        base: IrOp<'jcc>,
        index: IrOp<'jcc>,
        scale: NonZeroUsize,
        offset: usize,
    ) -> Self {
        Self {
            base,
            index: Some(index),
            scale: scale.into(),
            offset,
        }
    }
}

pub struct IrIntCnst {
    pub val: u64,
}

impl<T: Into<u64>> From<T> for IrIntCnst {
    fn from(value: T) -> Self {
        Self { val: value.into() }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IrCnst<'jcc> {
    pub var_ty: IrVarTy<'jcc>,
    pub cnst: IrCnstTy,
}

#[derive(Debug, Clone, Copy)]
pub enum IrCnstTy {
    Int(u64),
    Float(f64),
}

#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum IrUnOpTy {
    Fneg = IR_OP_UNARY_OP_TY_FNEG,
    Fsqrt = IR_OP_UNARY_OP_TY_FSQRT,
    Fabs = IR_OP_UNARY_OP_TY_FABS,
    Neg = IR_OP_UNARY_OP_TY_NEG,
    LogNot = IR_OP_UNARY_OP_TY_LOGICAL_NOT,
    Not = IR_OP_UNARY_OP_TY_NOT,
}

#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum IrBinOpTy {
    Eq = IR_OP_BINARY_OP_TY_EQ,
    Neq = IR_OP_BINARY_OP_TY_NEQ,
    Ugt = IR_OP_BINARY_OP_TY_UGT,
    Sgt = IR_OP_BINARY_OP_TY_SGT,
    Ugteq = IR_OP_BINARY_OP_TY_UGTEQ,
    Sgteq = IR_OP_BINARY_OP_TY_SGTEQ,
    Ult = IR_OP_BINARY_OP_TY_ULT,
    Slt = IR_OP_BINARY_OP_TY_SLT,
    Ulteq = IR_OP_BINARY_OP_TY_ULTEQ,
    Slteq = IR_OP_BINARY_OP_TY_SLTEQ,
    Fmax = IR_OP_BINARY_OP_TY_FMAX,
    Fmin = IR_OP_BINARY_OP_TY_FMIN,
    Feq = IR_OP_BINARY_OP_TY_FEQ,
    Fneq = IR_OP_BINARY_OP_TY_FNEQ,
    Fgt = IR_OP_BINARY_OP_TY_FGT,
    Fgteq = IR_OP_BINARY_OP_TY_FGTEQ,
    Flt = IR_OP_BINARY_OP_TY_FLT,
    Flteq = IR_OP_BINARY_OP_TY_FLTEQ,
    Lshift = IR_OP_BINARY_OP_TY_LSHIFT,
    Srshift = IR_OP_BINARY_OP_TY_SRSHIFT,
    Urshift = IR_OP_BINARY_OP_TY_URSHIFT,
    And = IR_OP_BINARY_OP_TY_AND,
    Or = IR_OP_BINARY_OP_TY_OR,
    Xor = IR_OP_BINARY_OP_TY_XOR,
    Add = IR_OP_BINARY_OP_TY_ADD,
    Sub = IR_OP_BINARY_OP_TY_SUB,
    Mul = IR_OP_BINARY_OP_TY_MUL,
    Sdiv = IR_OP_BINARY_OP_TY_SDIV,
    Udiv = IR_OP_BINARY_OP_TY_UDIV,
    Smod = IR_OP_BINARY_OP_TY_SMOD,
    Umod = IR_OP_BINARY_OP_TY_UMOD,
    Fadd = IR_OP_BINARY_OP_TY_FADD,
    Fsub = IR_OP_BINARY_OP_TY_FSUB,
    Fmul = IR_OP_BINARY_OP_TY_FMUL,
    Fdiv = IR_OP_BINARY_OP_TY_FDIV,
}

impl<'jcc> IrOp<'jcc> {
    pub fn var_ty(&self) -> IrVarTy<'jcc> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrVarTy::from_raw(p.var_ty)
    }

    // hacky should have general way to do this
    pub fn get_addr_lcl(&self) -> Option<IrLcl<'jcc>> {
        unsafe {
            let p = self.0.as_ptr().as_mut_unchecked();
            if p.ty == IR_OP_TY_ADDR && p._1.addr.ty == IR_OP_ADDR_TY_LCL {
                return Some(IrLcl::from_raw(p._1.addr._1.lcl));
            }
        }

        None
    }

    pub fn stmt(&self) -> IrStmt<'jcc> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        IrStmt::from_raw(p.stmt)
    }

    pub fn basicblock(&self) -> IrBasicBlock<'jcc> {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };
        unsafe { IrBasicBlock::from_raw((*p.stmt).basicblock) }
    }

    pub fn func(&self) -> IrFunc<'jcc> {
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

    pub fn mk_br_cond(&self, cond: IrOp) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        unsafe {
            p.ty = IR_OP_TY_BR_COND;
            p.var_ty = IR_VAR_TY_NONE;
            p._1.br_cond = ir_op_br_cond {
                cond: cond.as_mut_ptr(),
            };
        }
    }

    pub fn mk_br_switch(&self, value: IrOp) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        unsafe {
            p.ty = IR_OP_TY_BR_SWITCH;
            p.var_ty = IR_VAR_TY_NONE;
            p._1.br_switch = ir_op_br_switch {
                value: value.as_mut_ptr(),
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

    pub fn mk_mov_param(&self, IrVarTy(var_ty, ..): IrVarTy) {
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

    pub fn mk_mov(&self, IrVarTy(var_ty, ..): IrVarTy, op: IrOp) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        p.ty = IR_OP_TY_MOV;
        p.var_ty = var_ty;
        p._1.mov = ir_op_mov {
            value: op.as_mut_ptr(),
        };
    }

    pub fn mk_undf(&self, IrVarTy(var_ty, ..): IrVarTy) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        p.ty = IR_OP_TY_UNDF;
        p.var_ty = var_ty;
    }

    pub fn mk_cnst(&self, IrCnst { var_ty, cnst }: IrCnst) {
        match cnst {
            IrCnstTy::Int(val) => self.mk_cnst_int(var_ty, val),
            IrCnstTy::Float(val) => self.mk_cnst_float(var_ty, val),
        }
    }

    pub fn mk_cnst_int(&self, IrVarTy(var_ty, ..): IrVarTy, value: u64) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        p.ty = IR_OP_TY_CNST;
        p.var_ty = var_ty;
        p._1.cnst = ir_op_cnst {
            ty: IR_OP_CNST_TY_INT,
            _1: ir_op_cnst__bindgen_ty_1 { int_value: value },
        };
    }

    pub fn mk_cnst_float(&self, IrVarTy(var_ty, ..): IrVarTy, value: f64) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        p.ty = IR_OP_TY_CNST;
        p.var_ty = var_ty;
        p._1.cnst = ir_op_cnst {
            ty: IR_OP_CNST_TY_FLT,
            _1: ir_op_cnst__bindgen_ty_1 { flt_value: value },
        };
    }

    pub fn mk_load_addr(&self, IrVarTy(var_ty, ..): IrVarTy, addr: IrOp) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        p.ty = IR_OP_TY_LOAD;
        p.var_ty = var_ty;
        p._1.load = ir_op_load {
            ty: IR_OP_LOAD_TY_ADDR,
            _1: ir_op_load__bindgen_ty_1 {
                addr: addr.0.as_ptr(),
            },
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

    pub fn mk_memcpy(&self, src: IrOp, dst: IrOp, length: usize) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        unsafe {
            p.ty = IR_OP_TY_MEM_COPY;
            p.var_ty = IR_VAR_TY_NONE;
            p._1.mem_copy = ir_op_mem_copy {
                source: src.as_mut_ptr(),
                dest: dst.as_mut_ptr(),
                length,
            };
        }
    }

    pub fn mk_select(
        &self,
        IrVarTy(var_ty, ..): IrVarTy,
        cond: IrOp,
        true_op: IrOp,
        false_op: IrOp,
    ) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        p.ty = IR_OP_TY_SELECT;
        p.var_ty = var_ty;
        p._1.select = ir_op_select {
            cond: cond.as_mut_ptr(),
            true_op: true_op.as_mut_ptr(),
            false_op: false_op.as_mut_ptr(),
        };
    }

    pub fn mk_memset(&self, addr: IrOp, value: u8, length: usize) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        unsafe {
            p.ty = IR_OP_TY_MEM_SET;
            p.var_ty = IR_VAR_TY_NONE;
            p._1.mem_set = ir_op_mem_set {
                addr: addr.as_mut_ptr(),
                value,
                length,
            };
        }
    }

    pub fn mk_unnop(&self, ty: IrUnOpTy, IrVarTy(var_ty, ..): IrVarTy, value: IrOp) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        p.ty = IR_OP_TY_UNARY_OP;
        p.var_ty = var_ty;
        p._1.unary_op = ir_op_unary_op {
            ty: ty as u32,
            value: value.as_mut_ptr(),
        };
    }

    pub fn mk_binop(&self, ty: IrBinOpTy, IrVarTy(var_ty, ..): IrVarTy, lhs: IrOp, rhs: IrOp) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        p.ty = IR_OP_TY_BINARY_OP;
        p.var_ty = var_ty;
        p._1.binary_op = ir_op_binary_op {
            ty: ty as u32,
            lhs: lhs.as_mut_ptr(),
            rhs: rhs.as_mut_ptr(),
        };
    }

    fn mk_cast<const CAST_TY: u32>(&self, IrVarTy(var_ty, ..): IrVarTy, value: IrOp) {
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
        let IR_OP_TY_CNST = self.chk_ty() else {
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

    pub fn mk_addr_offset(&self, mut addr_offset: AddrOffset) {
        let p = unsafe { self.0.as_ptr().as_mut_unchecked() };

        // TODO: jcc should do this (cnst * 1 -> offset)
        if let AddrOffset {
            base,
            index: Some(index),
            scale: 1,
            offset,
        } = addr_offset
            && let Some(val) = index.get_int_cnst()
        {
            addr_offset = AddrOffset::offset(base, val.val as usize + offset)
        };

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
                offset: addr_offset.offset,
            };
        }
    }
}

#[repr(u32)]
pub enum IrLinkage {
    Internal = IR_LINKAGE_INTERNAL,
    External = IR_LINKAGE_EXTERNAL,
}

impl<'jcc> IrUnit<'jcc> {
    pub fn add_global_def_var(
        &self,
        IrVarTy(var_ty, ..): IrVarTy,
        name: Option<&str>,
        linkage: IrLinkage,
    ) -> IrGlb<'jcc> {
        unsafe {
            let arena = self.mk_arena();
            let name = name.map(|n| arena.alloc_str(n)).unwrap_or(ptr::null());

            let unit = self.ptr.as_ptr().as_mut_unchecked();

            let glb = ir_add_global(unit, IR_GLB_TY_DATA, &var_ty, IR_GLB_DEF_TY_DEFINED, name);

            // TODO: linkage
            (*glb).linkage = linkage as _;
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

            IrGlb::from_raw(glb)
        }
    }

    pub fn add_global_undef_func(
        &self,
        IrVarTy(var_ty, ..): IrVarTy,
        name: &str,
        linkage: IrLinkage,
    ) -> IrGlb<'jcc> {
        unsafe {
            let arena = self.mk_arena();
            let name = arena.alloc_str(name);

            let unit = self.ptr.as_ptr().as_mut_unchecked();

            debug_assert!(var_ty.ty == IR_VAR_TY_TY_FUNC, "expected func ty");

            let glb = ir_add_global(unit, IR_GLB_TY_FUNC, &var_ty, IR_GLB_DEF_TY_UNDEFINED, name);

            // TODO: linkage
            (*glb).linkage = linkage as _;
            (*glb)._1.func = ptr::null_mut();

            IrGlb::from_raw(glb)
        }
    }

    pub fn add_global_def_func(
        &self,
        IrVarTy(var_ty, ..): IrVarTy,
        name: &str,
        linkage: IrLinkage,
    ) -> IrGlb<'jcc> {
        unsafe {
            let arena = self.mk_arena();
            let name = arena.alloc_str(name);

            let unit = self.ptr.as_ptr().as_mut_unchecked();

            debug_assert!(var_ty.ty == IR_VAR_TY_TY_FUNC, "expected func ty");

            let glb = ir_add_global(unit, IR_GLB_TY_FUNC, &var_ty, IR_GLB_DEF_TY_DEFINED, name);
            let glb = IrGlb::from_raw(glb);

            glb.mk_def();

            glb
        }
    }
}

// TODO: expensive, take ref in more places
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct IrVarTy<'jcc>(ir_var_ty, PhantomData<&'jcc ir_var_ty>);

pub enum IrVarTyAggregateTy {
    Struct,
    Union,
}

pub struct IrVarTyFun<'a, 'jcc> {
    pub params: &'a [IrVarTy<'jcc>],
    pub ret: IrVarTy<'jcc>,
}

pub struct IrVarTyAggregate<'a, 'jcc> {
    pub ty: IrVarTyAggregateTy,
    pub fields: &'a [IrVarTy<'jcc>],
}

pub struct IrVarTyInfo {
    pub size: usize,
    pub alignment: usize,
}

#[repr(u32)]
pub enum IrVarTyTy {
    None = IR_VAR_TY_TY_NONE,
    Pointer = IR_VAR_TY_TY_POINTER,
    Primitive = IR_VAR_TY_TY_PRIMITIVE,
    Func = IR_VAR_TY_TY_FUNC,
    Struct = IR_VAR_TY_TY_STRUCT,
    Union = IR_VAR_TY_TY_UNION,
}

#[repr(u32)]
pub enum IrVarTyPrimitive {
    I1 = IR_VAR_PRIMITIVE_TY_I1,
    I8 = IR_VAR_PRIMITIVE_TY_I8,
    I16 = IR_VAR_PRIMITIVE_TY_I16,
    I32 = IR_VAR_PRIMITIVE_TY_I32,
    I64 = IR_VAR_PRIMITIVE_TY_I64,
    I128 = IR_VAR_PRIMITIVE_TY_I128,
    F16 = IR_VAR_PRIMITIVE_TY_F16,
    F32 = IR_VAR_PRIMITIVE_TY_F32,
    F64 = IR_VAR_PRIMITIVE_TY_F64,
}

impl IrVarTyPrimitive {
    pub fn bit_width(&self) -> usize {
        match self {
            IrVarTyPrimitive::I1 => 1,
            IrVarTyPrimitive::I8 => 8,
            IrVarTyPrimitive::I16 => 16,
            IrVarTyPrimitive::I32 => 32,
            IrVarTyPrimitive::I64 => 64,
            IrVarTyPrimitive::I128 => 128,
            IrVarTyPrimitive::F16 => 16,
            IrVarTyPrimitive::F32 => 32,
            IrVarTyPrimitive::F64 => 64,
        }
    }
}

impl<'jcc> IrVarTy<'jcc> {
    fn from_raw(var_ty: ir_var_ty) -> Self {
        Self(var_ty, PhantomData)
    }

    pub fn ty(&self) -> IrVarTyTy {
        match self.0.ty {
            IR_VAR_TY_TY_NONE => IrVarTyTy::None,
            IR_VAR_TY_TY_POINTER => IrVarTyTy::Pointer,
            IR_VAR_TY_TY_PRIMITIVE => IrVarTyTy::Primitive,
            IR_VAR_TY_TY_FUNC => IrVarTyTy::Func,
            IR_VAR_TY_TY_STRUCT => IrVarTyTy::Struct,
            IR_VAR_TY_TY_UNION => IrVarTyTy::Union,
            _ => panic!("bad ty"),
        }
    }

    pub fn ty_none() -> IrVarTy<'static> {
        IrVarTy::from_raw(unsafe { IR_VAR_TY_NONE })
    }
    pub fn ty_pointer() -> IrVarTy<'static> {
        IrVarTy::from_raw(unsafe { IR_VAR_TY_POINTER })
    }

    pub fn ty_i1() -> IrVarTy<'static> {
        IrVarTy::from_raw(unsafe { IR_VAR_TY_I1 })
    }
    pub fn ty_i8() -> IrVarTy<'static> {
        IrVarTy::from_raw(unsafe { IR_VAR_TY_I8 })
    }
    pub fn ty_i16() -> IrVarTy<'static> {
        IrVarTy::from_raw(unsafe { IR_VAR_TY_I16 })
    }
    pub fn ty_i32() -> IrVarTy<'static> {
        IrVarTy::from_raw(unsafe { IR_VAR_TY_I32 })
    }
    pub fn ty_i64() -> IrVarTy<'static> {
        IrVarTy::from_raw(unsafe { IR_VAR_TY_I64 })
    }
    pub fn ty_i128() -> IrVarTy<'static> {
        IrVarTy::from_raw(unsafe { IR_VAR_TY_I128 })
    }

    pub fn ty_f16() -> IrVarTy<'static> {
        IrVarTy::from_raw(unsafe { IR_VAR_TY_F16 })
    }
    pub fn ty_f32() -> IrVarTy<'static> {
        IrVarTy::from_raw(unsafe { IR_VAR_TY_F32 })
    }
    pub fn ty_f64() -> IrVarTy<'static> {
        IrVarTy::from_raw(unsafe { IR_VAR_TY_F64 })
    }
    // const F128: Self = Self(unsafe { IR_VAR_TY_F128 });

    pub fn fun(&self) -> Option<IrVarTyFun> {
        let IR_VAR_TY_TY_FUNC = self.0.ty else {
            return None;
        };

        unsafe {
            // SAFETY: repr(transparent)
            let params = slice::from_raw_parts(
                self.0._1.func.params.cast::<IrVarTy>(),
                self.0._1.func.num_params,
            );

            let ret = *self.0._1.func.ret_ty.cast::<IrVarTy>();

            Some(IrVarTyFun { params, ret })
        }
    }

    pub fn aggregate(&self) -> Option<IrVarTyAggregate> {
        let ty = match self.0.ty {
            IR_VAR_TY_TY_STRUCT => IrVarTyAggregateTy::Struct,
            IR_VAR_TY_TY_UNION => IrVarTyAggregateTy::Union,
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

    pub fn is_pointer(&self) -> bool {
        matches!(self.0.ty, IR_VAR_TY_TY_POINTER)
    }

    pub fn is_aggregate(&self) -> bool {
        matches!(self.0.ty, IR_VAR_TY_TY_STRUCT | IR_VAR_TY_TY_UNION)
    }

    pub fn int_cmp(&self, other: &Self) -> cmp::Ordering {
        debug_assert!(self.is_int() && other.is_int());

        unsafe { self.0._1.primitive.cmp(&other.0._1.primitive) }
    }

    pub fn is_primitive(&self) -> bool {
        self.primitive().is_some()
    }

    pub fn primitive(&self) -> Option<IrVarTyPrimitive> {
        let IR_VAR_TY_TY_PRIMITIVE = self.0.ty else {
            return None;
        };

        // SAFETY: #[repr(u32)]
        Some(unsafe { mem::transmute(self.0._1.primitive) })
    }

    pub fn is_i1(&self) -> bool {
        matches!(self.primitive(), Some(IrVarTyPrimitive::I1))
    }

    pub fn is_i8(&self) -> bool {
        matches!(self.primitive(), Some(IrVarTyPrimitive::I8))
    }

    pub fn is_f16(&self) -> bool {
        matches!(self.primitive(), Some(IrVarTyPrimitive::F16))
    }

    pub fn is_f32(&self) -> bool {
        matches!(self.primitive(), Some(IrVarTyPrimitive::F32))
    }

    pub fn is_f64(&self) -> bool {
        matches!(self.primitive(), Some(IrVarTyPrimitive::F64))
    }

    pub fn is_int(&self) -> bool {
        matches!(
            self.primitive(),
            Some(
                IrVarTyPrimitive::I1
                    | IrVarTyPrimitive::I8
                    | IrVarTyPrimitive::I16
                    | IrVarTyPrimitive::I32
                    | IrVarTyPrimitive::I64
                    | IrVarTyPrimitive::I128
            )
        )
    }

    pub fn is_fun(&self) -> bool {
        matches!(self.0.ty, IR_VAR_TY_TY_FUNC)
    }

    pub fn is_array(&self) -> bool {
        matches!(self.0.ty, IR_VAR_TY_TY_ARRAY)
    }

    pub fn is_none(&self) -> bool {
        matches!(self.0.ty, IR_VAR_TY_TY_NONE)
    }
}

impl Debug for IrVarTy<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f)?;
        let stderr = FlushGuard::stderr();
        unsafe {
            debug_print_var_ty_string(stderr.ptr, &self.0);
        }

        Ok(())
    }
}

impl PartialEq for IrVarTy<'_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { ir_var_ty_eq(&self.0, &other.0) }
    }
}

impl Eq for IrVarTy<'_> {}
