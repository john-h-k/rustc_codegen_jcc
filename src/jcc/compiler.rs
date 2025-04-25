use std::{
    ffi::c_char,
    marker::PhantomData,
    mem,
    path::Path,
    ptr::{self},
};

use crate::jcc_sys::*;

use super::{
    alloc::ArenaAllocRef,
    ir::{AsIrRaw, IrUnit},
};

pub trait DefaultNativeConvert: Sized {
    type Default: NativeConvert<Self>;
}

pub trait NativeConvert<T: ?Sized> {
    type NativeTy;

    fn into_native(arena: ArenaAllocRef, value: &T) -> Self::NativeTy;
}

pub struct UStrConvert;
impl NativeConvert<str> for UStrConvert {
    type NativeTy = ustr_t;

    fn into_native(_arena: ArenaAllocRef, value: &str) -> Self::NativeTy {
        ustr_t {
            str_: value.as_ptr().cast(),
            len: value.len(),
        }
    }
}

pub struct CStrConvert;
impl NativeConvert<str> for CStrConvert {
    type NativeTy = *const c_char;

    fn into_native(arena: ArenaAllocRef, value: &str) -> Self::NativeTy {
        arena.alloc_str(value)
    }
}

impl DefaultNativeConvert for () {
    type Default = UnitConvert;
}

pub struct UnitConvert;
impl NativeConvert<()> for UnitConvert {
    type NativeTy = ();

    fn into_native(_arena: ArenaAllocRef, _value: &()) -> Self::NativeTy {}
}

pub struct HashTbl<'tbl, K, V, KConv, VConv>
where
    K: ?Sized,
    V: ?Sized,
    KConv: NativeConvert<K>,
    VConv: NativeConvert<V>,
{
    ptr: *mut hashtbl,
    arena: ArenaAllocRef,
    _phantom: PhantomData<(&'tbl (), Box<K>, Box<V>, KConv, VConv)>,
}

impl<'tbl, V, VConv: NativeConvert<V>> HashTbl<'tbl, str, V, UStrConvert, VConv> {
    // NOTE: rust 'str' == jcc 'ustr_t'

    pub fn new_ustr_keyed_in(arena: ArenaAllocRef) -> Self {
        unsafe {
            let ptr = hashtbl_create_ustr_keyed_in_arena(arena.as_ptr(), mem::size_of::<V>());

            Self {
                ptr,
                arena,
                _phantom: PhantomData,
            }
        }
    }
}

// impl<'tbl, V, VConv: NativeConvert<V>> HashTbl<'tbl, str, V, CStrConvert, VConv> {
impl<'tbl, K, V, KConv, VConv> HashTbl<'tbl, K, V, KConv, VConv>
where
    K: ?Sized,
    KConv: NativeConvert<K>,
    VConv: NativeConvert<V>,
{
    pub fn new_str_keyed_in(arena: ArenaAllocRef) -> Self {
        unsafe {
            let ptr = hashtbl_create_str_keyed_in_arena(arena.as_ptr(), mem::size_of::<V>());

            Self {
                ptr,
                arena,
                _phantom: PhantomData,
            }
        }
    }
}

use std::fmt::Debug;

impl<'tbl, K, V, KConv, VConv> HashTbl<'tbl, K, V, KConv, VConv>
where
    K: ?Sized,
    KConv: NativeConvert<K>,
    VConv: NativeConvert<V>,
    KConv::NativeTy: Debug,
    VConv::NativeTy: Debug,
{
    pub fn get(&mut self, k: &K) -> Option<&V> {
        let k = KConv::into_native(self.arena, k);

        unsafe {
            let p = hashtbl_lookup(self.ptr, ptr::from_ref(&k).cast());
            (p as *const V).as_ref()
        }
    }

    pub fn insert(&mut self, k: &K, v: &V) {
        unsafe {
            let k = KConv::into_native(self.arena, k);
            let v = VConv::into_native(self.arena, v);
            dbg!(&k);
            dbg!(&v);

            hashtbl_insert(
                self.ptr,
                ptr::from_ref(&k).cast(),
                if mem::size_of_val(&v) == 0 {
                    ptr::null()
                } else {
                    ptr::from_ref(&v).cast()
                },
            );
        }
    }

    pub fn into_raw(self) -> *mut hashtbl {
        let p = self.ptr;
        mem::forget(self);
        p
    }
}

type CStrHashTbl<'tbl, V> =
    HashTbl<'tbl, str, V, CStrConvert, <V as DefaultNativeConvert>::Default>;

pub struct Compiler(*mut compiler);

impl Compiler {
    pub fn new(arena: &impl AsRef<ArenaAllocRef>, obj_out: &Path) -> Self {
        let arena = arena.as_ref();
        let mut fs = ptr::null_mut();
        unsafe {
            fs_create(arena.as_ptr(), FS_FLAG_ASSUME_CONSTANT, &mut fs);

            // FIXME: target
            let target = &AARCH64_MACOS_TARGET;
            let path = arena.alloc_str(obj_out);

            let output = compile_file {
                ty: COMPILE_FILE_TY_PATH,
                path,
                file: ptr::null_mut(),
            };

            let log_syms = ptr::null_mut();
            // let log_syms = {
            //     let mut tbl = CStrHashTbl::new_str_keyed_in(*arena);

            //     tbl.insert("main", &());
            //     tbl.insert("_ZN3std2rt10lang_start17h4f5efa6f8ab6dfaeE", &());
            //     tbl.insert("_ZN3std2rt19lang_start_internal17h171f0f2bb6b4ee07E", &());
            //     tbl.into_raw()
            // };

            let args = compiler_ir_create_args {
                fs,
                target,
                output,
                args: compile_ir_args {
                    target: COMPILE_TARGET_MACOS_ARM64,
                    log_flags: COMPILE_LOG_FLAGS_ALL,
                    // log_flags: COMPILE_LOG_FLAGS_NONE,
                    opts_level: COMPILE_OPTS_LEVEL_0,
                    // codegen_flags: CODEGEN_FLAG_ABI_LOWERED,
                    codegen_flags: CODEGEN_FLAG_NONE,
                    log_symbols: log_syms,
                    build_asm_file: false,
                    build_object_file: true,
                    verbose: false,
                    use_graphcol_regalloc: false,
                    output,
                },
            };

            let mut compiler = ptr::null_mut();
            compiler_create_for_ir(&args, &mut compiler);

            Self(compiler)
        }
    }

    pub fn compile_ir(&mut self, ir: IrUnit) -> Result<(), ()> {
        unsafe {
            match compile_ir(self.0, ir.as_mut_ptr()) {
                COMPILE_RESULT_SUCCESS => Ok(()),
                COMPILE_RESULT_FAILURE => Err(()),
                COMPILE_RESULT_BAD_FILE => Err(()),
                _ => unreachable!("bad 'compile_ir' return val"),
            }
        }
    }
}
