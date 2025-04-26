use std::{ffi::c_char, intrinsics, marker::PhantomData, mem, ptr};

use crate::jcc_sys::*;

use super::alloc::ArenaAllocRef;

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

    #[allow(clippy::type_complexity)]
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

impl<'tbl, K, V, KConv, VConv> HashTbl<'tbl, K, V, KConv, VConv>
where
    K: ?Sized,
    V: ?Sized,
    KConv: NativeConvert<K>,
    VConv: NativeConvert<V>,
{
    pub fn len(&self) -> usize {
        unsafe { hashtbl_size(self.ptr) }
    }
}

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

impl<'tbl, K, V, KConv, VConv> HashTbl<'tbl, K, V, KConv, VConv>
where
    K: ?Sized,
    KConv: NativeConvert<K>,
    VConv: NativeConvert<V>,
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
        // this has no drop atm, but when it does, you will need to `mem::forget`
        debug_assert!(
            !intrinsics::needs_drop::<Self>(),
            "now has drop you will need to mem::forget",
        );
        self.ptr
    }
}

pub type CStrHashTbl<'tbl, V> =
    HashTbl<'tbl, str, V, CStrConvert, <V as DefaultNativeConvert>::Default>;
