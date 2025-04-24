use std::{
    alloc::{AllocError, Allocator, Layout},
    ffi::{CStr, CString, c_char},
    fmt::{Debug, Formatter},
    fs::File,
    io,
    mem::{self, MaybeUninit},
    ops::Deref,
    os::{
        fd::{AsFd, AsRawFd},
        raw::c_void,
        unix::ffi::OsStrExt,
    },
    path::Path,
    ptr::{self, NonNull},
    slice,
};

use rustc_ast::token::TokenKind::AndAnd;

use crate::jcc_sys::*;

fn chk_null_safe(bytes: &[u8]) {
    if let Some(nul_pos) = memchr::memchr(0, bytes) {
        panic!(
            "str not null safe! nul char at pos {} (len {})",
            nul_pos,
            bytes.len()
        );
    }
}

// NON OWNING!
pub struct ArenaAllocRef(*mut arena_allocator);

// OWNING!
pub struct ArenaAlloc(ArenaAllocRef);

impl ArenaAlloc {
    pub fn new(name: &[u8]) -> Self {
        chk_null_safe(name);

        let mut arena = ptr::null_mut();
        unsafe {
            arena_allocator_create(name.as_ptr().cast(), &mut arena);
        }

        Self(ArenaAllocRef::from_raw(arena))
    }
}

impl AsRef<ArenaAllocRef> for ArenaAlloc {
    fn as_ref(&self) -> &ArenaAllocRef {
        &self
    }
}

impl Deref for ArenaAlloc {
    type Target = ArenaAllocRef;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

const DEFAULT_ALIGN: usize = 16;

pub trait ArenaStrLike {
    fn as_bytes(&self) -> &[u8];
}

impl ArenaStrLike for [u8] {
    fn as_bytes(&self) -> &[u8] {
        self
    }
}

impl ArenaStrLike for str {
    fn as_bytes(&self) -> &[u8] {
        self.as_bytes()
    }
}

impl ArenaStrLike for Path {
    fn as_bytes(&self) -> &[u8] {
        // FIXME: unix only
        self.as_os_str().as_bytes()
    }
}

impl ArenaAllocRef {
    pub fn from_raw(ptr: *mut arena_allocator) -> Self {
        Self(ptr)
    }

    pub fn as_ptr(&self) -> *mut arena_allocator {
        self.0
    }

    pub fn alloc_str<StrLike: ArenaStrLike + ?Sized>(&self, str: &StrLike) -> *const c_char {
        let bytes = str.as_bytes();
        chk_null_safe(bytes);

        let sz = mem::size_of_val(bytes);
        let ptr = self.alloc_raw(sz + 1).cast::<u8>();
        let ptr = NonNull::as_ptr(ptr);

        unsafe {
            ptr::copy_nonoverlapping(bytes.as_ptr().cast::<u8>(), ptr, sz);
            ptr.add(sz).write(0);
        }

        ptr.cast::<c_char>()
    }

    pub fn alloc_slice_copy<T>(&self, slice: &[T]) -> *mut T {
        if slice.len() == 0 {
            return NonNull::dangling().as_ptr();
        }

        let sz = mem::size_of_val(slice);
        let ptr = self.alloc_raw(sz).cast::<u8>();
        let ptr = NonNull::as_ptr(ptr);

        unsafe {
            ptr::copy_nonoverlapping(slice.as_ptr().cast::<u8>(), ptr, sz);
        }

        ptr.cast::<T>()
    }

    pub fn alloc_copy<T: Copy>(&self, value: &T) -> *mut T {
        let ptr = self.alloc::<T>();

        unsafe {
            *ptr = *value;
        }

        ptr.cast::<T>()
    }

    pub fn alloc<T>(&self) -> *mut T {
        let ptr = self.alloc_raw(mem::size_of::<T>());
        NonNull::as_ptr(ptr).cast::<T>()
    }

    pub fn alloc_raw(&self, size: usize) -> NonNull<[u8]> {
        unsafe {
            self.allocate(Layout::from_size_align_unchecked(size, DEFAULT_ALIGN))
                .unwrap()
        }
    }
}

unsafe impl Allocator for ArenaAllocRef {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        debug_assert!(layout.size() != 0, "0 byte alloc");
        assert!(layout.align() <= 16, "overly aligned!");

        unsafe {
            let p = aralloc(self.0, layout.size()) as *mut u8;
            let slice = slice::from_raw_parts_mut(p, layout.size());

            Ok(NonNull::from(slice))
        }
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
}
