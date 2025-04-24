use std::{
    alloc::{AllocError, Allocator, Layout},
    mem::{self, MaybeUninit},
    ptr::NonNull,
    slice,
};

use crate::jcc_sys::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrVarTy;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrFunc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrOp;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IrBasicBlock;

pub struct ArenaAlloc(*mut arena_allocator);

const DEFAULT_ALIGN: usize = 16;

impl ArenaAlloc {
    pub fn new() -> Self {
        let mut p = MaybeUninit::uninit();

        unsafe {
            arena_allocator_create(c"rustc_codegen_jcc".as_ptr(), p.as_mut_ptr());
            Self(p.assume_init())
        }
    }

    pub fn as_ptr(&self) -> *mut arena_allocator {
        self.0
    }

    pub fn alloc<T>(&self) -> *mut T {
        let ptr = self.alloc_raw(mem::size_of::<T>());
        NonNull::as_ptr(ptr) as _
    }

    pub fn alloc_raw(&self, size: usize) -> NonNull<[u8]> {
        unsafe {
            self.allocate(Layout::from_size_align_unchecked(size, DEFAULT_ALIGN))
                .unwrap()
        }
    }
}

impl Drop for ArenaAlloc {
    fn drop(&mut self) {
        unsafe {
            arena_allocator_free(&mut self.0);
        }
    }
}

unsafe impl Allocator for ArenaAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        assert!(layout.align() <= 16, "overly aligned!");

        unsafe {
            let p = aralloc(self.0, layout.size()) as *mut u8;
            let slice = slice::from_raw_parts_mut(p, layout.size());

            Ok(NonNull::from(slice))
        }
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}
}
