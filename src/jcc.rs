use std::{
    alloc::{AllocError, Allocator, Layout},
    ffi::{CString, c_char},
    fmt::{Debug, Formatter},
    fs::File,
    io,
    mem::{self, MaybeUninit},
    os::{
        fd::{AsFd, AsRawFd},
        raw::c_void,
    },
    ptr::{self, NonNull},
    slice,
};

pub mod alloc;
pub mod compiler;
pub mod ir;
