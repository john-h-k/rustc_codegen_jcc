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
    path::Path,
    ptr::{self, NonNull},
    slice,
};

use bitflags::bitflags;

use crate::jcc_sys::*;

use super::{
    alloc::ArenaAllocRef,
    ir::{AsIrRaw, FromIrRaw, IrUnit},
};

pub struct Compiler(*mut compiler);

impl Compiler {
    pub fn new(arena: &impl AsRef<ArenaAllocRef>, obj_out: &Path) -> Self {
        let arena = arena.as_ref();
        let mut fs = ptr::null_mut();
        unsafe {
            fs_create(arena.as_ptr(), FS_FLAG_ASSUME_CONSTANT, &mut fs);

            let target = &AARCH64_MACOS_TARGET;
            let path = arena.alloc_str(obj_out);

            let output = compile_file {
                ty: COMPILE_FILE_TY_PATH,
                path,
                file: ptr::null_mut(),
            };

            let args = compiler_ir_create_args {
                fs,
                target,
                output,
                args: compile_ir_args {
                    target: COMPILE_TARGET_MACOS_ARM64,
                    log_flags: COMPILE_LOG_FLAGS_ALL,
                    opts_level: COMPILE_OPTS_LEVEL_0,
                    // codegen_flags: CODEGEN_FLAG_ABI_LOWERED,
                    codegen_flags: CODEGEN_FLAG_NONE,
                    log_symbols: ptr::null_mut(),
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
