use std::{
    ffi::{CStr, c_char},
    io::Write,
    path::Path,
    ptr::{self},
};

use crate::jcc_sys::*;

use super::{
    alloc::ArenaAllocRef,
    hashtbl::CStrHashTbl,
    ir::{AsIrRaw, IrUnit},
};

unsafe extern "C" fn demangle(arena: *mut arena_allocator, str: *const c_char) -> *const c_char {
    unsafe {
        let arena = ArenaAllocRef::from_raw(arena);
        let str = CStr::from_ptr(str).to_string_lossy();
        let demangle = rustc_demangle::demangle(&str);

        let mut result = Vec::<u8, _>::new_in(arena);
        write!(result, "{demangle}")
            .map(|r| result.as_ptr().cast())
            .unwrap_or_default()
    }
}

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

            let log_flags = COMPILE_LOG_FLAGS_ALL;
            // let log_flags = COMPILE_LOG_FLAGS_NONE;
            let log_syms = [
                // "_ZN3std2rt10lang_start17h4f5efa6f8ab6dfaeE"
                // "_ZN4core3ptr101drop_in_place$LT$std..io..error..ErrorData$LT$alloc..boxed..Box$LT$std..io..error..Custom$GT$$GT$$GT$17h82cc1923ce8eb8b5E",
                // "_ZN4core6option15Option$LT$T$GT$6unwrap17hcac1e09190d54773E",
                // "_ZN11hello_world4main17h545698fc04f0cfe5E",
            ];

            let log_syms = {
                if log_syms.is_empty() {
                    ptr::null_mut()
                } else {
                    let mut tbl = CStrHashTbl::new_str_keyed_in(*arena);

                    for sym in log_syms {
                        tbl.insert(sym, &());
                    }

                    tbl.into_raw()
                }
            };

            let args = compiler_ir_create_args {
                fs,
                target,
                output,
                args: compile_ir_args {
                    target: COMPILE_TARGET_MACOS_ARM64,
                    log_flags,
                    opts_level: COMPILE_OPTS_LEVEL_0,
                    // codegen_flags: CODEGEN_FLAG_ABI_LOWERED,
                    codegen_flags: CODEGEN_FLAG_NONE,
                    log_symbols: log_syms,
                    demangle_sym: Some(demangle),
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
