#![feature(rustc_private)]
#![feature(let_chains)]
#![feature(f16, alloc_error_hook)]
#![feature(allocator_api)]
#![feature(ptr_as_ref_unchecked)]
// #![warn(clippy::pedantic)]

//! rustc_codegen_jcc - a JCC-based backend for rust

extern crate rustc_abi;

extern crate memchr;
extern crate rustc_ast;
extern crate rustc_codegen_ssa;
extern crate rustc_const_eval;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_index;
extern crate rustc_infer;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_symbol_mangling;
extern crate rustc_target;
extern crate rustc_ty_utils;
extern crate rustc_type_ir;

mod builder;
mod driver;
mod jcc;
mod jcc_sys;

use builder::Builder;
use driver::{CodegenCx, JccModule};
use jcc::{
    alloc::{ArenaAlloc, ArenaAllocRef},
    compiler::Compiler,
};
use rustc_ast::expand::{allocator::AllocatorKind, autodiff_attrs::AutoDiffItem};
use rustc_codegen_ssa::{
    CodegenResults, CompiledModule, CrateInfo, ModuleCodegen, ModuleKind,
    back::{
        archive::{ArArchiveBuilder, ArchiveBuilder, ArchiveBuilderBuilder},
        lto::{LtoModuleCodegen, SerializedModule, ThinModule},
        write::{
            CodegenContext, EmitObj, FatLtoInput, ModuleConfig, OngoingCodegen,
            TargetMachineFactoryFn,
        },
    },
    mono_item::MonoItemExt,
    traits::{
        BuilderMethods, CodegenBackend, ExtraBackendMethods, ModuleBufferMethods,
        ThinBufferMethods, WriteBackendMethods,
    },
};

use rustc_data_structures::fx::FxIndexMap;
use rustc_errors::{DiagCtxtHandle, FatalError};
use rustc_metadata::EncodedMetadata;
use rustc_middle::{
    dep_graph::{WorkProduct, WorkProductId},
    mir::mono::MonoItem,
    ty::TyCtxt,
};
use rustc_session::{
    Session,
    config::{OptLevel, OutputFilenames, OutputType},
};
use rustc_span::{Symbol, sym};

use std::{
    any::Any, ffi::CString, fs::File, io::Write, os::unix::ffi::OsStringExt, ptr, sync::Arc,
};

#[derive(Clone)]
struct JccCodegenBackend;

impl JccCodegenBackend {
    fn codegen_item<'tcx>(&self, item: MonoItem<'tcx>, tcx: TyCtxt<'tcx>) -> Result<(), ()> {
        Ok(())
    }
}

impl CodegenBackend for JccCodegenBackend {
    fn locale_resource(&self) -> &'static str {
        ""
    }

    fn init(&self, sess: &Session) {
        use rustc_session::config::{InstrumentCoverage, Lto};
        match sess.lto() {
            Lto::No | Lto::ThinLocal => {}
            Lto::Thin | Lto::Fat => sess
                .dcx()
                .warn("LTO is not supported. You may get a linker error."),
        }

        if sess.opts.cg.instrument_coverage() != InstrumentCoverage::No {
            sess.dcx()
                .fatal("`-Cinstrument-coverage` is LLVM specific and not supported by jcc");
        }
    }

    fn target_features_cfg(&self, sess: &Session) -> (Vec<Symbol>, Vec<Symbol>) {
        let feats = match (sess.target.arch.as_ref(), sess.target.os.as_ref()) {
            ("x86_64", "none") => vec![],
            ("x86_64", _) => vec![sym::fsxr, sym::sse, sym::sse2],
            ("aarch64", "none") => vec![],
            ("aarch64", "macos") => vec![sym::neon, sym::aes, sym::sha2, sym::sha3],
            ("aarch64", _) => vec![sym::neon],

            _ => vec![],
        };

        (feats.clone(), feats)
    }

    fn print_version(&self) {
        todo!("");
        // println!("jcc version: {}", ...);
    }

    fn codegen_crate(
        &self,
        tcx: TyCtxt<'_>,
        metadata: EncodedMetadata,
        need_metadata_module: bool,
    ) -> Box<dyn Any> {
        let cgus = tcx.collect_and_partition_mono_items(());

        if need_metadata_module {
            todo!("metadata module");
        }

        let target_cpu = "aarch64".to_string();

        Box::new(rustc_codegen_ssa::base::codegen_crate(
            self.clone(),
            tcx,
            target_cpu,
            metadata,
            need_metadata_module,
        ))
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        outputs: &OutputFilenames,
    ) -> (CodegenResults, FxIndexMap<WorkProductId, WorkProduct>) {
        ongoing_codegen
            .downcast::<OngoingCodegen<Self>>()
            .expect("in join_codegen: ongoing_codegen is not an OngoingCodegen")
            .join(sess)
    }
}

#[unsafe(no_mangle)]
pub extern "Rust" fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    std::alloc::set_alloc_error_hook(custom_alloc_error_hook);
    Box::new(JccCodegenBackend)
}

use std::alloc::Layout;

pub fn custom_alloc_error_hook(layout: Layout) {
    panic!("memory allocation of {} bytes failed", layout.size());
}

impl ExtraBackendMethods for JccCodegenBackend {
    fn codegen_allocator<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        module_name: &str,
        kind: AllocatorKind,
        alloc_error_handler_kind: AllocatorKind,
    ) -> Self::Module {
        todo!()
    }

    fn compile_codegen_unit(
        &self,
        tcx: TyCtxt<'_>,
        cgu_name: Symbol,
    ) -> (ModuleCodegen<Self::Module>, u64) {
        let cx = CodegenCx::new(tcx);
        let builder = Builder::with_cx(&cx);

        let cgu = tcx.codegen_unit(cgu_name);

        for (item, data) in cgu.items() {
            item.predefine::<Builder<'_, '_>>(&builder, data.linkage, data.visibility);
            item.define::<Builder<'_, '_>>(&builder);
        }

        let module = builder.module();

        let cost = 100;

        let module = ModuleCodegen::new_regular(cgu_name.to_string(), module);

        (module, cost)
    }

    fn target_machine_factory(
        &self,
        sess: &Session,
        opt_level: OptLevel,
        target_features: &[String],
    ) -> TargetMachineFactoryFn<Self> {
        Arc::new(|_| Ok(()))
    }
}

struct JccModuleBuffer;
impl ModuleBufferMethods for JccModuleBuffer {
    fn data(&self) -> &[u8] {
        todo!()
    }
}

struct JccThinBuffer;
impl ThinBufferMethods for JccThinBuffer {
    fn data(&self) -> &[u8] {
        todo!()
    }

    fn thin_link_data(&self) -> &[u8] {
        todo!()
    }
}

impl WriteBackendMethods for JccCodegenBackend {
    type Module = JccModule;

    type TargetMachine = ();

    type TargetMachineError = ();

    type ModuleBuffer = JccModuleBuffer;

    type ThinData = ();

    type ThinBuffer = JccThinBuffer;

    fn run_link(
        cgcx: &CodegenContext<Self>,
        dcx: DiagCtxtHandle<'_>,
        modules: Vec<ModuleCodegen<Self::Module>>,
    ) -> Result<ModuleCodegen<Self::Module>, FatalError> {
        todo!()
    }

    fn run_fat_lto(
        cgcx: &CodegenContext<Self>,
        modules: Vec<FatLtoInput<Self>>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> Result<LtoModuleCodegen<Self>, FatalError> {
        todo!()
    }

    fn run_thin_lto(
        cgcx: &CodegenContext<Self>,
        modules: Vec<(String, Self::ThinBuffer)>,
        cached_modules: Vec<(SerializedModule<Self::ModuleBuffer>, WorkProduct)>,
    ) -> Result<(Vec<LtoModuleCodegen<Self>>, Vec<WorkProduct>), FatalError> {
        todo!()
    }

    fn print_pass_timings(&self) {
        todo!()
    }

    fn print_statistics(&self) {
        todo!()
    }

    unsafe fn optimize(
        cgcx: &CodegenContext<Self>,
        dcx: DiagCtxtHandle<'_>,
        module: &mut ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> Result<(), FatalError> {
        // TODO: run opts
        Ok(())
    }

    fn optimize_fat(
        cgcx: &CodegenContext<Self>,
        llmod: &mut ModuleCodegen<Self::Module>,
    ) -> Result<(), FatalError> {
        todo!()
    }

    unsafe fn optimize_thin(
        cgcx: &CodegenContext<Self>,
        thin: ThinModule<Self>,
    ) -> Result<ModuleCodegen<Self::Module>, rustc_errors::FatalError> {
        todo!()
    }

    unsafe fn codegen(
        cgcx: &CodegenContext<Self>,
        dcx: DiagCtxtHandle<'_>,
        module: ModuleCodegen<Self::Module>,
        config: &ModuleConfig,
    ) -> Result<CompiledModule, FatalError> {
        let _timer = cgcx
            .prof
            .generic_activity_with_arg("JCC_module_codegen", &*module.name);

        {
            let jcc_mod = &module.module_llvm;
            let module_name = module.name.clone();
            let module_name = Some(&module_name[..]);

            let obj_out = cgcx.output_filenames.temp_path_for_cgu(
                OutputType::Object,
                &module.name,
                cgcx.invocation_temp.as_deref(),
            );

            match config.emit_obj {
                EmitObj::None => {}
                EmitObj::Bitcode => todo!(),
                EmitObj::ObjectCode(_bitcode_section) => {
                    // FIXME: should use shared arena
                    let arena = ArenaAlloc::new(b"compiler");
                    let mut compiler = Compiler::new(&arena, &obj_out);
                    compiler
                        .compile_ir(jcc_mod.unit)
                        .expect("ir compilation failed");
                }
            }

            Ok(module.into_compiled_module(
                config.emit_obj != EmitObj::None,
                false,
                config.emit_bc,
                config.emit_asm,
                config.emit_ir,
                &cgcx.output_filenames,
                cgcx.invocation_temp.as_deref(),
            ))
        }
    }

    fn prepare_thin(
        module: ModuleCodegen<Self::Module>,
        want_summary: bool,
    ) -> (String, Self::ThinBuffer) {
        todo!()
    }

    fn serialize_module(module: ModuleCodegen<Self::Module>) -> (String, Self::ModuleBuffer) {
        todo!()
    }

    fn autodiff(
        cgcx: &CodegenContext<Self>,
        module: &ModuleCodegen<Self::Module>,
        diff_fncs: Vec<AutoDiffItem>,
        config: &ModuleConfig,
    ) -> Result<(), FatalError> {
        todo!()
    }
}
