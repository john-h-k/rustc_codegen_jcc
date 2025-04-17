#![feature(rustc_private)]
#![feature(let_chains)]
#![feature(f16, alloc_error_hook)]
// #![warn(clippy::pedantic)]

//! rustc_codegen_jcc - a JCC-based backend for rust

extern crate rustc_abi;

extern crate rustc_codegen_ssa;
extern crate rustc_const_eval;
extern crate rustc_data_structures;
extern crate rustc_driver;
extern crate rustc_errors;
extern crate rustc_hir;
extern crate rustc_index;
extern crate rustc_metadata;
extern crate rustc_middle;
extern crate rustc_session;
extern crate rustc_span;
extern crate rustc_symbol_mangling;
extern crate rustc_target;
extern crate rustc_ty_utils;
extern crate rustc_type_ir;

mod driver;
mod jcc;

use driver::{CodegenCx, OngoingCodegen};
use rustc_codegen_ssa::{
    CodegenResults, CompiledModule, CrateInfo, ModuleKind,
    back::archive::{ArArchiveBuilder, ArchiveBuilder, ArchiveBuilderBuilder},
    traits::CodegenBackend,
};

use rustc_data_structures::fx::FxIndexMap;
use rustc_metadata::EncodedMetadata;
use rustc_middle::{
    dep_graph::{WorkProduct, WorkProductId},
    mir::mono::MonoItem,
    ty::TyCtxt,
};
use rustc_session::{
    Session,
    config::{OutputFilenames, OutputType},
};
use rustc_span::{Symbol, sym};

use std::{any::Any, fs::File};

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
                .fatal("`-Cinstrument-coverage` is LLVM specific and not supported by Cranelift");
        }
    }

    fn target_features_cfg(&self, sess: &Session) -> (Vec<Symbol>, Vec<Symbol>) {
        let feats = match (sess.target.arch.as_ref(), sess.target.os.as_ref()) {
            ("x86_64", "none") => vec![],
            ("x86_64", _) => vec![sym::fsxr, sym::sse, sym::sse2, Symbol::intern("x87")],
            ("aarch64", "none") => vec![],
            ("aarch64", "macos") => vec![sym::neon, sym::aes, sym::sha2, sym::sha3],
            ("aarch64", _) => vec![sym::neon],

            _ => vec![],
        };

        (feats.clone(), feats)
    }

    fn print_version(&self) {
        todo!("");
        // println!("JCC version: {}", cranelift_codegen::VERSION);
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

        let mut cx = CodegenCx::new(tcx);

        for cgu in cgus.codegen_units {
            for (item, _data) in cgu.items() {
                match item {
                    MonoItem::Fn(instance) => cx.codegen_fn(instance),
                    MonoItem::Static(def_id) => cx.codegen_static(def_id),
                    MonoItem::GlobalAsm(item_id) => todo!("global asm"),
                }
            }
        }

        let name = cgus.codegen_units.iter().next().unwrap().name().to_string();

        cx.emit_cgu(name, metadata, need_metadata_module)
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        outputs: &OutputFilenames,
    ) -> (CodegenResults, FxIndexMap<WorkProductId, WorkProduct>) {
        use std::io::Write;
        let codegen = *ongoing_codegen
            .downcast::<OngoingCodegen>()
            .expect("in join_codegen: ongoing_codegen is not an OngoingCodegen");

        let serialized_asm_path =
            outputs.temp_path_for_cgu(OutputType::Bitcode, &codegen.name, None);

        let mut asm_out = File::create(&serialized_asm_path)
            .expect("Could not create the temporary files necessary for building the assembly!");

        asm_out
            .write_all(&codegen.object)
            .expect("Could not save the tmp assembly file!");

        let modules = vec![CompiledModule {
            name: codegen.name,
            kind: ModuleKind::Regular,
            object: Some(serialized_asm_path),
            bytecode: None,
            dwarf_object: None,
            llvm_ir: None,
            assembly: None,
            links_from_incr_cache: Vec::new(),
        }];

        let codegen_results = CodegenResults {
            modules,
            allocator_module: None,
            metadata_module: None,
            metadata: codegen.metadata,
            crate_info: codegen.crate_info,
        };

        (codegen_results, FxIndexMap::default())
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
