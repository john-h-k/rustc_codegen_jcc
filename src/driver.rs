use rustc_codegen_ssa::CrateInfo;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_metadata::EncodedMetadata;
use rustc_middle::ty::{Instance, TyCtxt};
use rustc_span::def_id::DefId;

pub(crate) struct OngoingCodegen {
    pub(crate) name: String,
    pub(crate) object: Vec<u8>,
    pub(crate) metadata: EncodedMetadata,
    pub(crate) crate_info: CrateInfo,
}

pub(crate) struct CodegenCx<'tcx> {
    tcx: TyCtxt<'tcx>,
}

impl<'tcx> CodegenCx<'tcx> {
    pub(crate) fn new(tcx: TyCtxt<'tcx>) -> Self {
        Self { tcx }
    }

    pub(crate) fn codegen_fn(&mut self, inst: &Instance<'_>) {}
    pub(crate) fn codegen_static(&mut self, def_id: &DefId) {}

    pub(crate) fn emit_cgu(
        &mut self,
        name: String,
        metadata: EncodedMetadata,
        _need_metadata_module: bool,
    ) -> Box<OngoingCodegen> {
        Box::new(OngoingCodegen {
            name,
            metadata,
            object: vec![],
            crate_info: CrateInfo::new(self.tcx, "jcc".to_string()),
        })
    }
}
