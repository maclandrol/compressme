"""Public target registry. Listing a surveyed target does not imply an adapter.

These are dated evidence records, not automatic download or trust decisions.
"""
import copy
import json
from pathlib import Path


_TARGETS = {
    "nesso": {
        "name": "Nesso-1", "aliases": ["nesso1", "nesso-1"],
        "audited_on": "2026-09-08",
        "status": "pretrained_scoped_runtime_byte_verified_cpu_mps", "adapter_available": True,
        "repository": "https://github.com/recursionpharma/nesso",
        "paper": "https://doi.org/10.64898/2026.08.01.742196",
        "source_revision": "6c72f66720d9d3447fd73c515cda963e39128b1f",
        "hf_id": "recursionpharma/nesso", "hf_revision": "499ed12b0343918ab01b2519226390cf8eca038a",
        "checkpoint": "v1.0.0/model.safetensors", "checkpoint_bytes": 165426752,
        "parameters_before": 41223928, "parameters_after": 41223928,
        "lossless_packed_checkpoint_bytes": 140754378,
        "contract": "Original frozen native forward and predict_step; all tensor outputs and metadata, FP32 and original RNG calls retained",
        "macos": "CPU layout optimisation gives modest measured gains; original Nesso runs on MPS but tested wrappers offer no dependable speedup",
        "blockers": ["CUDA unavailable for validation", "No labelled biological accuracy benchmark", "ESM-2 preprocessing remains a separate 650M-parameter model"],
        "runtime": "compressme.nesso_inference_optimizations",
        "validation_scope": "Two prepared inputs: 20 or 130 protein residues plus ligand (23/143 tokens), genuine ESM-2 features, five recycles and native cropping. Complete 11 forward and 21 predict_step tensor leaves compared bytewise on each tested backend; source and input hashes recorded.",
        "packing_note": "Checkpoint disk/transport saving only; restored bytes verified, unchanged resident parameters and arithmetic",
    },
    "moljepa": {
        "name":"Mol-JEPA", "aliases":["mol-jepa"],
        "status":"pretrained_numerically_verified", "adapter_available":True,
        "repository":"https://github.com/Boehringer-Ingelheim/mol-jepa",
        "source_revision":"27f0065b69c73d18eec7a32ecc413b6dc3f33d90",
        "hf_id":"Flogrammer/Mol-JEPA", "hf_revision":"4c912b450175f31b5ba913a5dc921c03b27b985a",
        "contract":"SMILES-only input; predictions, CLS, all latent embeddings and requested attentions retained",
        "macos":"Actual float 32 CPU and Apple MPS numerical verification completed",
        "blockers":[],
    },
    "boltz2": {
        "name":"Boltz-2", "aliases":["boltz-2","boltz 2"],
        "status":"pretrained_shared_bundle_byte_verified_cpu_mps", "adapter_available":True,
        "repository":"https://github.com/jwohlwend/boltz",
        "source_revision":"b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc",
        "hf_id":"boltz-community/boltz-2", "hf_revision":"6fdef46d763fee7fbb83ca5501ccceff43b85607",
        "contract":"Complete original confidence and affinity native batch APIs; immutable frozen FP32 parameters; all native tensor outputs retained",
        "macos":"Native CPU and MPS full-schedule inference and fresh portable reload pass all 48 explicit output byte comparisons",
        "blockers":["Large-complex activation memory and biological quality unbenchmarked", "Cold load constructs both original architectures; pinned Boltz dependency and native chemistry assets remain required"],
        "registered_storage_before_bytes":4087121944, "registered_storage_after_bytes":2061868568,
        "artifact":"artifacts/boltz2-shared", "loader":"compressme.load_boltz2",
        "priority":"Current main target; general storage sharing accepted; invariant computation byte-verified but optional because speed is inconsistent",
        "tensor_file_bytes":2062669224, "lossless_packed_tensor_bytes":1759531547,
        "packing_note":"14.696% additional disk/transport saving from the already shared tensor file; complete restored bytes verified; no further resident weight or inference speed reduction",
        "validation_scope":"One 20-residue protein plus ethanol fixture; FP32 native default sampling schedules. All 48 tensors byte-identical before/after sharing and fresh reload on each backend. 49.55% joint registered storage reduction, unchanged logical parameter count and arithmetic; no speedup or biological benchmark",
    },
    "xcell": {
        "name":"X-Cell Mini", "aliases":["x-cell","xcell-mini"],
        "status":"blocked_unreleased", "adapter_available":False,
        "repository":"https://github.com/Xaira-Therapeutics/X-Cell",
        "source_revision":"7195c647b8316234ddaf51565701cfdaa939b443",
        "hf_id":"Xaira-Therapeutics/X-Cell", "hf_revision":"07737b494a3ea06c6a579d389fda0216ba61bf02",
        "contract":"Proposed AnnData/path input and perturbation symbol; predicted expression output",
        "macos":"Cannot assess unpublished operators",
        "blockers":["from_pretrained and inference are NotImplementedError stubs", "HF contains only README, image and .gitattributes; no weights"],
    },
    "state-st": {
        "name":"STATE ST (HVG Replogle, K562)", "aliases":["state","state-transition","st-hvg-replogle"],
        "status":"pretrained_numerically_verified", "adapter_available":True,
        "repository":"https://github.com/ArcInstitute/state",
        "source_revision":"9bbfe78a434a55205e4de834e1ea99f85f7a3add",
        "hf_id":"arcinstitute/ST-HVG-Replogle", "hf_revision":"bb6a9562cbbf1fd152df14cc53b4cc7517c77175",
        "checkpoint":"fewshot/k562/checkpoints/final.ckpt", "checkpoint_bytes":471695263,
        "contract":"Original HVG predict_step, count decoder output, metadata and full token-index lookup retained; frozen weights",
        "macos":"All predict_step outputs bitwise equal on CPU and Apple MPS float 32 for 1/7/64/128 cells",
        "blockers":["Published hidden328/head12 config needs compatible Transformers4", "Unused VCI decoder import needs a guarded lazy bridge"],
        "exact_candidate":"Generic bitwise-identical frozen embedding row deduplication; the real 32000×328 table is represented by one row without narrowing token IDs",
        "candidate_removed_parameters":10495672,
        "parameters_before":49396728, "parameters_after":38901056,
        "inference_tensor_bytes_before":197586912, "inference_tensor_bytes_after":155604224,
        "packed_bytes_before":132137704, "packed_bytes_after":132148096,
        "performance_note":"21.25% lower stored parameters; packed file is 0.008% larger because the original zero table already packs almost free; no active-FLOP speedup claimed",
        "validation_scope":"Synthetic numerical API probes on real pretrained weights, not a biological benchmark",
    },
    "state-se": {
        "name":"STATE SE-100M", "aliases":["state-embedding","se-100m"],
        "status":"pretrained_numerically_verified_cpu_mps", "adapter_available":True,
        "repository":"https://github.com/ArcInstitute/state",
        "source_revision":"9bbfe78a434a55205e4de834e1ea99f85f7a3add",
        "hf_id":"arcinstitute/SE-100M", "hf_revision":"bc72702320639217128df42673e94a9658f67d24",
        "checkpoint_bytes":869154536, "protein_embeddings_bytes":411105891,
        "stored_tensor_values":217284776,
        "contract":"AnnData h5ad embedding workflow, numerical gene-ID/count batch API and arbitrary raw-vector forward; all numerical heads retained; original name helper needs its external protein dictionary",
        "macos":"Finite tables: CPU-only, 105 comparisons, maxabs 6.68e-6. Lossless original table: 377 tensor byte comparisons per CPU/MPS portable reload; registered state reduced 6.31%, tested MPS calls 4.89-5.44x slower.",
        "blockers":["Finite-table artifact still fails MPS numerical gate", "Lossless original-table mode adds decode latency", "AnnData export remains CPU-only; CUDA and biological-quality validation remain separate"],
        "parameters_before":212038824, "parameters_after":151243944,
        "packed_weights_bytes":511190195,
        "lossless_original_embedding": {
            "devices": ["cpu", "mps"],
            "registered_state_bytes_before":848155296,
            "registered_state_bytes_after":794613047,
            "portable_reload_tensor_byte_comparisons_per_device":377,
            "contract":"Original numerical heads, gene-name helper, raw-vector forward and full weight snapshot retained; AnnData CPU-only",
            "performance_note":"Optional storage mode; no parameter or FLOP reduction, MPS latency 4.89-5.44x original in three paired rounds",
        },
        "performance_note":"Short CPU batch-API benchmark: 1.39x at 32 tokens and 1.11x at 2048 tokens; 5 interleaved rounds, no universal speed claim",
        "validation_scope":"Constructed numerical batches and seven synthetic h5ad variants using actual pretrained weights; AnnData preprocessing and all 1034 exported features bitwise equal, no biological quality benchmark",
        "exact_candidate":"Compile fixed gene lookup plus deterministic encoder into finite-domain lookup tables; preserve normalized and unnormalized paths and aliases",
    },
    "bioptimus-h0": {
        "name":"Bioptimus H-optimus-0", "aliases":["bioptimus","biooptimus","h-optimus-0"],
        "status":"metadata_surveyed_access_gated", "adapter_available":False,
        "repository":"https://github.com/bioptimus/releases/tree/main/models/h-optimus/v0",
        "source_revision":None,
        "hf_id":"bioptimus/H-optimus-0", "hf_revision":"b145cc1e6c6b30d3251aa8b1f844e6974188a743",
        "checkpoint_bytes":4539305018,
        "contract":"timm image tile tensor input and1536-dimensional tile features; preserve image normalization and registers",
        "macos":"Unverified pretrained memory/runtime;1.1B model",
        "blockers":["HF gated access requires user acceptance/contact sharing", "No weights or module validation"],
        "related_hf_models":[
            {"id":"bioptimus/H-optimus-1","revision":"3592cb220dec7a150c5d7813fb56e68bd57473b9","bytes":4539151464,"gated":"manual"},
            {"id":"bioptimus/H0-mini","revision":"5b5cc0505d19ae558270045eb0df8c34df4d9609","bytes":342974184,"gated":"manual","note":"Authors' distilled model; not evidence for compressor without distillation"}],
    },
    "omnicell": {
        "name":"OmniCell (BGIResearch)", "aliases":["omnicell-v1","omni-cell"],
        "status":"pretrained_submodule_candidates_rejected", "adapter_available":False,
        "repository":"https://github.com/BGIResearch/omnicell",
        "source_revision":"fc2d818d1d78345f0c7ccf686f34caff5bbd846a",
        "hf_id":None, "hf_revision":None,
        "checkpoint_source":"https://modelscope.cn/models/PJSucas/OmniCell-v1",
        "checkpoint_revision":"9a33a49daa919237189909e4d9ba48117f3ca12f",
        "checkpoint_bytes":294127497,
        "checkpoint_sha256":"12497eb1dc76985ca3bcd88845a6b6edb7fcfdbb6c6df43d3da56645d922c0c7",
        "identity_note":"Selected the single-cell/spatial foundation model, distinct from OmniCellAgent and OmniCellTOSG",
        "contract":"Gene tokens, expression and optional spatial context to cell/gene embeddings",
        "macos":"Actual pretrained embedder tested on CPU; whole-model macOS unvalidated. Existing use_flash=False branch changes attention layout and scale",
        "blockers":["Finite router increases resident storage and changes expert choice on a near-tie (0.2503 embedding error)", "Correct portable attention needs original FlashAttention reference validation", "No verified Hugging Face model ID"],
        "exact_candidate":"Parallel shared scalar affine experts save only 2048 values in real arithmetic; finite routing must retain the live original gene table and full load-balance outputs. Both need complete numerical gates",
    },
    "stformer": {
        "name":"stFormer (csh3)", "aliases":["st-former"],
        "status":"source_audited_cuda_backend_blocked", "adapter_available":False,
        "repository":"https://github.com/csh3/stFormer",
        "source_revision":"d69fcfd06fccadaef686ee7c223a8a8079350010",
        "hf_id":None, "hf_revision":None,
        "checkpoint_source":"https://zenodo.org/records/20755048",
        "contract":"Gene/value encoder and decoder sequences, masks and cross-attention bias; expression,cell and optional gene/CLS/GCL outputs",
        "macos":"CUDA-only FlashAttention assertions prevent existing GPU path on MPS",
        "blockers":["Weights packaged in9.3GB archive, no verified individual HF model", "Pinned flash_attn and torchtext stack needs a verified portable attention/padding adapter"],
        "exact_candidate":"Active model uses two independent scFoundation position tables and continuous-value encoders; unused GeneEncoder definition is not a reachable opportunity. Table sharing requires verified equality after training; cross-attention candidates must preserve masks and optional outputs",
    },
    "novomolgen": {'name': 'NovoMolGen',
     'aliases': ['novo-mol-gen', 'novo-molgen'],
     'status': 'pretrained_token_only_research_verified_cpu_mps',
     'adapter_available': False,
     'repository': 'https://github.com/chandar-lab/NovoMolGen',
     'source_revision': 'd7c520674701533e2b5e18a8ace826aed611e7b0',
     'collection': 'https://huggingface.co/collections/chandar-lab/novomolgen',
     'hf_id': 'chandar-lab/NovoMolGen_32M_SMILES_AtomWise',
     'hf_revision': 'dcd3f59261bebf84142c13617d4e129b4b0d0fdc',
     'checkpoint_bytes': 126236496,
     'contract': 'SMILES/token IDs to causal logits and sampled sequences; optional hidden states, masks, positions '
                 'and KV cache. Native HF inputs_embeds remains part of its broader API',
     'macos': 'Original native Llama runs on CPU/MPS; token-only shape-specific lookup research passes recorded '
              'complete outputs bitwise on both. No production compressed adapter or material speedup',
     'blockers': ['First-layer token lookup needs an explicit token-only contract to remove projection weights',
                  'Research prototype does not preserve arbitrary inputs_embeds or provide a portable whole-model recipe',
                  'Other five checkpoints are metadata-audited only; new shapes/backends require validation',
                  'Main custom API and native HF branch have different mask behavior; BPE tokenizer declares dropout '
                  '0.1'],
     'exact_candidate': 'Finite-domain precomputation of the first RMSNorm and Q/K/V projections; retain token '
                        'embeddings for the residual and hidden-state outputs',
     'variants': [{'hf_id': 'chandar-lab/NovoMolGen_32M_SMILES_AtomWise',
                   'revision': 'dcd3f59261bebf84142c13617d4e129b4b0d0fdc',
                   'checkpoint_bytes': 126236496,
                   'dtype': 'float32',
                   'stored_values': 31556096,
                   'vocabulary_size': 84},
                  {'hf_id': 'chandar-lab/NovoMolGen_32M_SMILES_BPE',
                   'revision': 'e1cab1287c3906ff9c7646e6c3d714f3a8c9a40e',
                   'checkpoint_bytes': 127940448,
                   'dtype': 'float32',
                   'stored_values': 31982080,
                   'vocabulary_size': 500},
                  {'hf_id': 'chandar-lab/NovoMolGen_157M_SMILES_AtomWise',
                   'revision': '9a9a78c46a8a745044ae17fbf0a39b027afdc522',
                   'checkpoint_bytes': 629725448,
                   'dtype': 'float32',
                   'stored_values': 157425280,
                   'vocabulary_size': 84},
                  {'hf_id': 'chandar-lab/NovoMolGen_157M_SMILES_BPE',
                   'revision': 'e133b9792d10e0d9ebd8fb5245c41106ba4f1e63',
                   'checkpoint_bytes': 631855376,
                   'dtype': 'float32',
                   'stored_values': 157957760,
                   'vocabulary_size': 500},
                  {'hf_id': 'chandar-lab/NovoMolGen_300M_SMILES_AtomWise',
                   'revision': '2be770a56c42eb29c5c74c5461e39be936827eb0',
                   'checkpoint_bytes': 1208707848,
                   'dtype': 'float32',
                   'stored_values': 302168832,
                   'vocabulary_size': 84},
                  {'hf_id': 'chandar-lab/NovoMolGen_300M_SMILES_BPE',
                   'revision': 'e63d6e00dec33370dc0ad846ae2b66b1bfb6813a',
                   'checkpoint_bytes': 1211263760,
                   'dtype': 'float32',
                   'stored_values': 302807808,
                   'vocabulary_size': 500}],
     'validation_scope': '32M AtomWise token-only whole-model research: 257 cases and 12599 tensor comparisons '
                         'per backend bitwise on CPU and MPS, including caches/generation/hidden states. '
                         'CPU saves 528896 parameters (1.676%); MPS saves 399872 (1.267%). Timing essentially neutral'},
}


_INACTIVE = {
    "xcell": "Removed from current scope by user: no released weights",
    "omnicell": "Removed from current scope by user",
    "stformer": "Set aside by user: large archive and substantial backend porting",
    "bioptimus-h0": "Deferred: supplied Hugging Face access still returned 403 GatedRepo for H-optimus-0 and H-optimus-1",
}


def get_target(name):
    """Return independent metadata; do not load code, weights or dependencies."""
    key=name.casefold().strip()
    if key not in _TARGETS:
        key=next((k for k,v in _TARGETS.items() if key in v["aliases"]),None)
    if key is None:
        raise KeyError(f"Unknown target: {name}")
    result = {"id":key,"audited_on":"2026-09-07",**copy.deepcopy(_TARGETS[key]),
              "in_scope": key not in _INACTIVE}
    if key in _INACTIVE:
        result["scope_note"] = _INACTIVE[key]
    if key == "novomolgen":
        result["scope_note"] = "Only the existing 32M SMILES AtomWise checkpoint is in scope; other variants were excluded by user"
        result["variants"] = [v for v in result["variants"] if v["hf_id"] == result["hf_id"]]
        result["blockers"] = [v for v in result["blockers"] if not v.startswith("Other five")]
    return result


def list_targets(*, adapter_available=None, include_inactive=False):
    """List current targets; explicitly opt into archived/deferred survey entries."""
    return tuple(get_target(k) for k,v in _TARGETS.items()
                 if (include_inactive or k not in _INACTIVE)
                 and (adapter_available is None or v["adapter_available"]==adapter_available))
