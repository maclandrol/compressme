# Mol-JEPA checkpoint audit and early Boltz-2 source inspection

Source inspection and local experiments recorded on 2026-09-07. The findings apply to the inspected execution paths; each architecture needs its own checks.

## Mol-JEPA artifacts

The SMILES-only artifact preserves all 12 predicted modality embeddings, the CLS vector, all 13 latent embeddings and optional attention maps. Affine reachability plus exact bilinear graph attention reduces the model from 45,406,721 to 19,763,160 parameters. Reversible packing reduces the weight file to approximately 67.51 MB, without further numerical changes. The full multimodal variant retains 35,225,561 parameters. The pass-by-pass experiments below explain those reductions. Export checks and later timings are in `benchmarks/` and `artifacts/export_verification.json`.

## Mol-JEPA

- Official source: https://github.com/Boehringer-Ingelheim/mol-jepa/tree/27f0065b69c73d18eec7a32ecc413b6dc3f33d90
- Official paper: https://arxiv.org/abs/2608.22642
- Pinned inference module: https://huggingface.co/Flogrammer/Mol-JEPA/blob/4c912b450175f31b5ba913a5dc921c03b27b985a/modeling_moljepa.py
- Pinned model configuration: https://huggingface.co/Flogrammer/Mol-JEPA/blob/4c912b450175f31b5ba913a5dc921c03b27b985a/config.json
- Weights: https://huggingface.co/Flogrammer/Mol-JEPA/resolve/4c912b450175f31b5ba913a5dc921c03b27b985a/model.safetensors
- Local model: `/private/tmp/compressme-models/Mol-JEPA`; safetensors file 181,651,424 bytes.
- Checkpoint values include buffers; actual module parameter count is **45,406,721**.
- License stated by both official repositories: CC BY-NC 4.0. Do not bundle its checkpoint into a compressor package under the package's own license.

Mol-JEPA produces molecular representations; it does not generate protein structures. Public `forward(smiles_list, return_attn=False, embeddings_data=None)` returns `MolJEPAOutput` with predictions `[B,12,512]`, cls `[B,512]`, embeddings `[B,13,512]`, and optional attention arrays. With plain SMILES it executes only the graph encoder. Optional embeddings_data activates additional encoders.

The graph encoder contains 20,216,320 parameters. The 2-layer, 512-dimensional transformer contains 6,305,792; the 13 prediction readouts contain 3,414,528. Eleven other encoders contain 15,462,401 parameters and are unreachable for `embeddings_data=None`. The graph convolution uses PyG Linear, not torch.nn.Linear. Its query/key/value projections are each 4096 by 512 (8 heads, 512 channels); a generic nn.Linear-only compressor misses most graph parameters.

### Exact affine composition

Atom features are projected 82 to 512, then supplied directly to the first graph convolution. Before any nonlinearity, four projections consume this same result: Q, K, V (4096 outputs each), and the skip map (512 outputs). Compose each with the 82-to-512 map while retaining the original map for the separate residual branch. The new projections have 82 input dimensions. This removes exactly `(4096*3+512)*(512-82)=5,504,000` parameters. Edge projections, attention softmax, head averaging, residual addition, normalization, pooling, other encoders and all public outputs remain intact.

The composition is a classical real-arithmetic identity and changes floating-point operation order. The checkpoint finding is the redundant affine expansion on an active inference route: the preceding 82-dimensional input proves that the wider consumer maps are unnecessary.

### Explicit domain specialization

Plain SMILES leaves eleven expert encoders inactive. Removing them saves 15,462,401 parameters while preserving all prediction heads. The resulting artifact must reject embeddings_data other than None, or load the excluded encoders through an explicit fallback to support optional modalities.

### Local evidence

Historical adapter: `/private/tmp/compressme_moljepa_adapter.py`.

- `fuse_moljepa_input_projections(model, inplace=False) -> (model, report)`: preserves the full multimodal input contract; 45,406,721 to 39,902,721 parameters, 12.12% reduction.
- `specialize_moljepa_smiles(model, inplace=False) -> (model, report)`: restricts embeddings_data to None with a clear guard; 45,406,721 to 29,944,320 parameters, 34.05% reduction.
- Combined: 24,440,320 parameters, 46.17% reduction.

CPU, float32, 12 varied SMILES, 4 CPU threads: inactive-encoder removal gives bit-identical predictions/CLS/embeddings. Affine composition gives max absolute prediction error 2.503e-6, relative L2 3.397e-7; max CLS error 3.278e-7; max latent embedding error 5.960e-7. Five full-API timings have medians 27.572 ms original and 24.913 ms combined, approximately 1.11x speedup. Evidence `/private/tmp/compressme_moljepa_audit.json`.

Additional mixed optional ECFP/Boltz-prediction inputs and `return_attn=True` pass atol=rtol=1e-5. Maximum prediction error 1.669e-6; attention errors below 6e-7. Non-inplace conversion leaves original projection shapes intact. Evidence `/private/tmp/compressme_moljepa_extended.json`.

A separate Apple M5 GPU experiment used MPS, float32 and 4 compounds: affine fusion max prediction error 1.192e-6, relative 2.32e-7; full SMILES API 13.236 to 12.724 ms (1.040x), graph-only component 6.371 to 5.862 ms (1.087x). No unsupported operator or fallback was needed. Evidence `/private/tmp/compressme_moljepa_exact_bench.json`. CPU and GPU timings have different batch sizes and should not be compared directly.

The HF source was read before execution. Local dependencies were torch2.14.0, torch-geometric2.8.0.post1, transformers5.16.1, rdkit2026.3.6, molfeat0.11.0, safetensors0.8.0. `from_pretrained` hits an upstream CPU/meta buffer mismatch with transformers5. Load unchanged source with AutoConfig.from_pretrained(local_path, trust_remote_code=True, local_files_only=True), AutoModel.from_config(config, trust_remote_code=True), then strict safetensors load. Set HF_HOME/HF_MODULES_CACHE to writable paths. The outer HF forward runs under no_grad; fine-tuning needs a lower-level training route.

## Boltz-2

This section records the initial source audit. The later complete CPU/MPS runs,
shared-storage checks and portable reload results are in
[the Boltz-2 report](boltz2.md).

Official code inspected at https://github.com/jwohlwend/boltz/tree/b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc, locally `/tmp/compressme-audit-boltz`. Official code references these checkpoints:

- https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt (2.29 GB)
- https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt (2.06 GB)
- https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar (1.86 GB)

Full checkpoint execution had not been attempted at this stage. The observations below come from source inspection; the linked Boltz-2 report covers model validation.

Norm-to-linear sites identified in the source:

- `model/layers/transition.py`: shared LayerNorm directly feeds fc1 and fc2; gated SiLU product and fc3 follow. Composite transition adapter needed for chunked path, which directly slices fc1/fc2/fc3.weight.
- `model/layers/pairformer.py`: pre_norm_s feeds attention projections Q/K/V/g; shared projection basis can be reused. Transition_s and transition_z use the preceding module.
- `model/layers/attentionv2.py`: proj_z is Sequential(LayerNorm, Linear, Rearrange).
- `model/modules/transformersv2.py`: AdaLN.s_norm feeds s_scale and s_bias. AdaLN's output itself has input-dependent scale/bias, so treating it as a fixed LayerNorm affine subspace is invalid.
- `model/modules/diffusionv2.py`: s_to_a_linear is Sequential(LayerNorm, Linear).
- `model/modules/trunkv2.py`: atom encoder bias projection is Sequential(LayerNorm, Linear); template z_norm feeds z_proj.

Several consumers bypass child forwards: pair_averaging and outer_product_mean slice weights directly, and cuEquivariance kernels take module weights. Triangle primitive Linear subclasses also alter dtype handling. Replacing every nn.Linear would miss these behaviours. Integration needs external kernels disabled and either a composite adapter or a verified execution graph. Upstream CLI has accelerator choices gpu/cpu/tpu, and use_kernels is disabled in predict setup without a compatible CUDA device. Complete MPS execution was untested at this initial stage; the CLI choices alone could not establish whether it would work. The separate Mol-JEPA graph experiment had already run on MPS.

Boltz also allocates quadratic pair activations and repeats diffusion computation. Smaller checkpoints alone therefore establish neither a speedup nor that a large complex fits in 16GB unified memory. Validation must retain structural coordinates, confidence and affinity outputs.
