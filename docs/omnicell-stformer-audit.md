# Pinned source audit: stFormer and OmniCell

Inspected on 2026-09-07. This records source and checkpoint metadata evidence; it does not establish whole-model compression or biological fidelity.

## stFormer active call path

Repository revision: `d69fcfd06fccadaef686ee7c223a8a8079350010`.

- `pretraining/pretraining.py:219` loads scFoundation with `load_model_frommmf`.
- Lines 228–231 pass independent deep copies of `token_emb` and `pos_emb` into `TransformerModel`.
- `tasks/scfoundation/load.py:125–148` constructs the model and loads a state dictionary; it does not freeze parameters.
- `tasks/scfoundation/pretrainmodels/mae_autobin.py:111–112` constructs continuous `AutoDiscretizationEmbedding2` and ordinary `nn.Embedding(max_seq_len+1, embed_dim)` for position/gene IDs.
- `stformer/model.py:69–72` and `85–88` call these position and continuous-value encoders, then add their outputs.
- `stformer/model.py:142–158` defines `GeneEncoder` with Embedding→LayerNorm, but repository search found no construction/call of `GeneEncoder` outside that definition.
- `pretraining/pretraining.py:236–242` only logs trainable parameter counts; the apparent freeze section is commented out. Lines 265–267 pass **all** `model.parameters()` to Adam. Consequently, neither freezing nor post-training equality of the two position tables follows from source.
- Some downstream classification/perturbation notebooks freeze the loaded backbone, but that happens after pretrained copies may have diverged and does not establish equality.

Forward contract (`stformer/model.py:105–139`): separate encoder/decoder gene IDs and values, both padding masks, and cross-attention bias; output includes `mlm_output` and `cell_emb`, with `gene_emb`, `cls_output`, and `gcl_output` controlled by existing flags. Value encoding includes continuous MLP/softmax computations and special mask/pad values, so finite gene lookup cannot absorb that value-dependent branch without changing the input domain.

Zenodo supports byte ranges. A bounded first-1-MiB compressed read was successful and identified source notebooks followed by a 105,978,498-byte H5AD file before reaching model weights. Because this is a gzip-compressed tar stream without a published index, arbitrary checkpoint extraction by HTTP range is not established. No full archive was downloaded.

Primary links: [active construction](https://github.com/csh3/stFormer/blob/d69fcfd06fccadaef686ee7c223a8a8079350010/pretraining/pretraining.py#L219), [model forward](https://github.com/csh3/stFormer/blob/d69fcfd06fccadaef686ee7c223a8a8079350010/stformer/model.py#L62), [scFoundation embedding](https://github.com/csh3/stFormer/blob/d69fcfd06fccadaef686ee7c223a8a8079350010/tasks/scfoundation/pretrainmodels/mae_autobin.py#L111), [release archive](https://zenodo.org/records/20755048).

## OmniCell reachable finite routing and affine experts

Repository revision: `fc2d818d1d78345f0c7ccf686f34caff5bbd846a`.

`OmniCell/Transformer/model.py:57–64` creates 5 shared Linear(1,512,bias=False) experts, 10 routed experts of the same size, and a 512→512→ReLU→10 router. Lines 138–144 look up each gene vector, route using that vector alone, and add the unmodified gene vector to the value-encoded result.

For gene ID g and continuous expression v, write E_g for its gene embedding, s_j for shared expert vectors, r_k for routed expert vectors, and a_g for the top-5-masked softmax router weights. In real arithmetic the embedder is:

`E_g + v * (sum_j s_j + sum_k a_g[k] * r_k)`.

This is a reusable finite-selector/continuous-scalar pattern, but the following distinctions matter:

- Full vocabulary is 60,607. `E_g` has a live output consumer and stays: 31,030,784 values.
- The router has 512×512 + 512×10 = 267,264 values. A full 60,607×10 logit/weight table has 606,070 values, increasing storage by 338,806 float values (1,355,224 bytes in FP32) if the router is removed and its input gene table remains.
- Folding 5 shared experts into one saves (5−1)×512 = 2,048 values (8,192 FP32 bytes), separately from any finite-table tradeoff.
- Compiling one complete 512-dimensional expression slope for every gene would add another 31,030,784-value table and is unlikely to be a sensible compression.
- `load_balance_loss` is updated as a public attribute on the value embedder, embedder, and transformer. Its calculation uses full routing probabilities and top-k usage across the current batch (`model.py:96–114`); a candidate that only stores slopes and drops this observable changes behavior.
- `MoE4Embedder.forward` itself accepts arbitrary dense gene vectors. A token-table replacement belongs at the parent gene-ID boundary or must retain the dense-vector route. It is not a drop-in replacement for the raw-vector submodule API.
- `Transformer.forward(gene,value,index)` returns all contextual token embeddings, while the separately registered `output.W` is used by higher-level projection utilities. It cannot be removed merely because `Transformer.forward` does not call it.

Attention correctness blocker: q/k/v are shaped B×T×H×D at lines 228–230. FlashAttention consumes that layout. The fallback at lines 266–280 passes it unchanged to PyTorch SDPA, which consumes B×H×T×D semantics, and also multiplies Q by `self.scale` before SDPA's own default scaling. A correct port would transpose T/H and apply scaling only once, but needs a same-output reference gate against the original flash semantics. This audit does not silently treat the existing fallback as equivalent.

Official checkpoint: ModelScope PJSucas/OmniCell-v1, revision `9a33a49daa919237189909e4d9ba48117f3ca12f`, `backbone.pth` 294,127,497 bytes, SHA256 `12497eb1dc76985ca3bcd88845a6b6edb7fcfdbb6c6df43d3da56645d922c0c7`. Configuration is 298 bytes, SHA256 `bb4fd00c29d8764058d986dbe3d2c2b036f1c3d0df91d661c63de71c2ea49322`. The file host honors range requests, and the checkpoint is separately available without the stFormer archive. Loading uses `weights_only=True`; unsafe pickle allowlisting is not required or authorized here.

Primary links: [router and embedding](https://github.com/BGIResearch/omnicell/blob/fc2d818d1d78345f0c7ccf686f34caff5bbd846a/OmniCell/Transformer/model.py#L47), [attention paths](https://github.com/BGIResearch/omnicell/blob/fc2d818d1d78345f0c7ccf686f34caff5bbd846a/OmniCell/Transformer/model.py#L189), [checkpoint loader](https://github.com/BGIResearch/omnicell/blob/fc2d818d1d78345f0c7ccf686f34caff5bbd846a/OmniCell/utils/checkpoint_loader.py#L32), [official weights](https://modelscope.cn/models/PJSucas/OmniCell-v1).

## Actual OmniCell numerical experiments

The downloaded checkpoint's entire SHA256 was verified against the published hash. `torch.load(..., weights_only=True, mmap=True)` succeeded. There are 73,521,152 tensor values in the checkpoint. Original audited class definitions were extracted directly from the pinned AST to avoid importing unavailable FlashAttention; no unsafe pickle globals were permitted. No whole-transformer forward or biological dataset result is claimed.

Finite router: the existing generic `compile_finite_lookup` evaluated Embedding→Linear→ReLU→Linear across all 60,607 gene IDs on CPU FP32, using enumeration chunks of 1,024 and validation chunks of 257 plus its shape probes. The local logit comparison passed: max absolute error 1.296401e-6. All 60,607 stored routing rows are distinct bitwise, and the original gene table has zero all-zero rows; exact row dictionary sharing did not offer a saving. Retaining the live original gene embedding changes the whole embedder from 31,305,728 to 31,644,534 values: **338,806 more values**.

Normal embedder probes at B×T shapes 1×1, 1×32, 4×32, 1×2000 and 2×2000, with continuous values sampled in [0,10], passed atol=rtol=1e-5 including the load-balance attribute. The four nonsingleton probes were bitwise identical. This initially reassuring result was insufficient.

An additional targeted probe selected the 128 genes with the smallest top-k boundary gaps. **Gene ID 6281, shape 1×1, expression 10** changes one selected expert between the original router and the compiled table. The compiled boundary probability gap was only 7.4505806e-9. The resulting full embedder max absolute error is **0.2503035069** and all 512 embedding features fail the unchanged 1e-5 gate. Load-balance error is only 1.1920929e-7, illustrating why that metric alone would miss the failure. This candidate is rejected. The generic lesson is that a local continuous-output tolerance does not certify a downstream discontinuous routing decision; decision margins and whole-output gates matter.

Parallel shared scalar experts: summing the five actual expert weight vectors in FP64 and storing the result in FP32 reduces this small submodule from 2,560 to 512 values. On a deterministic 4,097-point grid, output max error was 5.96e-8 for |v|≤1, 9.54e-7 for |v|≤10, and 7.63e-6 for |v|≤100. At |v|≤1,000,000, max error reached 0.0625 and 2,990 values failed atol=rtol=1e-5. This remains an exact real-arithmetic affine identity, but it is not an unbounded FP32 fidelity guarantee or a validated whole-model optimization. It saves only 8,192 bytes of FP32 weights in this checkpoint.

A separate synthetic attention-layout counterexample, shape B×T×H×D=2×5×3×4, compares the repository fallback with canonical SDPA respecting flash semantics. They return the same shape but differ by up to 1.9415. This establishes the fallback issue independently of the checkpoint. PyTorch documents SDPA's final two query axes as sequence and features and its default 1/sqrt(E) scaling: [official API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

Evidence is retained in `benchmarks/omnicell_submodule_audit.json`,
`benchmarks/omnicell_attention_counterexample.json` and the reproducible
`experiments/omnicell_submodule_audit.py` (requires the pinned source and weights).

No speedups were measured in this audit; CPU time was shared with other
correctness checks. No supported whole-model artifact was produced from these
two targets.
