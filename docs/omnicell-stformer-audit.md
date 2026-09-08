# stFormer and OmniCell: why the tested candidates did not qualify

Neither target produced a supported compressed model. OmniCell's finite router
passed a local numerical check but changed an expert choice, causing a 0.2503035069
embedding error. Its shared-expert rewrite saved only 8,192 bytes and failed the
floating-point gate at large inputs. stFormer remains a source and metadata
audit; its weights were not extracted from the full release archive.

Audited on 2026-09-07. Biological fidelity and speed remain unmeasured.

## stFormer: no verified frozen table to remove

Repository revision: `d69fcfd06fccadaef686ee7c223a8a8079350010`.

- `pretraining/pretraining.py:219` loads scFoundation with `load_model_frommmf`.
- Lines 228–231 pass independent deep copies of `token_emb` and `pos_emb` into `TransformerModel`.
- `tasks/scfoundation/load.py:125–148` constructs the model and loads a state dictionary; it does not freeze parameters.
- `tasks/scfoundation/pretrainmodels/mae_autobin.py:111–112` constructs continuous `AutoDiscretizationEmbedding2` and ordinary `nn.Embedding(max_seq_len+1, embed_dim)` for position/gene IDs.
- `stformer/model.py:69–72` and `85–88` call these position and continuous-value encoders, then add their outputs.
- `stformer/model.py:142–158` defines `GeneEncoder` with Embedding→LayerNorm, but repository search found no construction/call of `GeneEncoder` outside that definition.
- `pretraining/pretraining.py:236–242` logs trainable parameter counts; its freeze section is commented out. Lines 265–267 pass all `model.parameters()` to Adam, so source alone establishes neither frozen weights nor equality of the position tables after training.
- Some downstream classification/perturbation notebooks freeze the loaded backbone, but that happens after pretrained copies may have diverged and does not establish equality.

`stformer/model.py:105–139` accepts separate encoder/decoder gene IDs and values,
both padding masks and cross-attention bias. It returns `mlm_output` and
`cell_emb`, with `gene_emb`, `cls_output` and `gcl_output` controlled by existing
flags. Continuous MLP/softmax computations and special mask/pad values belong to
the value encoder. Absorbing that branch into finite gene lookup would change
the input domain.

A first-1-MiB HTTP range read from Zenodo found source notebooks followed by a
105,978,498-byte H5AD file before reaching the weights. The release is a
gzip-compressed tar stream without a published index; range support alone does
not provide arbitrary checkpoint extraction. The full archive was not downloaded.

Primary links: [active construction](https://github.com/csh3/stFormer/blob/d69fcfd06fccadaef686ee7c223a8a8079350010/pretraining/pretraining.py#L219), [model forward](https://github.com/csh3/stFormer/blob/d69fcfd06fccadaef686ee7c223a8a8079350010/stformer/model.py#L62), [scFoundation embedding](https://github.com/csh3/stFormer/blob/d69fcfd06fccadaef686ee7c223a8a8079350010/tasks/scfoundation/pretrainmodels/mae_autobin.py#L111), [release archive](https://zenodo.org/records/20755048).

## OmniCell: finite routing costs more storage

Repository revision: `fc2d818d1d78345f0c7ccf686f34caff5bbd846a`.

`OmniCell/Transformer/model.py:57–64` creates 5 shared Linear(1,512,bias=False) experts, 10 routed experts of the same size, and a 512→512→ReLU→10 router. Lines 138–144 look up each gene vector, route using that vector alone, and add the unmodified gene vector to the value-encoded result.

For gene ID g and continuous expression v, write E_g for its gene embedding, s_j for shared expert vectors, r_k for routed expert vectors, and a_g for the top-5-masked softmax router weights. In real arithmetic the embedder is:

`E_g + v * (sum_j s_j + sum_k a_g[k] * r_k)`.

Separating the finite gene selector from expression leaves these constraints:

- Full vocabulary is 60,607. `E_g` has a live output consumer and stays: 31,030,784 values.
- The router has 512×512 + 512×10 = 267,264 values. A full 60,607×10 logit/weight table has 606,070 values, increasing storage by 338,806 float values (1,355,224 bytes in FP32) if the router is removed and its input gene table remains.
- Folding 5 shared experts into one saves (5−1)×512 = 2,048 values (8,192 FP32 bytes), separately from any finite-table tradeoff.
- Compiling a 512-dimensional expression slope for every gene would add another 31,030,784-value table, working against the storage objective.
- `load_balance_loss` is updated as a public attribute on the value embedder, embedder, and transformer. Its calculation uses full routing probabilities and top-k usage across the current batch (`model.py:96–114`); a candidate that only stores slopes and drops this observable changes behavior.
- `MoE4Embedder.forward` itself accepts arbitrary dense gene vectors. A token-table replacement belongs at the parent gene-ID boundary or must retain the dense-vector route. It is not a drop-in replacement for the raw-vector submodule API.
- `Transformer.forward(gene,value,index)` returns all contextual token embeddings, while the separately registered `output.W` is used by higher-level projection utilities. It cannot be removed merely because `Transformer.forward` does not call it.

The attention fallback also blocks a faithful port. At lines 228–230, q/k/v use
FlashAttention's B×T×H×D layout. Lines 266–280 pass that layout unchanged to
PyTorch SDPA, which interprets it as B×H×T×D, and multiply Q by `self.scale`
before SDPA applies its own default scaling. A port would need to transpose T/H,
scale once, and pass a same-output check against original flash semantics. The
existing fallback cannot serve as that reference.

Official checkpoint: ModelScope PJSucas/OmniCell-v1, revision `9a33a49daa919237189909e4d9ba48117f3ca12f`, `backbone.pth` 294,127,497 bytes, SHA256 `12497eb1dc76985ca3bcd88845a6b6edb7fcfdbb6c6df43d3da56645d922c0c7`. Configuration is 298 bytes, SHA256 `bb4fd00c29d8764058d986dbe3d2c2b036f1c3d0df91d661c63de71c2ea49322`. The file host honors range requests, and the checkpoint is separately available without the stFormer archive. Loading uses `weights_only=True`; unsafe pickle allowlisting is not required or authorized here.

Primary links: [router and embedding](https://github.com/BGIResearch/omnicell/blob/fc2d818d1d78345f0c7ccf686f34caff5bbd846a/OmniCell/Transformer/model.py#L47), [attention paths](https://github.com/BGIResearch/omnicell/blob/fc2d818d1d78345f0c7ccf686f34caff5bbd846a/OmniCell/Transformer/model.py#L189), [checkpoint loader](https://github.com/BGIResearch/omnicell/blob/fc2d818d1d78345f0c7ccf686f34caff5bbd846a/OmniCell/utils/checkpoint_loader.py#L32), [official weights](https://modelscope.cn/models/PJSucas/OmniCell-v1).

## The real weights expose a routing failure

The complete downloaded checkpoint matches its published SHA256 and loads with
`torch.load(..., weights_only=True, mmap=True)`. It contains 73,521,152 tensor
values. Extracting the original class definitions from the pinned AST allowed
submodule checks without unavailable FlashAttention or unsafe pickle globals.
The full transformer was not run.

`compile_finite_lookup` evaluated Embedding→Linear→ReLU→Linear for all 60,607
gene IDs on CPU FP32, with enumeration chunks of 1,024, validation chunks of 257
and shape probes. Local logits passed at a maximum absolute error of 1.296401e-6.
All 60,607 routing rows are bitwise distinct, and the original gene table has
zero all-zero rows, leaving no saving from exact row dictionary sharing. Keeping
the live original gene embedding increases the embedder from 31,305,728 to
31,644,534 values: 338,806 more.

Probes at B×T shapes 1×1, 1×32, 4×32, 1×2000 and 2×2000, with continuous
values in [0,10], passed atol=rtol=1e-5, including the load-balance attribute.
The four nonsingleton probes were bitwise identical.

Among the 128 genes with the smallest top-k boundary gaps, gene ID 6281 at
shape 1×1 and expression 10 changes one selected expert. The compiled boundary
probability gap is 7.4505806e-9, yet full embedder error reaches 0.2503035069:
all 512 features fail the unchanged 1e-5 gate. Load-balance error is just
1.1920929e-7 and would miss the failure on its own. The candidate is rejected.
A local tolerance on continuous logits cannot certify the downstream discrete
routing choice; validation must include decision margins and complete outputs.

Summing the five shared expert weight vectors in FP64 and storing the result
in FP32 reduces that submodule from 2,560 to 512 values, saving 8,192 bytes.
On a deterministic 4,097-point grid, maximum error is 5.96e-8 for |v|≤1,
9.54e-7 for |v|≤10 and 7.63e-6 for |v|≤100. At |v|≤1,000,000, it reaches
0.0625, with 2,990 values failing atol=rtol=1e-5. The affine identity is exact
in real arithmetic; the failure rules out an unbounded FP32 fidelity guarantee.
The rewrite has not passed a whole-model check.

A separate synthetic attention-layout counterexample, shape B×T×H×D=2×5×3×4, compares the repository fallback with canonical SDPA respecting flash semantics. They return the same shape but differ by up to 1.9415. This establishes the fallback issue independently of the checkpoint. PyTorch documents SDPA's final two query axes as sequence and features and its default 1/sqrt(E) scaling: [official API](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html).

Evidence is retained in `benchmarks/omnicell_submodule_audit.json`,
`benchmarks/omnicell_attention_counterexample.json` and the reproducible
`experiments/omnicell_submodule_audit.py` (requires the pinned source and weights).

Timing was not measured because CPU time was shared with other correctness
checks.
