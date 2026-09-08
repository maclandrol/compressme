# Optional exact compression experiments

These experiments are separate from the working SMILES artifact. They are not production integrations and were not used to claim the current artifact's size or speed. Both preserve the relevant mathematical functions in real arithmetic; neither promises bit-identical floating-point outputs. Neither uses training, distillation, quantisation or low-rank truncation.

## Expanded-input self-attention

**General primitive:** `ExpandedInputSelfAttention` in `/private/tmp/compressme_expanded_mha.py`.

It contracts a known affine expansion into a following self-attention module. Q/K are contracted into bilinear score maps; output projection blocks are contracted into V; attention aggregates augmented raw features including a constant coordinate so value biases remain correct with dropout. The input width changes inside the primitive, while `ContractedAtomsEncoder` preserves the public Mol-JEPA API. Its explicit `padding_after_projection=True` contract requires padded raw rows to be zero and preserves the original padded query behavior separately.

The actual optional UMA encoder expands 128-dimensional atom features to 512 before attention with four 128-dimensional heads. Its first attention block has 1,050,624 parameters; the replacement has 331,264, saving **719,360 parameters**. The previously compressed full model changes from **35,225,561 to 34,506,201 parameters**. It does not benefit the SMILES-only artifact because that artifact already excludes the unused UMA encoder.

Public API validation: sixteen inputs with UMA atom counts 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 32, 64, 128 and 256; all predictions, CLS outputs, latent embeddings and attention maps passed `atol=rtol=1e-5` on CPU float32. Maximum prediction difference was **3.70e-6**, relative L2 **3.87e-7**; attention differences were below **1.32e-6**. Calls without UMA were bit-identical to the prior compressed model.

An internal UMA encoder stress test produced differences above a 1e-5 absolute threshold at some coordinates: 1.42e-5 in the final recorded case and 2.16e-5 in an earlier random case. Its separately stated internal threshold is 5e-5 absolute plus 1e-5 relative. The public model output threshold remains 1e-5. No actual-model MPS validation or latency claim was made for this experimental pass.

Files:

- Implementation: `/private/tmp/compressme_expanded_mha.py`
- Sixteen passing float64 semantic tests (biases, padding, dropout, attention weights, input derivatives): `/private/tmp/test_expanded_mha.py`
- Actual-model audit script: `/private/tmp/compressme_expanded_mha_audit.py`
- Actual-model recorded evidence: `/private/tmp/compressme_expanded_mha_audit.json`

## Attention gauge fixing with implicit identity blocks

**General primitive:** `GaugeFixedMHA` in `/private/tmp/compressme_mha_gauge.py`.

For each attention head, select an invertible square block P of Q. Replace Q by P^-1 Q and K by P^T K, preserving their dot products. The selected Q columns become the identity and need not be stored. Apply the analogous change to V and its corresponding output block O: V becomes T^-1 V and O becomes O T. Query/value biases are transformed; key bias can be removed because it adds only a softmax-row constant. No head or output dimension is removed.

The two cross-modal transformer layers save **263,168 parameters**, changing the SMILES model from **19,763,160 to 19,499,992 parameters**. Pivot indices add 65,536 bytes in this prototype, making the raw tensor storage reduction approximately **987,136 bytes** before lossless coding. Actual selected pivot condition numbers were between **25.5 and 66.5**; the prototype rejects condition numbers above 1,000.

CPU float32 validation on all 64 verification SMILES, both with and without returned attention maps, passed `atol=rtol=1e-5` for every public output. Maximum prediction difference versus the original model was **5.60e-6**, relative L2 **5.66e-7**; maximum attention difference was **3.28e-6**. Ten interleaved CPU runs were essentially neutral: prior compressed model median **16.49 ms**, gauge model **16.53 ms** (12-input timing subset).

**This is checkpoint compression only.** To retain native PyTorch fast paths, the prototype materializes and caches dense effective Q/K/V weights during inference. The two caches contain **6,291,456 bytes** of float32 matrices; the resulting attention resident memory is therefore larger, despite the smaller checkpoint. Caches are excluded from state_dict and checked against parameter versions. There is no claim of a GPU memory reduction or actual-model MPS validation.

Files:

- Implementation: `/private/tmp/compressme_mha_gauge.py`
- Actual-model numerical and timing test script: `/private/tmp/compressme_mha_gauge_audit.py`
- Actual-model recorded evidence: `/private/tmp/compressme_mha_gauge_audit.json`
- The 64 verification inputs used by both passes: `/private/tmp/compressme_verification_64.json`

Gauge fixing is not presented as new mathematical theory. It is an exact parameter redundancy that a model compressor can exploit. Independent fine-tuning in either transformed parameterization does not preserve the original optimization trajectory.
