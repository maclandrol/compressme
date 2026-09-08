# Request-boundary invariant sampling candidate

`boltz2_runtime.py` is a staged source-specific adapter, not yet enabled by the package. Its public entry is:

```python
with torch.no_grad():
    with boltz2_invariant_sampling(model, immutable_request=True) as audit:
        outputs = model.predict_step(batch, 0)
```

The model can be either native Boltz-2 checkpoint loaded through the shared artifact. Both structure/confidence and affinity still execute their own complete native model. No prediction, coordinate, token activation, attention output, or value from another request is cached.

The transformation computes the original atom-conditioning AdaLN scale/bias, condition-only output gates, and the request-constant SingleConditioning prefix once per actual sampling multiplicity. The native single-time branch, attention, residuals, nonlinear transitions, all coordinate-dependent operations and original sampler run unchanged. Tensor repetition and views retain the original shapes and operation order.

Compared with the first guarded prototype, this version keeps the prepared child modules installed for consecutive calls at the same multiplicity. It restores them before any fallback, before a multiplicity switch, and in request cleanup. It performs resolved model binding/version/configuration checks at request boundaries, rather than rebuilding the full producer inventory at all 200 score calls. The score controller still verifies each static input's identity, shape, stride, dtype, device, data pointer and tracked version on each call, along with no-grad/autocast eligibility. Internal prepared forwards rely on the exact pinned source callsites for their repeated layouts; their metadata checks are not repeated at every node.

This is an explicit narrower execution contract, not a claim of detecting arbitrary concurrent mutation. The caller must grant exclusive use of frozen eval modules and immutable conditioning for the context, with unchanged source methods. Full model bindings/state must also stay unchanged between requests in the outer context. Detected changes reject before a normal request return. Unsafe `.data` writes that bypass tensor versioning, inference-tensor writes without version counters, mutation temporarily reversed before exit, or monkeypatched class methods/globals violate that contract. Training, arbitrary custom subclasses, hooks, compiled wrappers and concurrent/reentrant execution are refused or unsupported as documented in the code.

Exact affected native class checks and SHA256 checks pin three Boltz source files at commit `b1ebfc46ecf57f5414e0d1a6f9027bbb122c53bc`. Optional Boltz/einops imports are lazy. The public immutable-request keyword has no default, so enabling it is explicit.

`test_fast_scope.py` is a routing/cleanup fixture, not a pretrained model gate. It checks wrapper persistence, multiplicity reuse, restore-before-fallback, input mutation rejection, parameter/storage and sampler-configuration boundary rejection, exception cleanup, between-request edits and explicit contract invocation. Independent native CPU/MPS output gates and timings must use the exact candidate file hash before this variant is promoted. Earlier numerical/timing reports for `request_constants.py` do not by themselves validate this variant.
