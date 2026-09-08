# Final status

The source-specific runtime was promoted as the optional
`compressme.boltz2_runtime` module, with explicit immutable request ownership.
Some earlier reports in this directory correctly said full pretrained gates
were pending when they were written. They are retained unchanged as history.

The final runtime SHA256 is
`25cedfc5a6769ff3893832a962c618a31edd5a9e1ed3c412e706885cf6ff1c02`.
Complete native CPU/MPS gates and final interleaved timings are now available in
`../boltz2-request-runtime/lean/README.md`. All 48 tensors passed byte comparisons
on both devices. MPS had no useful timing gain; CPU affinity showed a small
positive signal in three pairs. The runtime remains optional, not enabled by
the shared-weight loader. The generic FX discovery prototype is separate.
