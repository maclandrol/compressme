# Working on compressme

Use a separate environment for the library and each biology-model stack.
For generic library tests and documentation:

```sh
python -m pip install -e '.[test,torch,packing,hub,docs]'
python -m pytest -q tests
python -m mkdocs build --strict
```

Upstream-dependent tests skip when their optional dependency or pinned source
is absent. After the Mol-JEPA tutorial download, set
`COMPRESSME_MOLJEPA_SOURCE=work/moljepa-original/architecture` to enable its
source-dependent featurizer checks in a compatible model environment. Real
checkpoint comparisons are separate from these generic tests.

Keep downloads, reconstructed artifacts and fresh benchmark output under
`work/`. Virtual environments, weights, generated tensors, logs and built HTML
are ignored. Do not commit credentials or downloaded model archives. The
repository contains source, documentation, reproducibility helpers and recorded
research evidence; the installable package excludes the research directories.

State the input/output contract for every rewrite. Report parameter count,
unique stored bytes, temporary memory and measured runtime separately. Preserve
the original dtype and every requested output. A failed full-output check is a
rejection, even if an isolated operator check passes.

Pushing a commit builds the documentation as a private Actions artifact. It
does not execute the multi-gigabyte model tutorials or establish GPU correctness.
