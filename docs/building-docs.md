# Building and reading the documentation

A local Git commit records your changes. **Pushing a commit to GitHub triggers the Documentation workflow.** Pull requests and a manual run from the Actions tab also trigger it. Committing without pushing does not run GitHub Actions.

The workflow installs the pinned documentation tools, checks the source-link handling, and runs `mkdocs build --strict`. Missing documentation pages, local files and anchor links fail the build. It does not load a model, execute tutorial examples or download checkpoints.

## Build locally

From the repository root, using Python 3.12 (the same version as the documentation workflow):

```sh
python3.12 -m venv .venv-docs
. .venv-docs/bin/activate
python -m pip install --constraint requirements/docs.txt '.[docs]'
python -m unittest discover -s tests -p test_docs_hooks.py
python -m mkdocs build --strict
python -m mkdocs serve
```

Open the local address printed by MkDocs. The `docs` extra is independent of `torch`, `molecules`, `state` and `boltz`. The theme is Material for MkDocs 9.7.7, with MkDocs 1.6.1 and PyMdown Extensions 10.16.1. The complete documentation dependency versions are pinned in `requirements/docs.txt`; update them deliberately and rerun the strict build.

## Read a pushed build

1. Open the private repository's **Actions** tab and choose a successful **Documentation** run.
2. Download the artifact named `compressme-docs-<commit>` and extract it.
3. Open `index.html`, or serve the extracted directory with `python -m http.server 8000` and visit `http://127.0.0.1:8000`.

Pages use file-style URLs so ordinary navigation also works directly from the extracted files. Search works best through a local HTTP server.

Equations render with bundled KaTeX 0.16.22 scripts, styles and fonts. The extracted site needs no external math CDN. Its [license](assets/katex/LICENSE), [download provenance and file hashes](assets/katex/PROVENANCE.json) are included; the build tests verify those bytes. This follows the [KaTeX self-hosting approach](https://katex.org/docs/browser.html).

This is an HTML download, not a hosted website. The workflow does not enable GitHub Pages or deploy anything publicly. For a private repository, artifact downloads require a signed-in account with repository read access. The artifact retention is set to 14 days and is also subject to repository policy. See [GitHub's artifact access documentation](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/download-workflow-artifacts) and [artifact storage guide](https://docs.github.com/en/actions/tutorials/store-and-share-data).

## Source and historical evidence links

Documentation-to-documentation links stay inside the generated site. Links to checked-in source, scripts and historical benchmark reports point to the private repository. In Actions they use the build's commit; local builds default to `main`. You need repository access to follow them.

Three small legacy references originally lived beside excluded vendor or model artifact directories. Their license/provenance text and artifact README are copied into [bounded documentation attachments](assets/evidence/provenance.json); their original byte hashes are recorded there. No model tensors are copied into the site. Missing repository references fail the build instead of becoming guessed links.

The Markdown build hook only resolves links and verifies files. It does not import compressme or any architecture. A successful documentation build verifies documentation structure, not numerical model correctness. Use the [backend validation guide](gpu-validation.md) for model checks.

## Workflow pins

MkDocs is pinned to 1.6.1 and [Material for MkDocs to 9.7.7](https://github.com/squidfunk/mkdocs-material/releases/tag/9.7.7). The GitHub actions use immutable commit references verified against their official releases: [checkout v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1), [setup-python v7.0.0](https://github.com/actions/setup-python/releases/tag/v7.0.0), and [upload-artifact v7.0.1](https://github.com/actions/upload-artifact/releases/tag/v7.0.1). [MkDocs strict validation](https://www.mkdocs.org/user-guide/configuration/#validation) explains the local link checks.
