"""Small documentation-only tests; no Torch or model code is imported."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import tempfile
import unittest

try:
    import markdown
    from mkdocs.exceptions import PluginError
except ImportError:
    raise unittest.SkipTest("Install the docs extra to check the documentation hook")

HOOK = Path(__file__).resolve().parents[1] / "tools/docs_hooks.py"
if not HOOK.exists():
    raise unittest.SkipTest("Documentation build tooling is supplied with the source checkout")
spec = importlib.util.spec_from_file_location("_compressme_docs_hook_test", HOOK)
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


class DocumentationLinks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.docs = self.root / "docs"
        self.docs.mkdir()
        (self.docs / "index.md").write_text("# Home")
        self.context = hook.RepositoryLinks(self.root, self.docs,
                                            "https://github.com/maclandrol/compressme", "a" * 40)
        self.context.page = self.docs / "index.md"

    def test_real_repository_links_use_pinned_private_source(self):
        (self.root / "example.py").write_text("raise RuntimeError('never execute')")
        result = self.context.rewrite("../example.py#L1")
        self.assertEqual(result, "https://github.com/maclandrol/compressme/blob/" + "a" * 40 + "/example.py#L1")

    def test_docs_links_and_external_urls_are_unchanged(self):
        for value in ("guide.md#setup", "figures/a.png", "https://example.com/a", "#title"):
            self.assertEqual(self.context.rewrite(value), value)

    def test_historical_docs_prefix_is_canonicalized(self):
        self.assertEqual(self.context.rewrite("../docs/guide.md#setup"), "guide.md#setup")

    def test_inline_and_fenced_code_are_not_treated_as_links(self):
        rendered = markdown.markdown("`[x](../missing.py)`\n\n```\n[x](../missing.py)\n```",
                                     extensions=["fenced_code", hook._LinksExtension(self.context)])
        self.assertNotIn("href=", rendered)
        self.assertIn("[x](../missing.py)", rendered)

    def test_missing_source_or_path_escape_fails(self):
        for value in ("../missing.py", "../../outside.md"):
            with self.assertRaises(PluginError):
                self.context.rewrite(value)

    def test_excluded_artifact_needs_explicit_attachment(self):
        with self.assertRaises(PluginError):
            self.context.rewrite("../artifacts/model/README.md")
        self.context.attachments["artifacts/model/README.md"] = "assets/model-readme.txt"
        self.assertEqual(self.context.rewrite("../artifacts/model/README.md"), "assets/model-readme.txt")

    def test_revision_cannot_change_the_url_structure(self):
        with self.assertRaises(PluginError):
            hook.RepositoryLinks(self.root, self.docs, "https://github.com/maclandrol/compressme", "../../oops")


class BundledMathAssets(unittest.TestCase):
    def test_all_vendored_assets_match_the_published_byte_manifest(self):
        root = HOOK.parents[1] / "docs/assets/katex"
        manifest = json.loads((root / "PROVENANCE.json").read_text())
        expected = {"PROVENANCE.json"}
        for item in manifest["files"]:
            path = (root / item["path"]).resolve()
            self.assertTrue(path.is_relative_to(root.resolve()))
            data = path.read_bytes()
            self.assertEqual(len(data), item["bytes"])
            self.assertEqual(hashlib.sha256(data).hexdigest(), item["sha256"])
            expected.add(item["path"])
        self.assertEqual({p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}, expected)

    def test_math_fonts_are_bundled_and_never_fetched_from_a_cdn(self):
        root = HOOK.parents[1] / "docs/assets/katex"
        references = re.findall(r"url\(([^)]+)\)", (root / "katex.min.css").read_text())
        self.assertGreater(len(references), 0)
        for value in references:
            value = value.strip("\"'")
            self.assertFalse(":" in value or value.startswith("//"))
            self.assertTrue((root / value).is_file(), value)


if __name__ == "__main__":
    unittest.main()
