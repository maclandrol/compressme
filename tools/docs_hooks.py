"""Keep real documentation links local and point repository evidence at GitHub.

This hook only parses Markdown links and reads small attachment metadata. It
does not import the package, run examples, fetch URLs or open checkpoint data.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from markdown.extensions import Extension
from markdown.treeprocessors import Treeprocessor
from mkdocs.exceptions import PluginError


class RepositoryLinks:
    def __init__(self, root, docs, repo_url, revision="main", attachments=None):
        self.root = Path(root).resolve()
        self.docs = Path(docs).resolve()
        self.repo_url = repo_url.rstrip("/")
        if revision != "main" and not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise PluginError("COMPRESSME_DOCS_REVISION must be main or a full commit SHA")
        self.revision = revision
        self.attachments = attachments or {}
        self.page = None

    def rewrite(self, value):
        parsed = urlsplit(value)
        if parsed.scheme or parsed.netloc or not parsed.path:
            return value
        target = (self.page.parent / unquote(parsed.path)).resolve()
        if target.is_relative_to(self.docs):
            # MkDocs checks these paths and anchors and emits the right HTML URL.
            # Historical ../docs/foo.md paths must also be made relative to the
            # current Markdown page, rather than relying on a filesystem escape.
            path = Path(os.path.relpath(target, self.page.parent)).as_posix()
            return urlunsplit(("", "", path, parsed.query, parsed.fragment))
        if not target.is_relative_to(self.root):
            raise PluginError(f"Documentation link escapes the repository: {value}")
        relative = target.relative_to(self.root).as_posix()
        if relative in self.attachments:
            attachment = self.docs / self.attachments[relative]
            path = Path(os.path.relpath(attachment, self.page.parent)).as_posix()
            return urlunsplit(("", "", path, parsed.query, parsed.fragment))
        if any(part.startswith(".venv") or part in {
            "vendor", "artifacts", "checkpoints", "downloads", "reproduction", "reproduction-smoke", "site"
        } for part in target.relative_to(self.root).parts):
            raise PluginError(f"Link references excluded local data without a bounded attachment: {value}")
        if not target.exists():
            raise PluginError(f"Missing repository reference from {self.page.name}: {value}")
        kind = "tree" if target.is_dir() else "blob"
        path = f"{self.repo_url}/{kind}/{self.revision}/{quote(relative, safe='/')}"
        if parsed.query:
            path += "?" + parsed.query
        if parsed.fragment:
            path += "#" + parsed.fragment
        return path


class _LinkTreeprocessor(Treeprocessor):
    def __init__(self, context):
        super().__init__()
        self.context = context

    def run(self, root):
        for element in root.iter():
            attribute = "href" if element.tag == "a" else "src" if element.tag == "img" else None
            if attribute and element.get(attribute):
                element.set(attribute, self.context.rewrite(element.get(attribute)))


class _LinksExtension(Extension):
    def __init__(self, context):
        self.context = context
        super().__init__()

    def extendMarkdown(self, md):
        # Inline links are parsed at priority 20; MkDocs resolves paths at 0.
        # Work on real link nodes, so inline/fenced code is never rewritten.
        md.treeprocessors.register(_LinkTreeprocessor(self.context), "repository_links", 10)


_context = None


def on_config(config):
    global _context
    root = Path(config.config_file_path).resolve().parent
    docs = Path(config["docs_dir"]).resolve()
    manifest = json.loads((docs / "assets/evidence/provenance.json").read_text())
    attachments = {}
    for item in manifest["files"]:
        target = (docs / "assets/evidence" / item["attachment"]).resolve()
        if not target.is_relative_to(docs / "assets/evidence"):
            raise PluginError("Attachment path escapes its documentation directory")
        if not target.is_file() or target.stat().st_size > 65536:
            raise PluginError(f"Missing or oversized documentation attachment: {target.name}")
        if hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]:
            raise PluginError(f"Documentation attachment hash mismatch: {target.name}")
        attachments[item["source_repository_path"]] = target.relative_to(docs).as_posix()
    _context = RepositoryLinks(root, docs, config["repo_url"],
                               os.environ.get("COMPRESSME_DOCS_REVISION", "main"), attachments)
    config["edit_uri"] = f"blob/{_context.revision}/docs/"
    config["markdown_extensions"].append(_LinksExtension(_context))
    return config


def on_page_markdown(markdown, *, page, config, files):
    _context.page = Path(page.file.abs_src_path).resolve()
    return markdown
