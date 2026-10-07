"""Checked repository links for files kept outside the Sphinx manual."""

from pathlib import Path
from urllib.parse import quote

from docutils import nodes
from sphinx.util.docutils import ReferenceRole


class SourceRole(ReferenceRole):
    def run(self):
        root = Path(self.env.srcdir).resolve().parents[2]
        path, _, anchor = self.target.partition("#")
        target = (root / path).resolve()
        # These targets live outside the RST tree. An incremental build must
        # recheck the page when a repository file changes or disappears.
        self.env.note_dependency(str(target))
        if not target.is_relative_to(root) or not target.exists():
            message = self.inliner.reporter.error(
                f"repository link target does not exist: {self.target}",
                line=self.lineno,
            )
            return [self.inliner.problematic(self.rawtext, self.rawtext, message)], [message]
        kind = "tree" if target.is_dir() else "blob"
        url = f"https://github.com/skg7on/DSLCompiler/{kind}/main/{quote(path)}"
        if anchor:
            url += "#" + quote(anchor)
        return [nodes.reference(self.rawtext, self.title, refuri=url)], []


def setup(app):
    app.add_role("source", SourceRole())
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
