"""Exercise repository-link validation through Sphinx's incremental builder."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class SourceLinksTest(unittest.TestCase):
    def test_deleted_repository_target_invalidates_cached_page(self):
        extension_dir = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="manual source links ") as temporary:
            root = Path(temporary)
            docs = root / "docs"
            docs.mkdir()
            (docs / "conf.py").write_text(
                "import sys\n"
                f"sys.path.insert(0, {str(extension_dir)!r})\n"
                "extensions = ['source_links']\n"
                "root_doc = 'index'\n"
            )
            target = root / "implementation.cpp"
            target.write_text("// source fixture\n")
            (docs / "index.rst").write_text(
                "Manual\n======\n\n"
                ":source:`Implementation <implementation.cpp>`\n"
            )
            command = [
                sys.executable, "-m", "sphinx", "-q", "-b", "html",
                "-n", "-W", "--keep-going", str(docs), str(root / "html"),
            ]
            initial = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(0, initial.returncode, initial.stderr)
            self.assertIn(
                "https://github.com/skg7on/DSLCompiler/blob/main/implementation.cpp",
                (root / "html/index.html").read_text(),
            )
            target.unlink()
            incremental = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(0, incremental.returncode)
            self.assertIn("repository link target does not exist", incremental.stderr)


if __name__ == "__main__":
    unittest.main()
