#!/usr/bin/env python3
"""Check that the workflow and acceptance documents reference real things.

Two failure modes this guards against, both cheap to introduce and expensive to
notice:

  * a path or link that no longer exists -- a moved design doc, a renamed
    machine profile, a fixture that was deleted;
  * a documented flag that the tool does not have -- an example that reads as
    runnable but is not.

It does *not* try to prove an example is *useful* or that a described workflow
is complete; that is the reader's and the reviewer's job. What it proves is that
every repo path a reader is told to open exists, and every flag a reader is told
to pass is one the tool accepts.

Usage:
  check_doc_references.py <source_dir> <llk_opt> <llk_compile> <llk_tune> <micro_perf> [<llk_bench>]

If llk_bench is omitted, it is located beside llk_opt.
"""

import os
import re
import subprocess
import sys


# The documents whose commands and links a reader is expected to be able to
# follow literally.
DOCS = [
    "README.md",
    "docs/index.rst",
    "docs/getting-started.rst",
    "docs/concepts.rst",
    "docs/architecture.rst",
    "docs/features.rst",
    "docs/contributing.rst",
    "docs/building-docs.rst",
    "docs/examples/index.rst",
    "docs/tools/index.rst",
    "docs/tools/compiler.rst",
    "docs/tools/optimizer.rst",
    "docs/tools/performance.rst",
    "docs/tools/tuning.rst",
    "docs/tools/benchmark.rst",
    "docs/tutorials/index.rst",
    "docs/tutorials/01-first-kernel.rst",
    "docs/tutorials/02-mapping.rst",
    "docs/tutorials/03-performance.rst",
    "docs/tutorials/04-tuning.rst",
    "docs/design/micro-ir-mapping-workflow.md",
    "docs/reviews/issue67-final-acceptance.md",
]

# The tools whose documented flags are checked, keyed by the basename a command
# line starts with.
TOOLS = ("llk-opt", "llk-compile", "llk-tune", "micro-perf", "llk-bench")

# Repo-relative roots. A backticked token that starts with one of these is a
# path a reader will try to open, so it must exist. Bare filenames without a
# slash (`input.mlir`, `swiglu.micro.mlir`) are illustrative placeholders and
# are deliberately not checked.
ROOTS = (
    "ARCHITECTURE.md", "CLAUDE.md", "README.md", "CMakeLists.txt",
    "docs/", "machines/", "mapping/", "test/", "include/", "lib/",
    "runtime/", "tools/", "schedules/", "benchmark/",
)

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
CODE_RE = re.compile(r"(?<!`)(`{1,2})([^`\n]+)\1(?!`)")
ROLE_RE = re.compile(r":([A-Za-z][A-Za-z0-9_-]*):`([^`\n]+)`")
IMAGE_RE = re.compile(r"^\s*\.\.\s+(?:image|figure)::\s*(\S+)", re.MULTILINE)
FENCE_RE = re.compile(r"^\s*```(\w*)")
BLOCK_RE = re.compile(r"^(\s*)\.\.\s+(?:code-block|code)::\s*(\w+)\s*$")
FLAG_RE = re.compile(r"(?<![\w-])-{1,2}[A-Za-z][A-Za-z0-9_-]*")
SHELL_LANGUAGES = ("bash", "sh", "shell")


def read(path):
    with open(path, "r") as handle:
        return handle.read()


def is_repo_path(token):
    if not any(token.startswith(root) for root in ROOTS):
        return False
    # Real paths only: no globs, no angle placeholders, no anchors, no URLs.
    if any(ch in token for ch in "*<>#?") or "://" in token:
        return False
    return True


def check_paths(source_dir, doc, text, problems):
    """Check Markdown links, Sphinx file roles/images, and literal repo paths.

    Sphinx validates :ref: labels and toctree references during the HTML build.
    Absolute :doc:, :download:, and image paths are relative to its docs root;
    :source: paths are relative to the repository root.
    """
    doc_dir = os.path.dirname(os.path.join(source_dir, doc))
    docs_dir = os.path.join(source_dir, "docs")

    def check_target(target, base, description, document=False):
        target = target.split("#", 1)[0].strip()
        if not target or "://" in target or target.startswith("mailto:"):
            return
        if document and not target.endswith(".rst"):
            target += ".rst"
        resolved = os.path.normpath(os.path.join(base, target))
        exists = os.path.isfile(resolved) if document else os.path.exists(resolved)
        if not exists:
            problems.append("%s: %s does not exist: %s" % (doc, description, target))

    for target in LINK_RE.findall(text):
        check_target(target, doc_dir, "link target")

    for role, contents in ROLE_RE.findall(text):
        if role not in ("doc", "download", "source"):
            continue
        explicit = re.search(r"<([^<>]+)>\s*$", contents)
        target = explicit.group(1).strip() if explicit else contents.strip()
        if role == "source":
            base = source_dir
        else:
            base = docs_dir if target.startswith("/") else doc_dir
        check_target(target.lstrip("/"), base, "%s target" % role,
                     document=role == "doc")

    for target in IMAGE_RE.findall(text):
        base = docs_dir if target.startswith("/") else doc_dir
        check_target(target.lstrip("/"), base, "image target")

    # Roles have their own resolution rules; their contents are not literals.
    for _, token in CODE_RE.findall(ROLE_RE.sub("", text)):
        token = token.strip()
        if is_repo_path(token) and not os.path.exists(
                os.path.join(source_dir, token)):
            problems.append("%s: referenced path does not exist: %s"
                            % (doc, token))


def shell_blocks(text):
    """Yield shell block contents from Markdown fences and RST directives."""
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        fence = FENCE_RE.match(lines[index])
        if fence:
            language = fence.group(1)
            index += 1
            block = []
            while index < len(lines) and not FENCE_RE.match(lines[index]):
                block.append(lines[index])
                index += 1
            index += 1
            if language in SHELL_LANGUAGES:
                yield block
            continue

        directive = BLOCK_RE.match(lines[index])
        if not directive:
            index += 1
            continue
        indent = len(directive.group(1).expandtabs())
        language = directive.group(2)
        index += 1
        block = []
        body_started = False
        while index < len(lines):
            raw = lines[index].expandtabs()
            if raw.strip():
                if len(raw) - len(raw.lstrip()) <= indent:
                    break
                if not body_started and raw.strip().startswith(":"):
                    index += 1
                    continue
                body_started = True
            if body_started:
                block.append(raw)
            index += 1
        if language in SHELL_LANGUAGES:
            yield block


def shell_commands(text):
    """Join shell continuations without crossing a code block boundary."""
    for block in shell_blocks(text):
        logical = ""
        for raw in block:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            logical = logical + " " + line if logical else line
            if logical.endswith("\\"):
                logical = logical[:-1].rstrip()
                continue
            yield logical
            logical = ""
        if logical:
            yield logical


def check_flags(source_dir, doc, text, tool_paths, problems):
    """Every option on a documented tool command must be one the tool has."""
    help_names = {}
    for command in shell_commands(text):
        words = command.replace("$ ", "").split()
        if not words:
            continue
        first = os.path.basename(words[0])
        if first not in TOOLS:
            continue
        if first not in help_names:
            path = tool_paths.get(first)
            help_names[first] = None
            if not path or not os.path.exists(path):
                problems.append("%s: tool not found for flag check: %s" % (doc, first))
                continue
            try:
                result = subprocess.run([path, "--help"], capture_output=True,
                                        text=True)
            except OSError as error:
                problems.append("%s: cannot read %s --help: %s" % (doc, first, error))
                continue
            if result.returncode:
                problems.append("%s: %s --help failed with status %d"
                                % (doc, first, result.returncode))
                continue
            # cl accepts either dash form for each option.
            help_names[first] = {
                flag.lstrip("-") for flag in
                FLAG_RE.findall(result.stdout + "\n" + result.stderr)}
        if help_names[first] is None:
            continue
        for flag in FLAG_RE.findall(command):
            name = flag.lstrip("-")
            if name not in help_names[first]:
                problems.append(
                    "%s: documented flag %s is not accepted by %s"
                    % (doc, flag, first))


def main():
    if len(sys.argv) not in (6, 7):
        print(__doc__, file=sys.stderr)
        return 2
    source_dir, llk_opt, llk_compile, llk_tune, micro_perf = sys.argv[1:6]
    tool_paths = {
        "llk-opt": llk_opt,
        "llk-compile": llk_compile,
        "llk-tune": llk_tune,
        "micro-perf": micro_perf,
        "llk-bench": sys.argv[6] if len(sys.argv) == 7 else
        os.path.join(os.path.dirname(llk_opt), "llk-bench"),
    }

    problems = []
    for doc in DOCS:
        path = os.path.join(source_dir, doc)
        if not os.path.exists(path):
            problems.append("missing document: %s" % doc)
            continue
        text = read(path)
        check_paths(source_dir, doc, text, problems)
        check_flags(source_dir, doc, text, tool_paths, problems)

    if problems:
        print("documentation references are stale:", file=sys.stderr)
        for problem in problems:
            print("  " + problem, file=sys.stderr)
        return 1

    print("doc references ok: %d document(s) checked" % len(DOCS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
