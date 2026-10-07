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
  check_doc_references.py <source_dir> <llk_opt> <llk_compile> <llk_tune> <micro_perf>
"""

import os
import re
import subprocess
import sys


# The documents whose commands and links a reader is expected to be able to
# follow literally.
DOCS = [
    "docs/design/micro-ir-mapping-workflow.md",
    "docs/reviews/issue67-final-acceptance.md",
]

# The tools whose documented flags are checked, keyed by the basename a command
# line starts with.
TOOLS = ("llk-opt", "llk-compile", "llk-tune", "micro-perf")

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
CODE_RE = re.compile(r"`([^`\n]+)`")
FENCE_RE = re.compile(r"^\s*```(\w*)")
FLAG_RE = re.compile(r"--[A-Za-z][A-Za-z0-9-]*")


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
    """Every markdown link target and every backticked repo path must exist."""
    doc_dir = os.path.dirname(os.path.join(source_dir, doc))

    for target in LINK_RE.findall(text):
        target = target.split("#", 1)[0].strip()
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        resolved = os.path.normpath(os.path.join(doc_dir, target))
        if not os.path.exists(resolved):
            problems.append("%s: link target does not exist: %s" % (doc, target))

    for token in CODE_RE.findall(text):
        token = token.strip()
        if is_repo_path(token) and not os.path.exists(
                os.path.join(source_dir, token)):
            problems.append("%s: referenced path does not exist: %s"
                            % (doc, token))


def check_flags(source_dir, doc, text, tool_paths, problems):
    """Every `--flag` on a documented tool command must be one the tool has."""
    help_names = {}
    for tool in TOOLS:
        path = tool_paths.get(tool)
        if not path or not os.path.exists(path):
            problems.append("%s: tool not found for flag check: %s" % (doc, tool))
            continue
        result = subprocess.run([path, "--help"], capture_output=True, text=True)
        # Option names as `--flag` and `-f` spellings; a bare name matches
        # either dash form (cl accepts both for the same option).
        help_names[tool] = {
            name.lstrip("-") for name in
            re.findall(r"-{1,2}([A-Za-z][A-Za-z0-9_-]*)", result.stdout)}

    in_fence = False
    logical = ""
    for raw in text.splitlines():
        fence = FENCE_RE.match(raw)
        if fence:
            if not in_fence:
                in_fence = fence.group(1) in ("bash", "sh", "shell")
                logical = ""
            else:
                in_fence = False
            continue
        if not in_fence:
            continue

        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Join backslash continuations into one logical command.
        if logical:
            logical = logical + " " + line
        else:
            logical = line
        if logical.endswith("\\"):
            continue

        command = logical
        logical = ""
        words = command.replace("$ ", "").split()
        if not words:
            continue
        first = os.path.basename(words[0])
        if first not in help_names:
            continue
        for flag in FLAG_RE.findall(command):
            name = flag.lstrip("-")
            if name not in help_names[first]:
                problems.append(
                    "%s: documented flag %s is not accepted by %s"
                    % (doc, flag, first))


def main():
    if len(sys.argv) != 6:
        print(__doc__, file=sys.stderr)
        return 2
    source_dir, llk_opt, llk_compile, llk_tune, micro_perf = sys.argv[1:6]
    tool_paths = {
        "llk-opt": llk_opt,
        "llk-compile": llk_compile,
        "llk-tune": llk_tune,
        "micro-perf": micro_perf,
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
