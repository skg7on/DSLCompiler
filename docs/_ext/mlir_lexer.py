"""Lightweight highlighting for extensible MLIR assembly, not syntax validation."""

from pygments.lexer import RegexLexer
from pygments.token import Comment, Keyword, Name, Number, Punctuation, String, Text


class MLIRLexer(RegexLexer):
    name = "MLIR"
    aliases = ["mlir"]
    filenames = ["*.mlir"]
    tokens = {
        "root": [
            (r"//[^\n]*", Comment.Single),
            (r"/\*.*?\*/", Comment.Multiline),
            (r'"([^"\\]|\\.)*"', String),
            (r"[%@][\w.$-]+", Name.Variable),
            (r"[!#][\w.$-]+", Name.Class),
            (r"\b(?:return|module|attributes|ins|outs|iter_args|to|step)\b", Keyword),
            (r"\b(?:tensor|memref|vector|index|bf16|tf32|[if]\d+)\b", Keyword.Type),
            (r"[a-zA-Z_][\w]*\.[\w.]+", Name.Builtin),
            (r"\b\d+(?:\.\d+)?\b", Number),
            (r"[{}()\[\],:<> =+-]", Punctuation),
            (r"\s+|.", Text),
        ],
    }
