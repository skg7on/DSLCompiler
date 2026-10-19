"""Structured errors shared by Python frontend entry points."""


class LLKError(Exception):
    """A diagnostic with a stable machine-readable code."""

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class FrontendSyntaxError(LLKError):
    """The source cannot be represented by the frontend."""


class UnsupportedFeatureError(LLKError):
    """The requested frontend or execution capability is unavailable."""


class TensorContractError(LLKError):
    """An argument does not satisfy the tensor contract."""


class CompilationError(LLKError):
    """Native compilation failed."""


class ExecutionError(LLKError):
    """Native invocation failed."""
