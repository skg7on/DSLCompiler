"""Qualified LLK CPU compiler backend for upstream Triton 3.6.

The upstream ``@triton.jit`` runtime owns specialization, the TTIR stage, the
launch grid and the launcher protocol.  This backend only substitutes the
machine-code stages: its final bytes are a versioned, *unbound* launch template
containing the genuine TTIR plus the identities needed to bind it later.  No
machine code is claimed, and the template never carries runtime tensor shapes
or a runtime grid.

The Python frontend never imports this module unless upstream Triton is
requested; :mod:`llk.triton` exposes it lazily.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass

from llk._errors import CompilationError, UnsupportedFeatureError
from llk.runtime import _buffers
from llk.triton._toolchain import ensure_triton_importable

# The backend subclasses genuine upstream classes; importing this module always
# requires the qualified package, selected explicitly by the caller.
ensure_triton_importable()

BACKEND_NAME = "llk_cpu"
BACKEND_VERSION = "1"
ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_KIND = "llk-unbound-triton-template"
BINARY_EXT = "llktpl"
MAX_ARTIFACT_BYTES = 16 << 20
MAX_THREADS = 1024
MAX_SHARED_BYTES = 1 << 30

_FIELDS = (
    "schema_version",
    "kind",
    "entry",
    "target",
    "upstream_version",
    "backend_version",
    "compile_hash",
    "ttir",
    "ttir_hash",
)


def _artifact_failure(message, *, stage="load"):
    raise CompilationError(message, code="triton_artifact_mismatch", stage=stage)


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class UnboundLaunchArtifact:
    """The compiler's final bytes: a versioned unbound launch template.

    ``ttir`` is genuine upstream TTIR produced by the qualified pass pipeline.
    ``target`` records the LLK backend identity, never a Triton GPU target.
    """

    schema_version: int
    kind: str
    entry: str
    target: tuple[str, str, int]
    upstream_version: str
    backend_version: str
    compile_hash: str
    ttir: str
    ttir_hash: str

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != (
            ARTIFACT_SCHEMA_VERSION
        ):
            _artifact_failure("unsupported LLK Triton artifact schema version")
        if self.kind != ARTIFACT_KIND:
            _artifact_failure(f"unexpected artifact kind {self.kind!r}")
        if not isinstance(self.entry, str) or not self.entry:
            _artifact_failure("the artifact entry must be a nonempty string")
        target = self.target
        if (
            not isinstance(target, (tuple, list))
            or len(target) != 3
            or target[0] != BACKEND_NAME
            or not isinstance(target[1], (str, int))
            or type(target[2]) is not int
            or target[2] < 1
        ):
            _artifact_failure("the artifact target is not an llk_cpu target")
        object.__setattr__(
            self, "target", (str(target[0]), str(target[1]), int(target[2]))
        )
        for name in ("upstream_version", "backend_version", "compile_hash"):
            value = getattr(self, name)
            if not isinstance(value, str):
                _artifact_failure(f"the artifact {name} must be a string")
        if self.backend_version != BACKEND_VERSION:
            _artifact_failure(
                f"the artifact was produced by LLK backend {self.backend_version}; "
                f"this runtime implements {BACKEND_VERSION}"
            )
        if not isinstance(self.ttir, str) or not self.ttir:
            _artifact_failure("the artifact carries no TTIR text")
        if len(self.ttir.encode("utf-8")) > MAX_ARTIFACT_BYTES:
            _artifact_failure("the artifact TTIR exceeds the size limit")
        if self.ttir_hash != _digest(self.ttir):
            _artifact_failure("the artifact TTIR hash does not match its text")

    def identity(self) -> str:
        return hashlib.sha256(_canonical(self._body())).hexdigest()

    def _body(self):
        return {name: getattr(self, name) for name in _FIELDS}

    def metadata(self):
        """Honest artifact metadata for inspection; never a machine-code claim."""
        return {
            "kind": ARTIFACT_KIND,
            "artifact_schema_version": self.schema_version,
            "machine_code": False,
            "entry": self.entry,
            "target": list(self.target),
            "upstream_version": self.upstream_version,
            "backend_version": self.backend_version,
            "ttir_hash": self.ttir_hash,
        }

    def to_payload(self):
        body = self._body()
        body["target"] = list(self.target)
        body["artifact_identity"] = self.identity()
        return json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")

    @classmethod
    def from_payload(cls, data: bytes):
        if not isinstance(data, (bytes, bytearray)):
            _artifact_failure("an artifact must be bytes")
        if len(data) > MAX_ARTIFACT_BYTES:
            _artifact_failure("the artifact exceeds the size limit")
        try:
            text = bytes(data).decode("utf-8")
            value = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            _artifact_failure(f"the artifact is not valid UTF-8 JSON: {error}")
        if not isinstance(value, dict):
            _artifact_failure("the artifact must be a JSON object")
        identity = value.pop("artifact_identity", None)
        if set(value) != set(_FIELDS):
            _artifact_failure(
                "the artifact fields do not match the versioned schema"
            )
        try:
            artifact = cls(**value)
        except TypeError as error:
            _artifact_failure(f"the artifact is malformed: {error}")
        if identity != artifact.identity():
            _artifact_failure("the artifact content identity does not match")
        return artifact


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    """The retained template behind one loaded ``CompiledKernel``."""

    artifact: UnboundLaunchArtifact
    binary: bytes

    @property
    def metadata(self):
        return self.artifact.metadata()

    @property
    def template(self):
        return self.artifact

    def __repr__(self):
        return f"<LoadedArtifact entry={self.artifact.entry} kind={ARTIFACT_KIND}>"


class CPUUtils:
    """Driver utilities: retain the template, never load machine code."""

    def load_binary(self, name, kernel, shared, device):
        if not isinstance(name, str) or not name:
            _artifact_failure("a loaded artifact needs a kernel name")
        if type(shared) is not int or shared < 0:
            _artifact_failure("the artifact shared-memory extent is invalid")
        artifact = UnboundLaunchArtifact.from_payload(kernel)
        if artifact.entry != name:
            _artifact_failure(
                f"the artifact entry {artifact.entry!r} does not match {name!r}"
            )
        loaded = LoadedArtifact(artifact, bytes(kernel))
        # Upstream calls the second result ``function`` and hands it to the
        # launcher; the retained template is the only handle it needs.
        return loaded, loaded, 0, 0, MAX_THREADS

    def unload_module(self, module):
        # The template is plain Python data; dropping the reference is enough.
        return None

    def get_device_properties(self, device):
        return {
            "name": "llk-cpu",
            "warp_size": 1,
            "max_shared_mem": MAX_SHARED_BYTES,
            "max_threads_per_block": MAX_THREADS,
            "max_num_regs": 0,
            "multiprocessor_count": 1,
        }


@dataclass(frozen=True)
class LLKCPUOptions:
    """Backend options consumed by upstream Triton's JIT and cache keys."""

    num_warps: int = 1
    num_ctas: int = 1
    num_stages: int = 1
    warp_size: int = 1
    maxnreg: int | None = None
    ir_override: str | None = None
    enable_fp_fusion: bool = False
    launch_cooperative_grid: bool = False
    extern_libs: tuple = ()
    debug: bool = False
    backend_name: str = BACKEND_NAME
    sanitize_overflow: bool = False
    arch: str = "native"
    instrumentation_mode: str = ""
    allowed_dot_input_precisions: tuple = ("ieee",)
    default_dot_input_precision: str = "ieee"
    deprecated_fp8_dot_operand_dtypes: tuple = ()
    max_num_imprecise_acc_default: int = 0

    def __post_init__(self):
        if type(self.num_warps) is not int or self.num_warps != 1:
            raise ValueError("the LLK CPU backend compiles one upstream program")
        if type(self.warp_size) is not int or self.warp_size < 1:
            raise ValueError("warp_size must be a positive integer")
        if self.backend_name != BACKEND_NAME:
            raise ValueError(f"unsupported backend name {self.backend_name!r}")
        object.__setattr__(self, "extern_libs", tuple(self.extern_libs or ()))
        object.__setattr__(
            self,
            "allowed_dot_input_precisions",
            tuple(self.allowed_dot_input_precisions),
        )
        object.__setattr__(
            self,
            "deprecated_fp8_dot_operand_dtypes",
            tuple(self.deprecated_fp8_dot_operand_dtypes),
        )

    def hash(self):
        fields = {
            field.name: getattr(self, field.name)
            for field in dataclasses.fields(self)
        }
        return hashlib.sha256(_canonical(fields)).hexdigest()


_TRITON_OPTION_FIELDS = frozenset(
    field.name for field in dataclasses.fields(LLKCPUOptions)
)
_IGNORED_TRITON_OPTIONS = frozenset({"instrumentation_mode"})


def _unsupported_custom_type(*args, **kwargs):
    raise UnsupportedFeatureError(
        "custom or reduced-precision Triton types are outside the LLK CPU "
        "launch contract",
        code="unsupported_triton_launch",
    )


def _min_dot_size(*args, **kwargs):
    # The upstream backend interface requires the entry; the accepted launch
    # corpus does not contain tl.dot.
    return (1, 1, 1)


def _make_ttir(module):
    """Run the qualified upstream TTIR pipeline on the parsed module."""
    from triton._C.libtriton import ir, passes

    manager = ir.pass_manager(module.context)
    manager.enable_debug()
    passes.common.add_inliner(manager)
    passes.ttir.add_rewrite_tensor_descriptor_to_pointer(manager)
    passes.common.add_canonicalizer(manager)
    passes.ttir.add_combine(manager)
    passes.ttir.add_reorder_broadcast(manager)
    passes.common.add_cse(manager)
    passes.common.add_symbol_dce(manager)
    passes.ttir.add_loop_unroll(manager)
    manager.run(module, "make_ttir")
    return module


def _entry_name(module):
    try:
        return module.get_entry_func_name()
    except AttributeError:  # pragma: no cover - qualified binding provides it
        raise UnsupportedFeatureError(
            "the qualified Triton bindings do not expose the entry function",
            code="upstream_toolchain_unqualified",
        )


def _make_artifact(ttir, metadata, target):
    if not isinstance(ttir, str) or not ttir:
        raise UnsupportedFeatureError(
            "the TTIR stage produced no text", code="triton_compilation_failed"
        )
    entry = metadata.get("name")
    if not isinstance(entry, str) or not entry:
        raise UnsupportedFeatureError(
            "the TTIR stage did not record the entry function",
            code="triton_compilation_failed",
        )
    artifact = UnboundLaunchArtifact(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        kind=ARTIFACT_KIND,
        entry=entry,
        target=(target.backend, str(target.arch), int(target.warp_size)),
        upstream_version=str(metadata.get("triton_version", "")),
        backend_version=BACKEND_VERSION,
        compile_hash=str(metadata.get("hash", "")),
        ttir=ttir,
        ttir_hash=_digest(ttir),
    )
    return artifact.to_payload()


def _make_backend_class():
    from triton.backends.compiler import BaseBackend, Language

    class LLKCPUBackend(BaseBackend):
        """Upstream compiler backend whose final bytes are an unbound template."""

        BINARY_EXT = BINARY_EXT

        def __init__(self, target):
            super().__init__(target)
            self.binary_ext = BINARY_EXT

        @staticmethod
        def supports_target(target):
            return target.backend == BACKEND_NAME

        @staticmethod
        def get_tensor_specialization(arg, **kwargs):
            # Torch tensors expose ``data_ptr``; buffer-protocol providers do
            # not.  Deriving the same alignment fact the launcher will check
            # keeps the first-stage specialization honest for both.
            try:
                address = _buffers.layout_of(arg).address
            except Exception:
                return ""
            return "D" if address % 16 == 0 and kwargs.get("align", False) else ""

        def hash(self):
            return (
                f"{BACKEND_NAME}-backend-{BACKEND_VERSION}"
                f"-artifact-{ARTIFACT_SCHEMA_VERSION}"
            )

        def parse_options(self, opts):
            unknown = set(opts) - _TRITON_OPTION_FIELDS - _IGNORED_TRITON_OPTIONS
            if unknown:
                raise ValueError(
                    f"unsupported LLK Triton backend options {sorted(unknown)}"
                )
            values = {
                name: opts[name]
                for name in _TRITON_OPTION_FIELDS
                if name in opts and opts[name] is not None
            }
            values.setdefault("arch", str(self.target.arch))
            values.setdefault("warp_size", int(self.target.warp_size))
            return LLKCPUOptions(**values)

        def pack_metadata(self, metadata):
            return (metadata.name, metadata.hash, metadata.target.backend)

        def get_codegen_implementation(self, options):
            return {
                "convert_custom_types": _unsupported_custom_type,
                "min_dot_size": _min_dot_size,
            }

        def get_module_map(self):
            return {}

        def load_dialects(self, context):
            # ``compile`` loads the upstream dialects before this hook; the
            # LLK CPU target adds no dialect of its own.
            return None

        def add_stages(self, stages, options, language):
            if language is not Language.TRITON:
                raise UnsupportedFeatureError(
                    "the LLK CPU backend accepts only Triton source",
                    code="unsupported_triton_launch",
                )
            target = self.target

            def ttir_stage(module, metadata):
                module = _make_ttir(module)
                metadata["name"] = _entry_name(module)
                metadata["shared"] = 0
                return module.str_nodebug()

            def artifact_stage(ttir, metadata):
                return _make_artifact(ttir, metadata, target)

            stages["ttir"] = ttir_stage
            stages[BINARY_EXT] = artifact_stage

    return LLKCPUBackend


LLKCPUBackend = _make_backend_class()
