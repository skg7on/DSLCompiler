#!/usr/bin/env python3
"""Isolated TTIR export worker (protocol version 1).

The worker is always launched as a separate interpreter. It imports the
*pinned upstream* Triton package, generates genuine TTIR through Triton's own
compiler stages, walks that typed module through Triton's own MLIR bindings, and
writes one framed JSON result to stdout. No MLIR context, module, type or C++
object leaves this process.

Only standard-library imports happen before ``--triton-python`` is applied, so
the qualified package is selected explicitly rather than through ambient
``PYTHONPATH`` state. Diagnostics never share the result stream.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PROTOCOL_VERSION = 1
MARKER = "LLK-TRITON-EXPORT-V1"
MAX_REQUEST_BYTES = 1 << 20
MAX_TTIR_BYTES = 4 << 20

ELEMENT_DTYPES = frozenset(
    {"i1", "i8", "i16", "i32", "i64", "index", "f16", "bf16", "f32", "f64"}
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")
_TENSOR_TYPE = re.compile(r"tensor<(.*)>")
_POINTER_TYPE = re.compile(r"!tt\.ptr<([a-z0-9]+)>")

CMPI_PREDICATES = {
    0: "eq",
    1: "ne",
    2: "slt",
    3: "sle",
    4: "sgt",
    5: "sge",
    6: "ult",
    7: "ule",
    8: "ugt",
    9: "uge",
}
CMPF_PREDICATES = {
    1: "oeq",
    2: "ogt",
    3: "oge",
    4: "olt",
    5: "ole",
    6: "one",
    7: "ord",
    8: "ueq",
    9: "ugt",
    10: "uge",
    11: "ult",
    12: "ule",
    13: "une",
    14: "uno",
    15: "true",
}

BINARY_INTS = frozenset(
    {
        "addi",
        "subi",
        "muli",
        "divsi",
        "divui",
        "remsi",
        "remui",
        "andi",
        "ori",
        "xori",
        "shli",
        "shrsi",
        "shrui",
    }
)
BINARY_FLOATS = frozenset({"addf", "subf", "mulf", "divf", "remf"})
CASTS = frozenset(
    {
        "extf",
        "truncf",
        "extsi",
        "extui",
        "trunci",
        "sitofp",
        "uitofp",
        "fptosi",
        "fptoui",
        "bitcast",
        "index_cast",
        "index_castui",
    }
)
LAYOUT_OPS = frozenset({"tt.trans", "tt.reshape", "tt.expand_dims", "tt.broadcast"})
REGION_OPS = frozenset({"scf.while", "scf.if", "scf.index_switch"})


class ExportError(Exception):
    """A structured worker failure that becomes a framed error result."""

    def __init__(self, code, message, *, stage="export", details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage
        self.details = details or {}


def _emit(payload):
    """Write exactly one framed result to the real stdout."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    stream = sys.__stdout__
    stream.write(f"{MARKER}\n{body}\n")
    stream.flush()


def _read_request(path):
    file = Path(path)
    try:
        size = file.stat().st_size
    except OSError as error:
        raise ExportError(
            "invalid_triton_request", f"cannot read export request: {error}"
        ) from error
    if size > MAX_REQUEST_BYTES:
        raise ExportError(
            "invalid_triton_request",
            f"export request exceeds {MAX_REQUEST_BYTES} bytes",
        )
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExportError(
            "invalid_triton_request", f"malformed export request JSON: {error}"
        ) from error
    if not isinstance(data, dict):
        raise ExportError("invalid_triton_request", "export request must be an object")
    if data.get("protocol_version") != PROTOCOL_VERSION:
        raise ExportError(
            "upstream_toolchain_unqualified",
            "the export request names an unsupported protocol version",
        )
    return data


def _read_lock(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExportError(
            "upstream_toolchain_unqualified",
            f"cannot read the toolchain lock: {error}",
        ) from error
    if not isinstance(data, dict):
        raise ExportError(
            "upstream_toolchain_unqualified", "the toolchain lock must be an object"
        )
    return data


def _import_triton(python_path):
    if python_path:
        candidate = Path(python_path)
        if not candidate.is_dir():
            raise ExportError(
                "upstream_toolchain_unqualified",
                f"Triton Python path {candidate} is not a directory",
            )
        sys.path.insert(0, str(candidate))
    try:
        import triton  # noqa: F401
    except Exception as error:  # ImportError or a native-library load failure
        raise ExportError(
            "upstream_toolchain_unqualified",
            f"cannot import the pinned upstream Triton package: {error}",
        ) from error
    return triton


def _source_revision(root):
    if not (root / ".git").exists():
        return "unknown"
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def _toolchain_identity(triton, target):
    version = str(getattr(triton, "__version__", "") or "")
    if not version:
        raise ExportError(
            "upstream_toolchain_unqualified",
            "the upstream Triton package does not report a version",
        )
    root = Path(triton.__file__).resolve().parent.parent.parent
    hash_file = root / "cmake" / "llvm-hash.txt"
    llvm_revision = "unknown"
    if hash_file.is_file():
        llvm_revision = hash_file.read_text(encoding="utf-8").strip() or "unknown"
    return {
        "triton_version": version,
        "llvm_revision": llvm_revision,
        "source_revision": _source_revision(root),
        "protocol_version": PROTOCOL_VERSION,
        "target": target,
        "python_tag": f"{platform.python_implementation().lower()}-"
        f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform_tag": platform.platform(),
    }


def _require_qualified(identity, lock):
    """Compare the observed binaries with the checked-in qualification lock."""
    reasons = []
    if identity["protocol_version"] != lock.get("protocol_version"):
        reasons.append("export protocol version changed")
    if identity["triton_version"] != lock.get("triton_version"):
        reasons.append(
            f"Triton {identity['triton_version']} is not the locked "
            f"{lock.get('triton_version')}"
        )
    if identity["target"] != lock.get("target"):
        reasons.append(
            f"target {identity['target']} is not the locked {lock.get('target')}"
        )
    for key in ("llvm_revision", "source_revision"):
        expected = lock.get(key)
        observed = identity[key]
        if not isinstance(expected, str) or expected in ("", "unknown"):
            reasons.append(f"the lock does not record {key}")
        elif observed in ("", "unknown"):
            reasons.append(f"cannot determine the upstream {key}")
        elif observed != expected:
            reasons.append(f"{key} {observed} is not the locked {expected}")
    if reasons:
        raise ExportError(
            "upstream_toolchain_unqualified",
            "; ".join(reasons),
            details={"observed": identity, "locked": dict(lock)},
        )


def _load_kernel(module_path, function):
    import triton

    path = Path(module_path)
    if not path.is_file():
        raise ExportError(
            "invalid_triton_request", f"kernel module {path} does not exist"
        )
    if not _IDENTIFIER.fullmatch(function):
        raise ExportError("invalid_triton_request", "function name must be an identifier")
    module_name = f"_llk_triton_kernel_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ExportError(
            "invalid_triton_request", f"cannot load kernel module {path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        raise ExportError(
            "triton_compilation_failed", f"kernel module failed to load: {error}"
        ) from error
    finally:
        sys.modules.pop(module_name, None)
    kernel = getattr(module, function, None)
    if kernel is None:
        raise ExportError(
            "invalid_triton_request", f"module does not define {function!r}"
        )
    if not hasattr(kernel, "arg_names") or not hasattr(kernel, "cache_key"):
        raise ExportError(
            "invalid_triton_request", f"{function!r} is not an upstream triton.jit kernel"
        )
    return kernel


def _signature(signature):
    if not isinstance(signature, dict) or not signature:
        raise ExportError(
            "invalid_triton_signature", "signature must be a nonempty object"
        )
    for name, spec in signature.items():
        if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
            raise ExportError(
                "invalid_triton_signature", "signature names must be identifiers"
            )
        if not isinstance(spec, str) or not spec:
            raise ExportError(
                "invalid_triton_signature", f"signature for {name} must be a string"
            )
    return signature


def _specialization(signature, specialization):
    if specialization is None:
        return {}
    if not isinstance(specialization, dict):
        raise ExportError(
            "invalid_triton_signature", "specialization must be an object"
        )
    positions = {name: index for index, name in enumerate(signature)}
    attrs = {}
    for name, constraints in specialization.items():
        if name not in positions:
            raise ExportError(
                "invalid_triton_signature", f"specialization names unknown argument {name}"
            )
        if not isinstance(constraints, dict):
            raise ExportError(
                "invalid_triton_signature", f"specialization for {name} must be an object"
            )
        for key, value in constraints.items():
            if key != "divisibility":
                raise ExportError(
                    "invalid_triton_signature",
                    f"unsupported specialization constraint {key!r}",
                )
            if type(value) is not int or value < 1 or value & (value - 1):
                raise ExportError(
                    "invalid_triton_signature",
                    f"divisibility for {name} must be a positive power of two",
                )
            attrs[(positions[name],)] = [("tt.divisibility", value)]
    return attrs


def _constexpr_dtype(value, name):
    if type(value) is bool:
        return "i1"
    if type(value) is int:
        return "i32" if -(2**31) <= value < 2**31 else "i64"
    if type(value) is float:
        return "f32"
    raise ExportError(
        "invalid_triton_signature",
        f"constexpr {name} must be a boolean, integer or float literal",
    )


def _parse_type(text):
    """Split a typed MLIR value type into (dtype, compile-time tile shape)."""
    text = str(text).strip()
    match = _TENSOR_TYPE.fullmatch(text)
    shape = ()
    if match is not None:
        inner = match.group(1)
        parts = inner.split("x")
        if len(parts) < 2:
            raise ExportError(
                "unsupported_triton_layout", f"unsupported tensor type {text!r}"
            )
        dims = []
        for part in parts[:-1]:
            if not part.isdigit():
                raise ExportError(
                    "unsupported_triton_layout", f"dynamic tensor extent in {text!r}"
                )
            dims.append(int(part))
        shape = tuple(dims)
        text = parts[-1]
    pointer = _POINTER_TYPE.fullmatch(text)
    if pointer is not None:
        if pointer.group(1) not in ELEMENT_DTYPES:
            raise ExportError(
                "unsupported_triton_layout", f"unsupported pointer type {text!r}"
            )
        return f"ptr<{pointer.group(1)}>", shape
    if text in ELEMENT_DTYPES:
        return text, shape
    raise ExportError("unsupported_triton_layout", f"unsupported value type {text!r}")


def _collect(module):
    records = []
    blocks = {}

    def visit(op):
        name = op.get_name()
        block = op.get_block()
        block_id = block.id() if block is not None else None
        if block_id is not None and block_id not in blocks:
            arguments = []
            for index in range(block.get_num_arguments()):
                argument = block.get_argument(index)
                arguments.append((argument.id(), str(argument.get_type())))
            blocks[block_id] = arguments
        parent_region = block.get_parent().id() if block is not None else None
        results = []
        for index in range(op.get_num_results()):
            value = op.get_result(index)
            results.append((value.id(), str(value.get_type())))
        operands = [op.get_operand(index).id() for index in range(op.get_num_operands())]
        regions = [op.get_region(index).id() for index in range(op.get_num_regions())]
        attributes = {}
        for key in ("axis", "value", "start", "end", "predicate"):
            item = op.get_int_attr(key)
            if item is not None:
                attributes[key] = item
        if name == "arith.constant" and "value" not in attributes:
            boolean = op.get_bool_attr("value")
            if boolean is not None:
                attributes["value"] = boolean
        records.append(
            {
                "name": name,
                "block": block_id,
                "parent_region": parent_region,
                "results": results,
                "operands": operands,
                "regions": regions,
                "attributes": attributes,
                "sym_name": op.get_str_attr("sym_name"),
            }
        )

    module.walk(visit)
    return records, blocks


def _as_int(value, name):
    if type(value) is not int:
        raise ExportError("unsupported_triton_operation", f"{name} must be an integer")
    return value


class Translator:
    def __init__(self, records, blocks, entry, signature, specialization, constexprs):
        self.records = records
        self.blocks = blocks
        self.entry = entry
        self.signature = signature
        self.specialization = specialization
        self.constexprs = constexprs
        self.value_ids = {}
        self.values = []
        self.definer = {}
        self.op_counter = 0
        self._index()

    def _index(self):
        self.region_owner = {}
        for index, record in enumerate(self.records):
            for region in record["regions"]:
                self.region_owner[region] = index
        self.children = defaultdict(list)
        for index, record in enumerate(self.records):
            parent = self.region_owner.get(record["parent_region"])
            record["parent"] = parent
            if parent is not None:
                self.children[parent].append(index)
            for value_id, _ in record["results"]:
                self.definer[value_id] = index
        candidates = [
            index
            for index, record in enumerate(self.records)
            if record["name"] == "tt.func" and record["sym_name"] == self.entry
        ]
        if len(candidates) != 1:
            raise ExportError(
                "invalid_triton_signature",
                f"expected exactly one tt.func named {self.entry!r}",
            )
        self.function = candidates[0]
        children = self.children.get(self.function, [])
        if not children:
            raise ExportError(
                "invalid_triton_signature", "the entry function has no body"
            )
        self.entry_block = self.records[children[0]]["block"]
        self.entry_arguments = self.blocks[self.entry_block]
        runtime = [
            name for name, spec in self.signature.items() if spec != "constexpr"
        ]
        if len(runtime) != len(self.entry_arguments):
            raise ExportError(
                "invalid_triton_signature",
                "the signature and the emitted function arguments disagree",
            )
        self.runtime_names = runtime
        self.entry_argument_ids = {
            value_id: index
            for index, (value_id, _) in enumerate(self.entry_arguments)
        }

    def _groups(self, index):
        groups = {region: [] for region in self.records[index]["regions"]}
        for child in self.children.get(index, []):
            parent_region = self.records[child]["parent_region"]
            if parent_region in groups:
                groups[parent_region].append(child)
        return groups

    def _declare(self, value_id, type_text):
        if value_id in self.value_ids:
            return
        dtype, shape = _parse_type(type_text)
        self.value_ids[value_id] = f"v{len(self.value_ids)}"
        self.values.append({"id": self.value_ids[value_id], "dtype": dtype, "shape": list(shape)})

    def _assign(self, index):
        record = self.records[index]
        for value_id, type_text in record["results"]:
            self._declare(value_id, type_text)
        groups = self._groups(index)
        for region in record["regions"]:
            children = groups[region]
            if not children:
                raise ExportError(
                    "unsupported_triton_operation", "empty control-flow region"
                )
            block = self.records[children[0]]["block"]
            for value_id, type_text in self.blocks[block]:
                self._declare(value_id, type_text)
            for child in children:
                self._assign(child)

    def _map(self, record):
        name = record["name"]
        attributes = record["attributes"]
        if name.startswith("tt.atomic"):
            raise ExportError(
                "unsupported_triton_memory_effect", "atomic memory effects are unsupported"
            )
        if name in LAYOUT_OPS:
            raise ExportError(
                "unsupported_triton_layout", f"{name} layout conversion is unsupported"
            )
        if name in REGION_OPS:
            raise ExportError(
                "unsupported_triton_operation", f"{name} control flow is unsupported"
            )
        if name.startswith("tt.reduce") or name.startswith("tt.scan"):
            raise ExportError(
                "unsupported_triton_operation", f"{name} reduction is unsupported"
            )
        if name == "tt.get_program_id":
            return "program_id", {"axis": _as_int(attributes.get("axis"), "program axis")}
        if name == "arith.constant":
            if "value" not in attributes:
                raise ExportError(
                    "unsupported_triton_constant",
                    "only integer and boolean literal constants are supported",
                )
            return "constant", {"value": attributes["value"]}
        if name == "tt.make_range":
            return "range", {
                "start": _as_int(attributes.get("start"), "range start"),
                "end": _as_int(attributes.get("end"), "range end"),
            }
        if name == "tt.splat":
            return "splat", {}
        if name == "tt.addptr":
            return "add_ptr", {}
        if name == "tt.load":
            return "load", {"masked": len(record["operands"]) >= 2}
        if name == "tt.store":
            return "store", {"masked": len(record["operands"]) >= 3}
        if name == "tt.return":
            return "return", {}
        if name == "scf.for":
            return "for", {}
        if name == "scf.yield":
            return "yield", {}
        if name == "arith.cmpi":
            predicate = CMPI_PREDICATES.get(attributes.get("predicate"))
            if predicate is None:
                raise ExportError(
                    "unsupported_triton_operation", "unsupported integer comparison"
                )
            return "cmpi", {"predicate": predicate}
        if name == "arith.cmpf":
            predicate = CMPF_PREDICATES.get(attributes.get("predicate"))
            if predicate is None:
                raise ExportError(
                    "unsupported_triton_operation", "unsupported float comparison"
                )
            return "cmpf", {"predicate": predicate}
        if name == "arith.select":
            return "select", {}
        if not name.startswith("arith."):
            raise ExportError(
                "unsupported_triton_operation", f"unsupported operation {name!r}"
            )
        suffix = name.split(".", 1)[1]
        if suffix in BINARY_INTS or suffix in BINARY_FLOATS:
            return suffix, {}
        if suffix == "negf":
            return "negf", {}
        if suffix in CASTS:
            source = self._value_dtype(record["operands"][0])
            target = self._value_dtype(record["results"][0][0])
            return "cast", {"from": source, "to": target}
        raise ExportError(
            "unsupported_triton_operation", f"unsupported arithmetic operation {name!r}"
        )

    def _value_dtype(self, value_id):
        for value in self.values:
            if value["id"] == self.value_ids.get(value_id):
                return value["dtype"]
        raise ExportError("unsupported_triton_operation", "unknown operand value")

    def _root_argument(self, value_id):
        if value_id in self.entry_argument_ids:
            return self.entry_argument_ids[value_id]
        index = self.definer.get(value_id)
        if index is None:
            return None
        record = self.records[index]
        name = record["name"]
        if name in ("tt.splat", "tt.addptr") and record["operands"]:
            return self._root_argument(record["operands"][0])
        if name.startswith("arith.") and name.split(".", 1)[1] in CASTS:
            return self._root_argument(record["operands"][0])
        return None

    def _accesses(self):
        access = {}
        for record in self.records:
            if record["name"] not in ("tt.load", "tt.store") or not record["operands"]:
                continue
            argument = self._root_argument(record["operands"][0])
            if argument is None:
                raise ExportError(
                    "unsupported_triton_memory_effect",
                    "memory access is not attributable to a kernel argument",
                )
            mode = "read" if record["name"] == "tt.load" else "write"
            previous = access.get(argument)
            if previous and previous != mode:
                raise ExportError(
                    "unsupported_triton_memory_effect",
                    "a kernel argument is both read and written",
                )
            access[argument] = mode
        return access

    def _emit(self, index):
        record = self.records[index]
        opcode, attributes = self._map(record)
        operands = [self.value_ids[value] for value in record["operands"]]
        results = [self.value_ids[value] for value, _ in record["results"]]
        regions = []
        groups = self._groups(index)
        for region in record["regions"]:
            children = groups[region]
            block = self.records[children[0]]["block"]
            regions.append(
                {
                    "arguments": [
                        self.value_ids[value] for value, _ in self.blocks[block]
                    ],
                    "operations": [self._emit(child) for child in children],
                }
            )
        operation = {
            "id": f"op{self.op_counter}",
            "opcode": opcode,
            "operands": operands,
            "results": results,
            "attributes": attributes,
            "regions": regions,
        }
        self.op_counter += 1
        return operation

    def parameters(self):
        access = self._accesses()
        parameters = []
        for index, name in enumerate(self.runtime_names):
            value_id, type_text = self.entry_arguments[index]
            dtype, _ = _parse_type(type_text)
            kind = "pointer" if dtype.startswith("ptr<") else "scalar"
            constraints = self.specialization.get(name, {})
            alignment = constraints.get("divisibility", 1)
            parameters.append(
                {
                    "name": name,
                    "kind": kind,
                    "dtype": dtype[4:-1] if kind == "pointer" else dtype,
                    "access": access.get(index, "none"),
                    "alignment_bytes": alignment,
                    "value_id": self.value_ids[value_id],
                    "value": None,
                }
            )
        for name, spec in self.signature.items():
            if spec != "constexpr":
                continue
            value = self.constexprs.get(name)
            parameters.append(
                {
                    "name": name,
                    "kind": "constexpr",
                    "dtype": _constexpr_dtype(value, name),
                    "access": "none",
                    "alignment_bytes": 1,
                    "value_id": "",
                    "value": value,
                }
            )
        return parameters

    def translate(self):
        for value_id, type_text in self.entry_arguments:
            self._declare(value_id, type_text)
        for child in self.children[self.function]:
            self._assign(child)
        operations = [self._emit(child) for child in self.children[self.function]]
        return {
            "entry": self.entry,
            "parameters": self.parameters(),
            "values": self.values,
            "operations": operations,
        }


def _generate_ttir(triton, kernel, request):
    signature = _signature(request.get("signature"))
    constexprs = request.get("constexprs") or {}
    if not isinstance(constexprs, dict):
        raise ExportError("invalid_triton_signature", "constexprs must be an object")
    for name, value in constexprs.items():
        if signature.get(name) != "constexpr":
            raise ExportError(
                "invalid_triton_signature", f"constexpr {name} is not declared constexpr"
            )
        _constexpr_dtype(value, name)
    missing = [
        name for name, spec in signature.items() if spec == "constexpr" and name not in constexprs
    ]
    if missing:
        raise ExportError(
            "invalid_triton_signature", f"missing constexpr values for {missing}"
        )
    attrs = _specialization(signature, request.get("specialization"))

    options = request.get("options") or {}
    if not isinstance(options, dict):
        raise ExportError("invalid_triton_signature", "options must be an object")
    target_name = options.get("target", "cuda")
    arch = options.get("arch", 80)
    warp_size = options.get("warp_size", 32)
    if not isinstance(target_name, str) or type(arch) is not int or type(warp_size) is not int:
        raise ExportError("invalid_triton_signature", "invalid target options")
    option_keys = {
        "num_warps",
        "num_stages",
        "num_ctas",
        "maxnreg",
        "enable_fp_fusion",
        "launch_cooperative_grid",
        "sanitize_overflow",
        "debug",
        "ptx_version",
    }
    unknown = set(options) - option_keys - {"target", "arch", "warp_size"}
    if unknown:
        raise ExportError(
            "invalid_triton_signature", f"unsupported backend options {sorted(unknown)}"
        )
    parsed_options = {key: value for key, value in options.items() if key in option_keys}

    from triton.backends.compiler import GPUTarget
    from triton.compiler.compiler import ASTSource, make_backend

    source = ASTSource(
        fn=kernel, signature=signature, constexprs=constexprs, attrs=attrs
    )
    target = GPUTarget(target_name, arch, warp_size)
    backend = make_backend(target)
    active = backend.parse_options(parsed_options)
    context = triton._C.libtriton.ir.context()
    triton._C.libtriton.ir.load_dialects(context)
    backend.load_dialects(context)
    codegen_fns = backend.get_codegen_implementation(active)
    module_map = backend.get_module_map()
    try:
        module = source.make_ir(target, active, codegen_fns, module_map, context)
        stages = {}
        backend.add_stages(stages, active, source.language)
        if "ttir" not in stages:
            raise ExportError(
                "upstream_toolchain_unqualified",
                "the qualified backend does not expose a TTIR stage",
            )
        module = stages["ttir"](module, {"hash": "llk-export"})
        text = module.str_nodebug()
    except ExportError:
        raise
    except Exception as error:
        raise ExportError(
            "triton_compilation_failed", f"upstream TTIR generation failed: {error}"
        ) from error
    if len(text.encode("utf-8")) > MAX_TTIR_BYTES:
        raise ExportError("triton_export_failed", "generated TTIR is too large")
    return module, text


def _request_target(request):
    options = request.get("options") or {}
    if not isinstance(options, dict):
        raise ExportError("invalid_triton_signature", "options must be an object")
    target_name = options.get("target", "cuda")
    arch = options.get("arch", 80)
    if not isinstance(target_name, str) or not target_name or type(arch) is not int:
        raise ExportError("invalid_triton_signature", "invalid target options")
    return f"{target_name}:{arch}"


def _export(request, triton, identity):
    module_path = request.get("module_path")
    function = request.get("function")
    if not isinstance(module_path, str) or not module_path:
        raise ExportError("invalid_triton_request", "module_path is required")
    if not isinstance(function, str) or not _IDENTIFIER.fullmatch(function):
        raise ExportError(
            "invalid_triton_request", "function must be an identifier"
        )
    signature = _signature(request.get("signature"))
    specialization = request.get("specialization") or {}
    if not isinstance(specialization, dict):
        raise ExportError("invalid_triton_signature", "specialization must be an object")
    kernel = _load_kernel(module_path, function)
    constexprs = request.get("constexprs") or {}
    expected = list(getattr(kernel, "arg_names", ()))
    if list(signature) != expected:
        raise ExportError(
            "invalid_triton_signature",
            f"signature order {list(signature)} does not match kernel "
            f"argument order {expected}",
        )
    module, text = _generate_ttir(triton, kernel, request)
    records, blocks = _collect(module)
    translator = Translator(records, blocks, function, signature, specialization, constexprs)
    graph = translator.translate()

    source_bytes = Path(module_path).read_bytes()
    template = {
        "schema_version": 1,
        "required_runtime_grid": True,
        "entry": graph["entry"],
        "parameters": graph["parameters"],
        "values": graph["values"],
        "operations": graph["operations"],
        "source_hash": hashlib.sha256(source_bytes).hexdigest(),
        "ttir_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "toolchain": identity,
    }
    return {"ttir": text, "template": template, "toolchain": identity}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", help="path to the protocol-v1 request JSON")
    parser.add_argument("--check", action="store_true", help="qualify the toolchain only")
    parser.add_argument(
        "--identity",
        action="store_true",
        help="report the observed toolchain identity without a lock",
    )
    parser.add_argument("--target", default="cuda:80")
    parser.add_argument("--triton-python", default=None)
    parser.add_argument("--lock", default=None)
    args = parser.parse_args(argv)
    modes = [bool(args.request), args.check, args.identity]
    if sum(modes) != 1:
        _emit(
            {
                "protocol_version": PROTOCOL_VERSION,
                "ok": False,
                "error": {
                    "code": "invalid_triton_request",
                    "message": "pass exactly one of --request, --check or --identity",
                    "stage": "request",
                },
            }
        )
        return 2
    if not args.identity and not args.lock:
        _emit(
            {
                "protocol_version": PROTOCOL_VERSION,
                "ok": False,
                "error": {
                    "code": "invalid_triton_request",
                    "message": "--lock is required outside --identity",
                    "stage": "request",
                },
            }
        )
        return 2

    # Keep every upstream print away from the framed result stream.
    sys.stdout = sys.stderr
    try:
        triton = _import_triton(args.triton_python)
        if args.identity:
            result = {
                "toolchain": _toolchain_identity(triton, args.target),
            }
        else:
            lock = _read_lock(args.lock)
            target = lock.get("target")
            if not isinstance(target, str) or not target:
                raise ExportError(
                    "upstream_toolchain_unqualified", "the lock does not record a target"
                )
            if args.check:
                identity = _toolchain_identity(triton, target)
                _require_qualified(identity, lock)
                result = {"toolchain": identity}
            else:
                request = _read_request(args.request)
                identity = _toolchain_identity(triton, _request_target(request))
                _require_qualified(identity, lock)
                result = _export(request, triton, identity)
        _emit({"protocol_version": PROTOCOL_VERSION, "ok": True, "result": result})
        return 0
    except ExportError as error:
        _emit(
            {
                "protocol_version": PROTOCOL_VERSION,
                "ok": False,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "stage": error.stage,
                    "details": error.details,
                },
            }
        )
        return 1
    except BaseException as error:  # never let an exception escape unframed
        _emit(
            {
                "protocol_version": PROTOCOL_VERSION,
                "ok": False,
                "error": {
                    "code": "triton_export_failed",
                    "message": f"{type(error).__name__}: {error}",
                    "stage": "export",
                    "details": {},
                },
            }
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
