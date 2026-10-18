"""Read the bounded YAML schema emitted by micro-perf, without dependencies.

This accepts its indented maps, quoted diagnostic lists, scalar values and
empty collections. Other YAML constructs are deliberately rejected; this is
an artifact-format reader, not a general YAML loader.
"""

import json
import re


def parse_report(text):
    if not isinstance(text, str):
        raise ValueError("report must be text")
    lines = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if "\t" in line or indent % 2:
            raise ValueError(f"invalid indentation at line {number}")
        lines.append((indent, line.strip()))

    def scalar(value):
        if value == "{}":
            return {}
        if value == "[]":
            return []
        if value in ("true", "false"):
            return value == "true"
        if re.fullmatch(r"-?\d+", value):
            return int(value)
        if re.fullmatch(r"-?\d+\.\d+", value):
            return float(value)
        if value.startswith('"'):
            return json.loads(value)
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_./<>()=,+-]*", value):
            return value
        raise ValueError(f"unsupported report scalar: {value!r}")

    def block(index, indent):
        is_list = lines[index][1].startswith("- ")
        result = [] if is_list else {}
        while index < len(lines) and lines[index][0] >= indent:
            depth, line = lines[index]
            if depth != indent:
                raise ValueError("unexpected report indentation")
            if is_list:
                if not line.startswith('- "'):
                    raise ValueError("expected quoted diagnostic list item")
                value = scalar(line[2:])
                if not isinstance(value, str):
                    raise ValueError("diagnostic must be a string")
                result.append(value)
                index += 1
                continue
            match = re.fullmatch(r"([^:#]+):(?: +(.*))?", line)
            if not match:
                raise ValueError(f"expected report mapping: {line!r}")
            key, value = match.groups()
            if key != key.strip() or key in result:
                raise ValueError(f"invalid or duplicate report key: {key!r}")
            index += 1
            if value is None:
                if index == len(lines) or lines[index][0] != indent + 2:
                    raise ValueError(f"missing report collection: {key}")
                result[key], index = block(index, indent + 2)
            else:
                result[key] = scalar(value)
        return result, index

    if not lines or lines[0][0] != 0:
        raise ValueError("missing report root")
    report, end = block(0, 0)
    if end != len(lines) or not isinstance(report, dict):
        raise ValueError("report root must be a mapping")
    return report


def validate_analysis(text, symbol, machine):
    """Return parsed strict report, or raise a reason-bearing ValueError."""
    report = parse_report(text)
    if type(report.get("schema_version")) is not int or \
            report["schema_version"] != 1:
        raise ValueError("unsupported report schema_version")
    if report.get("kernel") != symbol or report.get("machine") != machine:
        raise ValueError("report kernel/machine does not match row")
    if type(report.get("level")) is not int or report["level"] != 1:
        raise ValueError("report must contain level-one analysis")
    if report.get("complete") is not True or \
            report.get("incomplete_reasons") != []:
        raise ValueError("analysis is not strictly complete")
    if report.get("capacity_violations") != []:
        raise ValueError("analysis has missing or nonempty capacity violations")
    for section, field in (("totals", "bytes"), ("tiles", "live_bytes")):
        counters = report.get(section, {})
        counters = counters.get(field) if isinstance(counters, dict) else None
        if not isinstance(counters, dict) or any(
                type(value) is not int or value < 0
                for value in counters.values()):
            raise ValueError(f"analysis {section}.{field} must be unsigned counters")
    return report
