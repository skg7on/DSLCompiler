#!/usr/bin/env python3
"""Reject missing, skipped, failed, or incomplete required CTest evidence."""

import argparse
import sys
import xml.etree.ElementTree as ET


REQUIRED = {"MappedAVX2Acceptance", "MappedAllocationLifetimeTest"}


def parse_outcomes(xml_text):
    root = ET.fromstring(xml_text)
    outcomes = {}
    for case in root.iter("testcase"):
        name = case.attrib.get("name")
        if not name:
            continue
        if case.find("failure") is not None or case.find("error") is not None:
            outcomes[name] = "failed"
        elif case.find("skipped") is not None:
            outcomes[name] = "skipped"
        else:
            outcomes[name] = "passed"
    return outcomes


def validate(outcomes, required=REQUIRED):
    missing = sorted(required - outcomes.keys())
    if missing:
        return "missing required execution: " + ", ".join(missing)
    bad = sorted((name, outcomes[name]) for name in required
                 if outcomes[name] != "passed")
    if bad:
        return "required execution did not pass: " + ", ".join(
            "%s=%s" % item for item in bad)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("junit_xml")
    args = parser.parse_args()
    try:
        with open(args.junit_xml, encoding="utf-8") as handle:
            outcomes = parse_outcomes(handle.read())
    except (OSError, ET.ParseError) as error:
        print("cannot read CTest evidence: %s" % error, file=sys.stderr)
        return 1
    error = validate(outcomes)
    if error:
        print(error, file=sys.stderr)
        return 1
    print("required selected execution evidence passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
