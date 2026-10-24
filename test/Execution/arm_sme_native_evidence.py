#!/usr/bin/env python3
"""Native Apple M5 SME contraction evidence (E11, tier T3).

This is the gating run. It cross-emits a Mach-O object, links it with an
assembly shim that is the only code that changes PSTATE.SM, and executes it on
real SME hardware. It is not reproducible in CI; the ledger pins the host,
toolchain and artifact digests.
"""

import os
import signal
import subprocess
import sys

import arm_sme_evidence_common as common


def _run(command, timeout=300):
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=timeout)
    if result.returncode:
        raise SystemExit("command failed: %s\n%s\n%s" %
                         (" ".join(command), result.stdout, result.stderr))
    return result


def main():
    if len(sys.argv) != 6:
        print("usage: arm_sme_native_evidence.py <llk-compile> <source-dir> "
              "<work-dir> <clang> <llvm-objdump>", file=sys.stderr)
        return 2
    llk_compile, source_dir, work_dir, clang, objdump = sys.argv[1:]
    os.makedirs(work_dir, exist_ok=True)
    failures = []

    obj, record, wrapper = common.emit_object(
        llk_compile, source_dir, work_dir, common.TRIPLE_DARWIN,
        common.FEATURES)
    common.check_object_format(obj, "darwin")
    common.check_identity(record, common.TRIPLE_DARWIN, wrapper, failures,
                          "contraction_f32")
    common.check_sme_instructions(objdump, obj, failures, "contraction_f32",
                                  ["--mattr=+sme"])

    shim = os.path.join(work_dir, "shim.S")
    harness_c = os.path.join(work_dir, "harness.c")
    with open(shim, "w", encoding="utf-8") as handle:
        handle.write(common.render_shim("_"))
    with open(harness_c, "w", encoding="utf-8") as handle:
        handle.write(common.render_native_harness(wrapper))
    shim_o = os.path.join(work_dir, "shim.o")
    harness_o = os.path.join(work_dir, "harness.o")
    binary = os.path.join(work_dir, "harness")
    _run([clang, "-march=armv9-a+sme", "-c", shim, "-o", shim_o])
    _run([clang, "-O2", "-c", harness_c, "-o", harness_o])
    _run([clang, harness_o, shim_o, obj, "-o", binary])

    run = subprocess.run([binary], capture_output=True, text=True, timeout=120)
    if run.returncode != 0:
        failures.append("native run exit %d\n%s%s" %
                        (run.returncode, run.stdout, run.stderr))

    # Negative control 1: smstart sm alone does not enable ZA.
    probe_c = os.path.join(work_dir, "za_probe.c")
    with open(probe_c, "w", encoding="utf-8") as handle:
        handle.write("""
int main(void) {
  __asm__ volatile("smstart sm");
  __asm__ volatile("zero {za}");
  __asm__ volatile("smstop sm");
  return 0;
}
""")
    probe = os.path.join(work_dir, "za_probe")
    _run([clang, "-O2", "-march=armv9-a+sme", probe_c, "-o", probe])
    za = subprocess.run([probe], capture_output=True, text=True, timeout=30)
    if za.returncode != -signal.SIGILL:
        failures.append(
            "ZA probe exited %d; smstart sm alone must leave ZA disabled and "
            "trap with SIGILL" % za.returncode)

    # Negative control 2: the streaming wrapper cannot be called with SM off.
    off_c = os.path.join(work_dir, "harness_without_shim.c")
    with open(off_c, "w", encoding="utf-8") as handle:
        handle.write(common.render_native_harness(wrapper, without_shim=True))
    off_o = os.path.join(work_dir, "harness_without_shim.o")
    off_bin = os.path.join(work_dir, "harness_without_shim")
    _run([clang, "-O2", "-c", off_c, "-o", off_o])
    _run([clang, off_o, shim_o, obj, "-o", off_bin])
    try:
        off = subprocess.run([off_bin], capture_output=True, text=True,
                             timeout=30)
        if off.returncode == 0:
            failures.append("the wrapper ran successfully with SM off")
    except subprocess.TimeoutExpired:
        pass  # did not produce a passing result; counted as a refusal

    version = _run([clang, "--version"]).stdout.splitlines()[0]
    os_version = _run(["sw_vers", "-productVersion"]).stdout.strip()
    model = _run(["sysctl", "-n", "hw.model"]).stdout.strip()
    print(
        "arm sme native evidence: host=%s macos=%s clang=%r wrapper=%s "
        "plan=%s graph=%s abi=%s svl=512 selected-groups=%s "
        "reference-groups=%s object-sha256=%s harness-sha256=%s "
        "shim-sha256=%s za-probe=SIGILL" % (
            model, os_version, version, wrapper, record.get("planId"),
            record.get("sourceGraphHash"), record.get("abiHash"),
            record.get("selectedGroupsVerified"),
            record.get("referenceGroupsLowered"), common.sha256_file(obj),
            common.sha256_file(harness_c), common.sha256_file(shim)))

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("arm sme native evidence: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
