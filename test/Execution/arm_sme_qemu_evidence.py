#!/usr/bin/env python3
"""QEMU SME1 contraction evidence (E11, tier T2).

Emulated functional evidence only; it never calibrates native timing. QEMU's
`sme-default-vector-length` is in bytes and defaults to 32 (256 bits), so the
positive run requests 64 bytes explicitly (512 bits).
"""

import os
import signal
import subprocess
import sys

import arm_sme_evidence_common as common

POSITIVE_CPU = "max,sme=on,sme-default-vector-length=64"
NO_SME_CPU = "max,sme=off"
WRONG_SVL_CPU = "max,sme=on,sme-default-vector-length=32"

ZA_PROBE = r"""
	.arch	armv9-a+sme
	.text
	.p2align 2
	.globl	_start
_start:
	smstart	sm
	zero	{za}
	smstop	sm
	mov	x0, #0
	mov	x8, #93
	svc	#0
"""


def _run(command, timeout=300):
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=timeout)
    if result.returncode:
        raise SystemExit("command failed: %s\n%s\n%s" %
                         (" ".join(command), result.stdout, result.stderr))
    return result


def main():
    if len(sys.argv) != 7:
        print("usage: arm_sme_qemu_evidence.py <llk-compile> <source-dir> "
              "<work-dir> <qemu-aarch64> <aarch64-gcc> <aarch64-objdump>",
              file=sys.stderr)
        return 2
    (llk_compile, source_dir, work_dir, qemu, cross_gcc,
     objdump) = sys.argv[1:]
    os.makedirs(work_dir, exist_ok=True)
    failures = []

    obj, record, wrapper = common.emit_object(
        llk_compile, source_dir, work_dir, common.TRIPLE_LINUX,
        common.FEATURES)
    common.check_object_format(obj, "linux")
    common.check_identity(record, common.TRIPLE_LINUX, wrapper, failures,
                          "contraction_f32")
    common.check_sme_instructions(objdump, obj, failures, "contraction_f32")

    shim = os.path.join(work_dir, "shim.S")
    harness_c = os.path.join(work_dir, "harness.c")
    with open(shim, "w", encoding="utf-8") as handle:
        handle.write(common.render_shim("", ".arch\tarmv9-a+sme\n"))
    with open(harness_c, "w", encoding="utf-8") as handle:
        handle.write(common.render_qemu_harness(wrapper))
    shim_o = os.path.join(work_dir, "shim.o")
    binary = os.path.join(work_dir, "harness")
    _run([cross_gcc, "-c", shim, "-o", shim_o])
    _run([cross_gcc, "-static", "-O2", "-nostdlib", "-ffreestanding",
          "-fno-builtin", "-fno-stack-protector", "-Wl,-e,_start",
          harness_c, shim_o, obj, "-o", binary])

    run = subprocess.run([qemu, "-cpu", POSITIVE_CPU, binary],
                         capture_output=True, text=True, timeout=120)
    if run.returncode != 0:
        failures.append("positive run exit %d\n%s%s" %
                        (run.returncode, run.stdout, run.stderr))

    for label, cpu, expected in (("no-sme", NO_SME_CPU, 70),
                                 ("wrong-svl", WRONG_SVL_CPU, 71)):
        result = subprocess.run([qemu, "-cpu", cpu, binary],
                                capture_output=True, text=True, timeout=120)
        if result.returncode != expected:
            failures.append("%s: exit %d, expected %d\n%s%s" %
                            (label, result.returncode, expected,
                             result.stdout, result.stderr))

    # QEMU may or may not model "ZA disabled" as a trap. Record what it did;
    # the native T3 control is authoritative for this distinction.
    probe_s = os.path.join(work_dir, "za_probe.S")
    with open(probe_s, "w", encoding="utf-8") as handle:
        handle.write(ZA_PROBE)
    probe_o = os.path.join(work_dir, "za_probe.o")
    probe = os.path.join(work_dir, "za_probe")
    _run([cross_gcc, "-c", probe_s, "-o", probe_o])
    _run([cross_gcc, "-static", "-nostdlib", "-Wl,-e,_start", probe_o,
          "-o", probe])
    za = subprocess.run([qemu, "-cpu", POSITIVE_CPU, probe],
                        capture_output=True, text=True, timeout=60)
    if za.returncode in (-signal.SIGILL, 132):
        za_note = "SIGILL"
    elif za.returncode == 0:
        za_note = "not-modeled-by-qemu"
    else:
        failures.append("ZA probe exited %d" % za.returncode)
        za_note = "unexpected"

    off_c = os.path.join(work_dir, "harness_without_shim.c")
    with open(off_c, "w", encoding="utf-8") as handle:
        handle.write(common.render_qemu_harness(wrapper, without_shim=True))
    off_bin = os.path.join(work_dir, "harness_without_shim")
    _run([cross_gcc, "-static", "-O2", "-nostdlib", "-ffreestanding",
          "-fno-builtin", "-fno-stack-protector", "-Wl,-e,_start",
          off_c, shim_o, obj, "-o", off_bin])
    try:
        off = subprocess.run([qemu, "-cpu", POSITIVE_CPU, off_bin],
                             capture_output=True, text=True, timeout=30)
        if off.returncode == 0:
            failures.append("the wrapper ran successfully with SM off")
    except subprocess.TimeoutExpired:
        pass

    version = _run([qemu, "--version"]).stdout.splitlines()[0]
    print(
        "arm sme qemu evidence: qemu=%r cpu=%s wrapper=%s plan=%s graph=%s "
        "abi=%s svl=512 selected-groups=%s reference-groups=%s "
        "object-sha256=%s harness-sha256=%s shim-sha256=%s za-probe=%s" % (
            version, POSITIVE_CPU, wrapper, record.get("planId"),
            record.get("sourceGraphHash"), record.get("abiHash"),
            record.get("selectedGroupsVerified"),
            record.get("referenceGroupsLowered"), common.sha256_file(obj),
            common.sha256_file(harness_c), common.sha256_file(shim),
            za_note))

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("arm sme qemu evidence: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
