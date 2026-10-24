#!/usr/bin/env python3
"""Run the selected SVE object under qemu and check its numbers (E10).

This is functional evidence only: the emulated run shows the selected AArch64
subset computes the fixture's numbers. It is not a measurement, and the harness
is deliberately libc-free and static so the object under test is the compiler's
output rather than a runtime's.

Every object that links is also replayed on two CPUs the harness must refuse --
one without SVE and one whose vector length is not the modeled 512 bits -- and
must exit with the harness's distinct rejection codes (70 and 71). That keeps
the host checks that qualify the executable subset live rather than assumed.

The script fails -- never skips -- when qemu or the cross compiler is missing;
CTest's `REQUIRED` lookups refuse to configure in that case.
"""

import json
import os
import re
import shutil
import struct
import subprocess
import sys

CASES = [
    ("vector_add_f32", "arm_vector_add_f32", "add", 8, 8),
    ("vector_mul_f32", "arm_vector_mul_f32", "mul", 8, 8),
    ("vector_add_f32_full", "arm_vector_add_f32_full", "add", 16, 16),
    ("vector_mul_f32_full", "arm_vector_mul_f32_full", "mul", 16, 16),
]

# The harness's own host checks are the executable subset's guardrails: it
# exits 70 when the CPU does not advertise SVE and 71 when PR_SVE_GET_VL is not
# the modeled 512 bits. Running every object on emulated CPUs that violate each
# condition proves those branches are live -- a zero exit would mean the
# artifact computed numbers on a host the subset is not contracted for.
# `max,sve-max-vq=8` caps the vector length at 8 x 128 = 1024 bits.
NEGATIVE_CONTROLS = [
    ("no-sve", "max,sve=off", 70),
    ("wrong-vl", "max,sve-max-vq=8", 71),
]

HARNESS = r"""
#include <stdint.h>
#include <stddef.h>

typedef struct {
  float *allocated;
  float *aligned;
  int64_t offset;
  int64_t size0;
  int64_t size1;
  int64_t stride0;
  int64_t stride1;
} MemRef2D;

extern void WRAPPER(MemRef2D *a, MemRef2D *b, MemRef2D *out);

static unsigned char heap[1 << 20] __attribute__((aligned(64)));
static size_t used;

void *malloc(size_t n) {
  size_t start = (used + 63) & ~(size_t)63;
  if (start + n > sizeof(heap))
    for (;;) { }
  used = start + n;
  return heap + start;
}
void *aligned_alloc(size_t alignment, size_t n) {
  size_t start = (used + alignment - 1) & ~(alignment - 1);
  if (start + n > sizeof(heap))
    for (;;) { }
  used = start + n;
  return heap + start;
}
void free(void *p) { (void)p; }
void *memset(void *dst, int value, size_t n) {
  unsigned char *bytes = dst;
  for (size_t i = 0; i < n; ++i)
    bytes[i] = (unsigned char)value;
  return dst;
}
void *memcpy(void *dst, const void *src, size_t n) {
  unsigned char *out = dst;
  const unsigned char *in = src;
  for (size_t i = 0; i < n; ++i)
    out[i] = in[i];
  return dst;
}
static long syscall3(long, long, long, long);
void memrefCopy(int64_t element_size, void *source, void *destination) {
  (void)element_size;
  (void)source;
  (void)destination;
  syscall3(93, 75, 0, 0);  // this subset must not reach a generic copy
  for (;;) { }
}

static const uint32_t LHS[ROWS * COLS] = { LHS_DATA };
static const uint32_t RHS[ROWS * COLS] = { RHS_DATA };
static const uint32_t EXPECTED[ROWS * COLS] = { EXPECTED_DATA };

static long syscall3(long n, long a, long b, long c) {
  register long x8 __asm__("x8") = n;
  register long x0 __asm__("x0") = a;
  register long x1 __asm__("x1") = b;
  register long x2 __asm__("x2") = c;
  __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8), "r"(x1), "r"(x2) : "memory");
  return x0;
}

static void write_all(const char *text, long length) {
  syscall3(64, 1, (long)text, length);
}
#define WRITE_LITERAL(text) write_all((text), (long)(sizeof(text) - 1))

static float bits(uint32_t value) {
  union { uint32_t u; float f; } cast;
  cast.u = value;
  return cast.f;
}

static uint32_t as_bits(float value) {
  union { uint32_t u; float f; } cast;
  cast.f = value;
  return cast.u;
}

static int is_nan32(uint32_t value) {
  return (value & 0x7f800000u) == 0x7f800000u && (value & 0x7fffffu);
}

void _start(void) {
  unsigned long hwcap = 0;
  /* getauxval is unavailable without libc; qemu sets HWCAP in the auxv the
     kernel passes on the stack. Parsing it here keeps the harness libc-free. */
  unsigned long *stack;
  __asm__("mov %0, sp" : "=r"(stack));
  unsigned long argc = stack[0];
  unsigned long *aux = stack + 1 + argc + 1;  // envp[0]
  while (*aux)
    ++aux;                                    // skip envp
  ++aux;                                      // skip its NULL
  while (*aux) {
    if (aux[0] == 16) { hwcap = aux[1]; break; }
    aux += 2;
  }
  if (!(hwcap & (1UL << 22))) {
    WRITE_LITERAL("unsupported: no sve hwcap\n");
    syscall3(93, 70, 0, 0);
  }
  /* PR_SVE_GET_VL = 51; PR_SVE_VL_LEN_MASK is the low 16 bits and the kernel
     reports the length there in bytes, so bytes * 8 is the bit width. */
  long vl = syscall3(167, 51, 0, 0);
  if (vl < 0 || (vl & 0xffff) * 8 != 512) {
    WRITE_LITERAL("unsupported: vector length is not 512\n");
    syscall3(93, 71, 0, 0);
  }

  static float lhs[ROWS * COLS] __attribute__((aligned(64)));
  static float rhs[ROWS * COLS] __attribute__((aligned(64)));
  /* The mapped ABI appends borrowed results after the inputs, so the output
     descriptor is the *last* argument. The caller owns that buffer and passes
     it in; the callee reads its fields and writes through aligned + offset.
     The 32-word tail is a canary for a write past the ROWS * COLS result. */
  static float out[ROWS * COLS + 32] __attribute__((aligned(64)));
  for (int i = 0; i < ROWS * COLS; ++i) {
    lhs[i] = bits(LHS[i]);
    rhs[i] = bits(RHS[i]);
  }
  for (unsigned i = 0; i < sizeof(out) / sizeof(out[0]); ++i)
    out[i] = bits(0x5a5a5a5au);
  MemRef2D a = {lhs, lhs, 0, ROWS, COLS, COLS, 1};
  MemRef2D b = {rhs, rhs, 0, ROWS, COLS, COLS, 1};
  MemRef2D result = {out, out, 0, ROWS, COLS, COLS, 1};
  WRAPPER(&a, &b, &result);
  const float *values = result.aligned + result.offset;
  for (int i = 0; i < ROWS * COLS; ++i) {
    uint32_t got = as_bits(values[i]);
    if (is_nan32(EXPECTED[i]) ? !is_nan32(got) : got != EXPECTED[i]) {
      WRITE_LITERAL("failed: numeric mismatch\n");
      syscall3(93, 73, 0, 0);
    }
  }
  for (unsigned i = 0; i < sizeof(out) / sizeof(out[0]) - ROWS * COLS; ++i) {
    if (as_bits(out[ROWS * COLS + i]) != 0x5a5a5a5au) {
      WRITE_LITERAL("failed: canary written\n");
      syscall3(93, 74, 0, 0);
    }
  }
  WRITE_LITERAL("pass\n");
  syscall3(93, 0, 0, 0);
}
"""


# The first four elements of both operands in every case: positive zero,
# negative zero, positive infinity and a quiet NaN. Signed zero and infinity
# have one exact bit pattern each, so they pin the arithmetic's sign handling;
# the NaN is compared by classification instead (the harness accepts any NaN).
SPECIAL = [0x00000000, 0x80000000, 0x7f800000, 0x7fc00000]  # +0, -0, +inf, NaN

# The correctly rounded result of the SPECIAL operands for each operation,
# written as bits so signed zero is asserted rather than coerced by a Python
# float round-trip. `-0 + -0` stays negative; `-0 * -0` is positive.
SPECIAL_EXPECTED = {
    "add": [0x00000000, 0x80000000, 0x7f800000, 0x7fc00000],
    "mul": [0x00000000, 0x00000000, 0x7f800000, 0x7fc00000],
}


def bit_patterns(values):
    return ", ".join("0x%08xu" % value for value in values)


def _float_from_bits(value):
    return struct.unpack("<f", struct.pack("<I", value))[0]


def _bits_from_float(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def expected_data(name, rows, cols):
    """Return the case's operand and expected tables as f32 bit patterns.

    The table is bits, not Python floats, so the harness receives exactly the
    value the case names -- including -0.0 and the NaN payload class -- and the
    ordinary elements keep the fixture's deterministic pattern. Python's
    binary64 arithmetic is exact here: two f32 operands add and multiply into
    values a double represents exactly, so rounding once to f32 is the
    correctly rounded f32 result.
    """
    lhs, rhs = [], []
    for i in range(rows * cols):
        if i < len(SPECIAL):
            lhs.append(SPECIAL[i])
            rhs.append(SPECIAL[i])
        else:
            lhs.append(0x3f800000 + i * 7)
            rhs.append(0x40000000 + i * 3)
    values = list(SPECIAL_EXPECTED[name])
    for a_bits, b_bits in zip(lhs[len(SPECIAL):], rhs[len(SPECIAL):]):
        a, b = _float_from_bits(a_bits), _float_from_bits(b_bits)
        values.append(_bits_from_float(a + b if name == "add" else a * b))
    return lhs, rhs, values


# Substitute every placeholder in one pass: a sequential str.replace chain
# would rescan each replacement, so a value that happened to contain a later
# placeholder name would be corrupted.
_PLACEHOLDER = re.compile(
    r"\b(?:WRAPPER|ROWS|COLS|LHS_DATA|RHS_DATA|EXPECTED_DATA)\b")


def render_harness(wrapper, rows, cols, lhs, rhs, expected):
    values = {
        "WRAPPER": wrapper,
        "ROWS": str(rows),
        "COLS": str(cols),
        "LHS_DATA": bit_patterns(lhs),
        "RHS_DATA": bit_patterns(rhs),
        "EXPECTED_DATA": bit_patterns(expected),
    }
    return _PLACEHOLDER.sub(lambda match: values[match.group(0)], HARNESS)


def resolve_tool(path, description):
    """Return the executable path, or exit non-zero when it is missing.

    A missing runner is a failure, not a skipped test: the evidence exists to
    prove execution, and no execution happened.
    """
    resolved = shutil.which(path)
    if not resolved:
        raise SystemExit("arm sve evidence: missing %s '%s'" % (description, path))
    return resolved


def run_negative_controls(qemu, binaries, failures):
    """Run every built object on CPUs the harness must reject.

    `binaries` maps a case name to its linked harness. Each object runs once
    per entry in NEGATIVE_CONTROLS and must exit with exactly that entry's
    code. Zero is the failure the control exists to catch: it would mean the
    harness ran and compared numbers on a CPU without the contracted SVE
    support. Any other code is also a failure, because it is not the branch
    being asserted.
    """
    for fixture, binary in binaries.items():
        for label, cpu, expected in NEGATIVE_CONTROLS:
            try:
                run = subprocess.run([qemu, "-cpu", cpu, binary],
                                     capture_output=True, text=True,
                                     timeout=120)
            except subprocess.TimeoutExpired:
                failures.append("%s: %s negative run timed out" %
                                (fixture, label))
                continue
            detail = "%s%s" % (run.stdout, run.stderr)
            if run.returncode == 0:
                failures.append(
                    "%s: %s negative run exited 0; the harness accepted CPU "
                    "'%s'\n%s" % (fixture, label, cpu, detail))
            elif run.returncode != expected:
                failures.append(
                    "%s: %s negative run exited %d, expected %d\n%s" %
                    (fixture, label, run.returncode, expected, detail))


def main():
    if len(sys.argv) != 6:
        print("usage: arm_sve_evidence.py <llk-compile> <source-dir> "
              "<work-dir> <qemu-aarch64> <aarch64-gcc>", file=sys.stderr)
        return 2
    llk_compile, source_dir, work_dir, qemu, cross_gcc = sys.argv[1:]
    llk_compile = resolve_tool(llk_compile, "llk-compile")
    qemu = resolve_tool(qemu, "qemu-aarch64")
    cross_gcc = resolve_tool(cross_gcc, "aarch64 cross compiler")
    os.makedirs(work_dir, exist_ok=True)
    failures = []
    graph_ids = {}
    binaries = {}
    for fixture, symbol, name, rows, cols in CASES:
        case_dir = os.path.join(work_dir, fixture)
        os.makedirs(case_dir, exist_ok=True)
        source = os.path.join(source_dir, "test/Conversion/MicroMapping",
                              "Inputs/arm", fixture + ".mlir")
        obj = os.path.join(case_dir, "kernel.o")
        identity = os.path.join(case_dir, "identity.json")
        command = [
            llk_compile, "--mapping-target=arm-sve",
            "--mapping-root=" + source_dir,
            "--mapping-backend=selected-target",
            "--mapping-stop=lowered", "--emit=object",
            "--target-triple=aarch64-unknown-linux-gnu",
            "--target-features=+sve", "--output=" + obj,
            "--emit-identity=" + identity, source,
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=300)
        except subprocess.TimeoutExpired:
            failures.append("%s: object emission timed out" % fixture)
            continue
        if result.returncode:
            failures.append("%s: object emission failed\n%s" %
                            (fixture, result.stderr))
            continue
        try:
            with open(identity, encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError) as error:
            failures.append("%s: unreadable identity record: %s" %
                            (fixture, error))
            continue
        # The selected backend must carry the whole kernel: a fallback to the
        # reference bridge would make "executed under qemu" a claim about a
        # different lowering than the one this task validates.
        if record.get("referenceGroupsLowered") != 0 or \
                record.get("backendGroupsRealized", 0) <= 0:
            failures.append("%s: selected lowering fell back\n%s" %
                            (fixture, record))
            continue
        if not record.get("planId"):
            failures.append("%s: identity record has no planId\n%s" %
                            (fixture, record))
            continue
        graph_id = record.get("sourceGraphHash")
        if not graph_id:
            failures.append("%s: identity record has no sourceGraphHash\n%s" %
                            (fixture, record))
            continue
        graph_ids[fixture] = graph_id
        expected_wrapper = "_mlir_ciface_" + symbol
        if record.get("wrapperSymbol") != expected_wrapper:
            failures.append("%s: identity names wrapper %r, expected %r" %
                            (fixture, record.get("wrapperSymbol"),
                             expected_wrapper))
            continue
        lhs, rhs, expected = expected_data(name, rows, cols)
        harness_path = os.path.join(case_dir, "harness.c")
        with open(harness_path, "w", encoding="utf-8") as handle:
            handle.write(render_harness(expected_wrapper, rows, cols,
                                        lhs, rhs, expected))
        binary = os.path.join(case_dir, "harness")
        try:
            build = subprocess.run(
                # -fno-stack-protector: a freestanding -nostdlib link has no
                # __stack_chk_fail or TLS guard, and _start takes the address
                # of its descriptors, which a hardening default would
                # instrument.
                [cross_gcc, "-static", "-O2", "-nostdlib", "-ffreestanding",
                 "-fno-builtin", "-fno-stack-protector", "-Wl,-e,_start",
                 harness_path, obj, "-o", binary],
                capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            failures.append("%s: cross link timed out" % fixture)
            continue
        if build.returncode:
            failures.append("%s: cross link failed\n%s%s" %
                            (fixture, build.stdout, build.stderr))
            continue
        binaries[fixture] = binary
        try:
            run = subprocess.run([qemu, "-cpu", "max,sve-max-vq=4", binary],
                                 capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            # A wedged emulator is a failure, not a skipped test.
            failures.append("%s: qemu timed out" % fixture)
            continue
        if run.returncode != 0:
            failures.append("%s: qemu exit %d\n%s%s" %
                            (fixture, run.returncode, run.stdout, run.stderr))
    # Every object that linked is replayed on CPUs the harness must reject, so
    # the passing runs above are not the only thing that ever exercises its
    # host checks.
    run_negative_controls(qemu, binaries, failures)
    # The 8x8 and 16x16 runs must be distinct bindings, not one artifact
    # reused. A plan id hashes the plan's *decisions* -- binding, placements,
    # routes and parameters -- and is deliberately shape-agnostic, so the same
    # rule on the same layout solves to the same plan id at either width. The
    # shape lives in the source-graph identity, which must differ; the equality
    # below would catch a cached or mis-bound full-width run.
    for tail, full in (("vector_add_f32", "vector_add_f32_full"),
                       ("vector_mul_f32", "vector_mul_f32_full")):
        if tail in graph_ids and full in graph_ids and \
                graph_ids[tail] == graph_ids[full]:
            failures.append("%s and %s share the source graph %s" %
                            (tail, full, graph_ids[tail]))
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("arm sve evidence: %d cases passed" % len(CASES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
