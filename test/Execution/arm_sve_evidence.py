#!/usr/bin/env python3
"""Run the selected SVE object under qemu and check its numbers (E10).

This is functional evidence only: the emulated run shows the selected AArch64
subset computes the fixture's numbers. It is not a measurement, and the harness
is deliberately libc-free and static so the object under test is the compiler's
output rather than a runtime's.

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


def float_bits(values):
    return ", ".join("0x%08xu" % struct.unpack("<I", struct.pack("<f", v))[0]
                     for v in values)


def expected_data(name, rows, cols):
    lhs, rhs = [], []
    for i in range(rows * cols):
        lhs.append(struct.unpack("<f", struct.pack("<I", 0x3f800000 + i * 7))[0])
        rhs.append(struct.unpack("<f", struct.pack("<I", 0x40000000 + i * 3))[0])
    if name == "add":
        values = [struct.unpack("<f", struct.pack("<f", a + b))[0]
                  for a, b in zip(lhs, rhs)]
    else:
        values = [struct.unpack("<f", struct.pack("<f", a * b))[0]
                  for a, b in zip(lhs, rhs)]
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
        "LHS_DATA": float_bits(lhs),
        "RHS_DATA": float_bits(rhs),
        "EXPECTED_DATA": float_bits(expected),
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
                not record.get("backendGroupsRealized"):
            failures.append("%s: selected lowering fell back\n%s" %
                            (fixture, record))
            continue
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
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("arm sve evidence: %d cases passed" % len(CASES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
