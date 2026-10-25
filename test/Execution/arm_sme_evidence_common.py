"""Shared Arm SME evidence helpers (E11).

The numerical data is integer-valued f32: every product and every partial sum
is exactly representable, so the expected result is independent of the
hardware's internal accumulation order. The harness compares literal bits.
"""

import hashlib
import json
import re
import struct
import subprocess

ROWS = 16
COLS = 16
K = 16
CANARY = 32
SENTINEL = 0x5A5A5A5A
CASE = ("contraction_f32", "arm_contraction")
TRIPLE_DARWIN = "arm64-apple-macosx14.0.0"
TRIPLE_LINUX = "aarch64-unknown-linux-gnu"
FEATURES = ["+sme"]

SME_FMOPA = re.compile(r"\bfmopa\b")
SME_ZERO_ZA = re.compile(r"\bzero\s*\{\s*za\s*\}")
SME_SMSTART_ZA = re.compile(r"\bsmstart\s+za\b")


def parse_hex_identity(value):
    if not isinstance(value, str):
        return None
    try:
        return int(value, 16)
    except ValueError:
        return None


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_sme_instructions(disassembly):
    """True only for code that accumulates, zeroes ZA and enters streaming ZA."""
    return (bool(SME_FMOPA.search(disassembly)) and
            bool(SME_ZERO_ZA.search(disassembly)) and
            bool(SME_SMSTART_ZA.search(disassembly)))


def _f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _dataset(lhs_base, rhs_base, init):
    a = [[lhs_base + i for _ in range(COLS)] for i in range(ROWS)]
    b = [[rhs_base + j for j in range(COLS)] for _ in range(ROWS)]
    initial = [[init for _ in range(COLS)] for _ in range(ROWS)]
    # C[i][j] = init + sum_k A[i][k] * B[k][j]
    #         = init + K * (lhs_base + i) * (rhs_base + j)
    expected = [[init + K * (lhs_base + i) * (rhs_base + j)
                 for j in range(COLS)] for i in range(ROWS)]
    return a, b, initial, expected


def render_tables():
    a1, b1, i1, e1 = _dataset(1, 1, 1)
    a2, b2, i2, e2 = _dataset(2, 2, 2)
    names = {"A1": a1, "B1": b1, "I1": i1, "E1": e1,
             "A2": a2, "B2": b2, "I2": i2, "E2": e2}
    rendered = []
    for name in ("A1", "B1", "I1", "E1", "A2", "B2", "I2", "E2"):
        values = [_f32_bits(v) for row in names[name] for v in row]
        body = ",\n  ".join(
            ", ".join("0x%08xu" % v for v in values[i:i + COLS])
            for i in range(0, len(values), COLS))
        rendered.append(
            "static const uint32_t %s[ROWS * COLS] = {\n  %s\n};" % (name, body))
    return "\n\n".join(rendered)


def render_shim(symbol_prefix, arch_directive=""):
    # The ELF path passes `.arch armv9-a+sme` because GNU as defaults to the
    # base architecture; the Mach-O path passes an empty directive and enables
    # SME with `clang -march=armv9-a+sme`.
    #
    # The shim clears TPIDR2_EL0 before the kernel call. A top-level kernel
    # with `new_za` owns a fresh ZA array and has no incoming ZA save buffer;
    # macOS starts with TPIDR2_EL0 zero, but QEMU/Linux does not, and the
    # kernel's prologue would otherwise call `__arm_tpidr2_save`. Making the
    # precondition explicit keeps the native and emulated paths identical.
    return arch_directive + r"""
	.text
	.p2align 2
	.globl	%(p)ssme_call
%(p)ssme_call:
	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	mov	x16, x0
	mov	x0, x1
	mov	x1, x2
	mov	x2, x3
	mov	x3, x4
	smstart	sm
	msr	tpidr2_el0, xzr
	blr	x16
	smstop	sm
	ldp	x29, x30, [sp], #16
	ret

	.globl	%(p)ssme_streaming_bytes
%(p)ssme_streaming_bytes:
	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	smstart	sm
	cntb	x0
	smstop	sm
	ldp	x29, x30, [sp], #16
	ret
""" % {"p": symbol_prefix}


# The streaming-compatible memory entry points are part of the platform's
# streaming ABI, not of the kernel under test: a `memref.copy` inside a
# streaming function lowers to a call to `__arm_sc_memcpy`. Apple's SDK does
# not export these symbols (Linux compiler-rt does), so both harness renderers
# supply scalar byte loops in their place. These are runner-provided streaming
# ABI shims, not the platform implementation.
#
# The pointers are volatile so the compiler cannot recognize the loops and
# rewrite them back into a libc `memcpy`/`memmove` call: libc's routines may use
# vector registers, which are not available in streaming mode. The loops
# themselves touch only scalar registers, so they are safe to call with
# PSTATE.SM set.
_STREAMING_ABI_SHIMS = r"""
void *__arm_sc_memcpy(void *dst, const void *src, size_t n) {
  volatile unsigned char *out = dst;
  const volatile unsigned char *in = src;
  for (size_t i = 0; i < n; ++i)
    out[i] = in[i];
  return dst;
}

void *__arm_sc_memmove(void *dst, const void *src, size_t n) {
  volatile unsigned char *out = dst;
  const volatile unsigned char *in = src;
  uintptr_t outAddress = (uintptr_t)dst;
  uintptr_t inAddress = (uintptr_t)src;
  if (outAddress <= inAddress) {
    for (size_t i = 0; i < n; ++i)
      out[i] = in[i];
  } else {
    for (size_t i = n; i > 0; --i)
      out[i - 1] = in[i - 1];
  }
  return dst;
}
"""

_HARNESS_HEAD = r"""
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

typedef void (*KernelFn)(MemRef2D *, MemRef2D *, MemRef2D *, MemRef2D *);
extern void %(wrapper)s(MemRef2D *, MemRef2D *, MemRef2D *, MemRef2D *);
extern void sme_call(KernelFn, MemRef2D *, MemRef2D *, MemRef2D *, MemRef2D *);
extern unsigned long sme_streaming_bytes(void);
%(shims)s

#define ROWS 16
#define COLS 16
#define CANARY 32
#define SENTINEL 0x5a5a5a5au

%(tables)s

static float a[ROWS * COLS] __attribute__((aligned(64)));
static float b[ROWS * COLS] __attribute__((aligned(64)));
static float init[ROWS * COLS] __attribute__((aligned(64)));
static float out[ROWS * COLS + CANARY] __attribute__((aligned(64)));

static void load(float *dst, const uint32_t *src) {
  for (size_t i = 0; i < ROWS * COLS; ++i) {
    uint32_t bits = src[i];
    float value;
    __builtin_memcpy(&value, &bits, sizeof(value));
    dst[i] = value;
  }
}

static int matches(const float *result, const uint32_t *expected) {
  for (size_t i = 0; i < ROWS * COLS; ++i) {
    uint32_t bits;
    __builtin_memcpy(&bits, &result[i], sizeof(bits));
    if (bits != expected[i])
      return 0;
  }
  return 1;
}

static int canary_holds(void) {
  for (size_t i = 0; i < CANARY; ++i) {
    uint32_t bits;
    __builtin_memcpy(&bits, &out[ROWS * COLS + i], sizeof(bits));
    if (bits != SENTINEL)
      return 0;
  }
  return 1;
}

static void sentinel_fill(void) {
  for (size_t i = 0; i < ROWS * COLS + CANARY; ++i)
    out[i] = 0.0f;
  for (size_t i = 0; i < ROWS * COLS + CANARY; ++i) {
    uint32_t bits = SENTINEL;
    __builtin_memcpy(&out[i], &bits, sizeof(bits));
  }
}

static MemRef2D descriptor(float *buffer) {
  MemRef2D result = {buffer, buffer, 0, ROWS, COLS, COLS, 1};
  return result;
}
"""


def render_native_harness(wrapper, without_shim=False):
    call = ("%s(&da, &db, &di, &dout);" % wrapper if without_shim
            else "sme_call(&%s, &da, &db, &di, &dout);" % wrapper)
    body = r"""
#include <sys/sysctl.h>
#include <unistd.h>

#define SAY(text) write(1, (text), sizeof(text) - 1)

static int host_has_sme(void) {
  int value = 0;
  size_t size = sizeof(value);
  if (sysctlbyname("hw.optional.arm.FEAT_SME", &value, &size, NULL, 0) != 0)
    return 0;
  return value != 0;
}

int main(void) {
  if (!host_has_sme()) {
    SAY("unsupported: no sme\n");
    return 70;
  }
  if (sme_streaming_bytes() != 64) {
    SAY("unsupported: svl is not 512\n");
    return 71;
  }
  MemRef2D da = descriptor(a), db = descriptor(b);
  MemRef2D di = descriptor(init), dout = descriptor(out);

  load(a, A1); load(b, B1); load(init, I1); sentinel_fill();
  %(call)s
  if (!matches(out, E1)) return 73;
  if (!canary_holds()) return 74;

  load(a, A2); load(b, B2); load(init, I2); sentinel_fill();
  %(call)s
  if (!matches(out, E2)) return 73;
  if (!canary_holds()) return 74;

  SAY("pass\n");
  return 0;
}
""" % {"call": call}
    return _HARNESS_HEAD % {"wrapper": wrapper,
                            "tables": render_tables(),
                            "shims": _STREAMING_ABI_SHIMS} + body


def emit_object(llk_compile, source_dir, work_dir, triple, features):
    source = ("%s/test/Conversion/MicroMapping/Inputs/arm/%s.mlir" %
              (source_dir, CASE[0]))
    obj = "%s/kernel.o" % work_dir
    identity = "%s/identity.json" % work_dir
    command = [
        llk_compile, "--mapping-target=arm-sme",
        "--mapping-root=" + source_dir,
        "--mapping-backend=selected-target", "--mapping-stop=lowered",
        "--emit=object", "--target-triple=" + triple,
        "--target-features=" + ",".join(features),
        "--output=" + obj, "--emit-identity=" + identity, source,
    ]
    result = subprocess.run(command, capture_output=True, text=True,
                            timeout=300)
    if result.returncode:
        raise SystemExit("object emission failed:\n%s\n%s" %
                         (result.stdout, result.stderr))
    with open(identity, encoding="utf-8") as handle:
        record = json.load(handle)
    return obj, record, "_mlir_ciface_" + CASE[1]


def check_object_format(path, platform):
    data = open(path, "rb").read()
    if platform == "darwin":
        if data[:4] != b"\xcf\xfa\xed\xfe":
            raise SystemExit("output is not a 64-bit Mach-O object")
        if struct.unpack_from("<I", data, 4)[0] != 0x0100000C:
            raise SystemExit("Mach-O object is not CPU_TYPE_ARM64")
    else:
        if data[:4] != b"\x7fELF":
            raise SystemExit("output is not an ELF object")
        if struct.unpack_from("<H", data, 18)[0] != 183:
            raise SystemExit("ELF object is not EM_AARCH64")


def check_identity(record, triple, wrapper, failures, label):
    if record.get("schemaVersion") != 2:
        failures.append("%s: identity schemaVersion is not 2" % label)
    if record.get("targetTriple") != triple:
        failures.append("%s: identity triple %r != %r" %
                        (label, record.get("targetTriple"), triple))
    if record.get("targetFeatures") != FEATURES:
        failures.append("%s: identity features %r != %r" %
                        (label, record.get("targetFeatures"), FEATURES))
    if record.get("streamingVectorLengthBits") != 512:
        failures.append("%s: identity streamingVectorLengthBits is %r" %
                        (label, record.get("streamingVectorLengthBits")))
    if record.get("invocationAttributes") != ["aarch64_pstate_sm_enabled"]:
        failures.append("%s: identity invocationAttributes is %r" %
                        (label, record.get("invocationAttributes")))
    if record.get("referenceGroupsLowered") != 0:
        failures.append("%s: selected lowering fell back (%r reference groups)" %
                        (label, record.get("referenceGroupsLowered")))
    if record.get("backendGroupsRealized", 0) <= 0:
        failures.append("%s: selected backend realized no groups" % label)
    if record.get("wrapperSymbol") != wrapper:
        failures.append("%s: identity wrapper %r != %r" %
                        (label, record.get("wrapperSymbol"), wrapper))
    for key in ("planId", "sourceGraphHash", "abiHash"):
        value = parse_hex_identity(record.get(key))
        if value is None or value == 0:
            failures.append("%s: identity %s is not a nonzero hex hash: %r" %
                            (label, key, record.get(key)))


def check_sme_instructions(disassembler, obj, failures, label,
                           extra_flags=()):
    # LLVM's disassembler needs `--mattr=+sme` to decode SME encodings; without
    # it they print as `.long 0x...`. GNU's aarch64 objdump decodes them by
    # default, so the QEMU driver passes no extra flags.
    result = subprocess.run([disassembler, "-d", *extra_flags, obj],
                            capture_output=True, text=True, timeout=120)
    if result.returncode:
        failures.append("%s: disassembly failed\n%s%s" %
                        (label, result.stdout, result.stderr))
        return
    if not has_sme_instructions(result.stdout):
        failures.append(
            "%s: object has no fmopa + zero {za} + smstart za sequence\n%s" %
            (label, result.stdout))


_QEMU_PREAMBLE = r"""
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

typedef void (*KernelFn)(MemRef2D *, MemRef2D *, MemRef2D *, MemRef2D *);
extern void %(wrapper)s(MemRef2D *, MemRef2D *, MemRef2D *, MemRef2D *);
extern void sme_call(KernelFn, MemRef2D *, MemRef2D *, MemRef2D *, MemRef2D *);
extern unsigned long sme_streaming_bytes(void);
%(shims)s

#define ROWS 16
#define COLS 16
#define CANARY 32
#define SENTINEL 0x5a5a5a5au

static unsigned char heap[1 << 20] __attribute__((aligned(64)));
static size_t used;

void *malloc(size_t n) {
  size_t start = (used + 63) & ~(size_t)63;
  if (start + n > sizeof(heap)) for (;;) { }
  used = start + n;
  return heap + start;
}
void *aligned_alloc(size_t alignment, size_t n) {
  size_t start = (used + alignment - 1) & ~(alignment - 1);
  if (start + n > sizeof(heap)) for (;;) { }
  used = start + n;
  return heap + start;
}
void free(void *p) { (void)p; }
void *memset(void *dst, int value, size_t n) {
  unsigned char *bytes = dst;
  for (size_t i = 0; i < n; ++i) bytes[i] = (unsigned char)value;
  return dst;
}
void *memcpy(void *dst, const void *src, size_t n) {
  unsigned char *out = dst;
  const unsigned char *in = src;
  for (size_t i = 0; i < n; ++i) out[i] = in[i];
  return dst;
}
static long syscall3(long, long, long, long);
void memrefCopy(int64_t element_size, void *source, void *destination) {
  (void)element_size; (void)source; (void)destination;
  syscall3(93, 75, 0, 0);
  for (;;) { }
}
/* new_za consults TPIDR2_EL0 and calls this only when it is nonzero. At the
   exported boundary it is zero; a call means the ABI assumption broke. */
void __arm_tpidr2_save(void) { syscall3(93, 76, 0, 0); for (;;) { } }
void __arm_tpidr2_restore(void) { syscall3(93, 76, 0, 0); for (;;) { } }

static long syscall3(long n, long a, long b, long c) {
  register long x8 __asm__("x8") = n;
  register long x0 __asm__("x0") = a;
  register long x1 __asm__("x1") = b;
  register long x2 __asm__("x2") = c;
  __asm__ volatile("svc #0" : "+r"(x0) : "r"(x8), "r"(x1), "r"(x2)
                   : "memory");
  return x0;
}
#define SAY(text) syscall3(64, 1, (long)(text), sizeof(text) - 1)
#define EXIT(code) do { syscall3(93, (code), 0, 0); for (;;) { } } while (0)

__asm__(".globl _start\n"
        "_start:\n"
        "  mov x0, sp\n"
        "  bl harness_main\n");

%(tables)s

static float a[ROWS * COLS] __attribute__((aligned(64)));
static float b[ROWS * COLS] __attribute__((aligned(64)));
static float init[ROWS * COLS] __attribute__((aligned(64)));
static float out[ROWS * COLS + CANARY] __attribute__((aligned(64)));

static void load(float *dst, const uint32_t *src) {
  for (size_t i = 0; i < ROWS * COLS; ++i) {
    uint32_t bits = src[i];
    float value;
    __builtin_memcpy(&value, &bits, sizeof(value));
    dst[i] = value;
  }
}
static int matches(const float *result, const uint32_t *expected) {
  for (size_t i = 0; i < ROWS * COLS; ++i) {
    uint32_t bits;
    __builtin_memcpy(&bits, &result[i], sizeof(bits));
    if (bits != expected[i]) return 0;
  }
  return 1;
}
static int canary_holds(void) {
  for (size_t i = 0; i < CANARY; ++i) {
    uint32_t bits;
    __builtin_memcpy(&bits, &out[ROWS * COLS + i], sizeof(bits));
    if (bits != SENTINEL) return 0;
  }
  return 1;
}
static void sentinel_fill(void) {
  for (size_t i = 0; i < ROWS * COLS + CANARY; ++i) {
    uint32_t bits = SENTINEL;
    __builtin_memcpy(&out[i], &bits, sizeof(bits));
  }
}
static MemRef2D descriptor(float *buffer) {
  MemRef2D result = {buffer, buffer, 0, ROWS, COLS, COLS, 1};
  return result;
}
"""

_QEMU_BODY = r"""
static int has_hwcap2_sme(unsigned long *stack) {
  unsigned long argc = stack[0];
  unsigned long *p = stack + 1 + argc + 1;
  while (*p) ++p;   /* envp */
  ++p;              /* envp NULL */
  for (; p[0]; p += 2)
    if (p[0] == 26) return (p[1] & (1ul << 23)) != 0;  /* AT_HWCAP2 */
  return 0;
}

void harness_main(unsigned long *stack) {
  if (!has_hwcap2_sme(stack)) {
    SAY("unsupported: no sme\n");
    EXIT(70);
  }
  if (sme_streaming_bytes() != 64) {
    SAY("unsupported: svl is not 512\n");
    EXIT(71);
  }
  MemRef2D da = descriptor(a), db = descriptor(b);
  MemRef2D di = descriptor(init), dout = descriptor(out);

  load(a, A1); load(b, B1); load(init, I1); sentinel_fill();
  %(call)s
  if (!matches(out, E1)) EXIT(73);
  if (!canary_holds()) EXIT(74);

  load(a, A2); load(b, B2); load(init, I2); sentinel_fill();
  %(call)s
  if (!matches(out, E2)) EXIT(73);
  if (!canary_holds()) EXIT(74);

  SAY("pass\n");
  EXIT(0);
}
"""


def render_qemu_harness(wrapper, without_shim=False):
    call = ("%s(&da, &db, &di, &dout);" % wrapper if without_shim
            else "sme_call(&%s, &da, &db, &di, &dout);" % wrapper)
    preamble = _QEMU_PREAMBLE % {"wrapper": wrapper,
                                 "tables": render_tables(),
                                 "shims": _STREAMING_ABI_SHIMS}
    return preamble + _QEMU_BODY % {"call": call}
