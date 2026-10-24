//===- frontend.c - Plain C consumer of the LLK frontend C ABI ------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// The Python frontend reaches native code through ctypes, so this consumer
// exercises exactly that surface: no C++, no LLVM headers, and no knowledge of
// the native InvocationBuffer2D layout. The guard floats around every buffer
// catch any access outside the declared allocation.
//===----------------------------------------------------------------------===//

#include "LLK/CAPI/Frontend.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GUARD 2
#define ROWS 8
#define COLUMNS 16
#define ELEMENTS ((size_t)ROWS * (size_t)COLUMNS)
#define FLOATS (ELEMENTS + 2 * GUARD)
#define GUARD_VALUE -777.0f

typedef struct {
  float data[FLOATS];
} Storage;

static int failures = 0;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,         \
              #condition);                                                     \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

static char *read_fixture(size_t *length) {
  FILE *file = fopen(LLK_CAPI_FIXTURE, "rb");
  char *text = NULL;
  long size = 0;
  if (!file) {
    fprintf(stderr, "cannot open fixture %s\n", LLK_CAPI_FIXTURE);
    return NULL;
  }
  if (fseek(file, 0, SEEK_END) == 0 && (size = ftell(file)) > 0 &&
      fseek(file, 0, SEEK_SET) == 0) {
    text = malloc((size_t)size);
    if (text && fread(text, 1, (size_t)size, file) != (size_t)size) {
      free(text);
      text = NULL;
    }
  }
  fclose(file);
  if (text)
    *length = (size_t)size;
  return text;
}

static void fill(Storage *storage) {
  size_t i;
  for (i = 0; i < FLOATS; ++i)
    storage->data[i] = GUARD_VALUE;
}

// A compact row-major descriptor whose declared allocation is exactly the
// payload; the surrounding guards are outside it, so a stray write is visible
// here rather than silently attributed to the kernel's own storage.
static LLKBuffer2D described(Storage *storage) {
  LLKBuffer2D buffer;
  memset(&buffer, 0, sizeof(buffer));
  buffer.allocated = storage->data + GUARD;
  buffer.aligned = storage->data + GUARD;
  buffer.offset = 0;
  buffer.sizes[0] = ROWS;
  buffer.sizes[1] = COLUMNS;
  buffer.strides[0] = COLUMNS;
  buffer.strides[1] = 1;
  buffer.element_type = LLK_ELEMENT_F32;
  buffer.allocation_bytes = (uint64_t)(ELEMENTS * sizeof(float));
  return buffer;
}

static int contains(LLKString text, const char *needle) {
  size_t length = strlen(needle);
  size_t i;
  if (!text.data || text.length < length)
    return 0;
  for (i = 0; i + length <= text.length; ++i)
    if (memcmp(text.data + i, needle, length) == 0)
      return 1;
  return 0;
}

// One rejected invocation: the call reports an error, writes no handle, and
// carries a stable code the Python layer can translate.
static void expectRejected(LLKExecutable *handle, const LLKBuffer2D *inputs,
                           size_t inputCount, const LLKBuffer2D *outputs,
                           size_t outputCount, const char *code) {
  LLKError *error = (LLKError *)(uintptr_t)0x1;
  int status = llk_frontend_invoke_v1(handle, inputs, inputCount, outputs,
                                      outputCount, &error);
  CHECK(status == 1);
  CHECK(error != NULL && error != (LLKError *)(uintptr_t)0x1);
  if (error) {
    LLKString json = llk_frontend_error_json_v1(error);
    CHECK(json.data != NULL && json.length != 0);
    CHECK(contains(json, code));
    llk_frontend_error_free_v1(error);
  }
}

static void checkLayout(void) {
  CHECK(llk_frontend_api_version() == 1);
  CHECK(sizeof(float) == 4);
  CHECK(LLK_ELEMENT_F32 == 1);
  CHECK(LLK_ELEMENT_BF16 == 2);
  CHECK(LLK_ELEMENT_F16 == 3);
  CHECK(LLK_ELEMENT_I32 == 4);
  CHECK(LLK_ELEMENT_I8 == 5);
#if UINTPTR_MAX == UINT64_MAX
  // The Python ctypes Structure mirrors this field order exactly.
  CHECK(sizeof(void *) == 8);
  CHECK(offsetof(LLKBuffer2D, allocated) == 0);
  CHECK(offsetof(LLKBuffer2D, aligned) == 8);
  CHECK(offsetof(LLKBuffer2D, offset) == 16);
  CHECK(offsetof(LLKBuffer2D, sizes) == 24);
  CHECK(offsetof(LLKBuffer2D, strides) == 40);
  CHECK(offsetof(LLKBuffer2D, element_type) == 56);
  CHECK(offsetof(LLKBuffer2D, allocation_bytes) == 64);
  CHECK(sizeof(LLKBuffer2D) == 72);
  CHECK(sizeof(LLKString) == 16);
#endif
}

static void checkRejectedBuffers(LLKExecutable *handle) {
  Storage a, b, out;
  LLKBuffer2D inputs[2], outputs[1], bad;
  fill(&a);
  fill(&b);
  fill(&out);
  inputs[0] = described(&a);
  inputs[1] = described(&b);
  outputs[0] = described(&out);

  expectRejected(handle, inputs, 1, outputs, 1, "invalid_frontend_buffer");
  expectRejected(handle, inputs, 2, outputs, 2, "invalid_frontend_buffer");

  bad = inputs[0];
  bad.element_type = LLK_ELEMENT_BF16;
  inputs[0] = bad;
  expectRejected(handle, inputs, 2, outputs, 1, "invalid_frontend_buffer");
  inputs[0] = described(&a);

  bad = inputs[1];
  bad.sizes[0] = ROWS + 1;
  inputs[1] = bad;
  expectRejected(handle, inputs, 2, outputs, 1, "invalid_frontend_buffer");
  inputs[1] = described(&b);

  bad = outputs[0];
  bad.strides[0] = 1;
  bad.strides[1] = COLUMNS;
  outputs[0] = bad;
  expectRejected(handle, inputs, 2, outputs, 1, "invalid_frontend_buffer");
  outputs[0] = described(&out);

  bad = outputs[0];
  bad.offset = 4;
  outputs[0] = bad;
  expectRejected(handle, inputs, 2, outputs, 1, "invalid_frontend_buffer");
  outputs[0] = described(&out);

  bad = inputs[0];
  bad.allocation_bytes = (uint64_t)(ELEMENTS * sizeof(float)) - sizeof(float);
  inputs[0] = bad;
  expectRejected(handle, inputs, 2, outputs, 1, "invalid_frontend_buffer");
  inputs[0] = described(&a);

  // The output aliases the first input: writes must never land in an input.
  expectRejected(handle, inputs, 2, inputs, 1, "invalid_frontend_buffer");
}

static void checkExecution(LLKExecutable *handle) {
  Storage a, b, out;
  LLKBuffer2D inputs[2], outputs[1];
  LLKError *error = NULL;
  size_t i;

  fill(&a);
  fill(&b);
  fill(&out);
  for (i = 0; i < ELEMENTS; ++i) {
    a.data[GUARD + i] = (float)(i + 1);
    b.data[GUARD + i] = (float)(257 + 2 * i);
  }
  inputs[0] = described(&a);
  inputs[1] = described(&b);
  outputs[0] = described(&out);

  CHECK(llk_frontend_invoke_v1(handle, inputs, 2, outputs, 1, &error) == 0);
  CHECK(error == NULL);
  for (i = 0; i < ELEMENTS; ++i)
    CHECK(out.data[GUARD + i] == (float)(258 + 3 * i));
  for (i = 0; i < ELEMENTS; ++i) {
    CHECK(a.data[GUARD + i] == (float)(i + 1));
    CHECK(b.data[GUARD + i] == (float)(257 + 2 * i));
  }
  for (i = 0; i < GUARD; ++i) {
    CHECK(a.data[i] == GUARD_VALUE &&
          a.data[GUARD + ELEMENTS + i] == GUARD_VALUE);
    CHECK(b.data[i] == GUARD_VALUE &&
          b.data[GUARD + ELEMENTS + i] == GUARD_VALUE);
    CHECK(out.data[i] == GUARD_VALUE &&
          out.data[GUARD + ELEMENTS + i] == GUARD_VALUE);
  }
}

int main(void) {
  char *text;
  size_t length = 0;
  int round;
  LLKExecutable *handle = NULL;
  LLKError *error = (LLKError *)(uintptr_t)0x1;
  LLKString report;
  LLKString micro;

  checkLayout();

  text = read_fixture(&length);
  CHECK(text != NULL);
  if (!text)
    return 1;

  // Malformed input fails with a structured schema error and a null handle.
  CHECK(llk_frontend_compile_v1(text, length / 2, &handle, &error) == 1);
  CHECK(handle == NULL);
  CHECK(error != NULL && error != (LLKError *)(uintptr_t)0x1);
  if (error) {
    LLKString json = llk_frontend_error_json_v1(error);
    CHECK(json.data != NULL && json.length != 0);
    CHECK(contains(json, "invalid_frontend_schema"));
    llk_frontend_error_free_v1(error);
    error = NULL;
  }

  // Repeated compile/invoke/release: every round owns its own handle and
  // produces the same literal results.
  for (round = 0; round < 3; ++round) {
    handle = (LLKExecutable *)(uintptr_t)0x1;
    error = (LLKError *)(uintptr_t)0x1;
    CHECK(llk_frontend_compile_v1(text, length, &handle, &error) == 0);
    CHECK(handle != NULL && handle != (LLKExecutable *)(uintptr_t)0x1);
    CHECK(error == NULL);
    if (!handle)
      break;

    report.data = NULL;
    report.length = 0;
    CHECK(llk_frontend_report_v1(handle, &report, &error) == 0);
    CHECK(report.data != NULL && report.length != 0);
    CHECK(contains(report, "\"backend\":\"reference\""));
    CHECK(contains(report, "\"source_adapter\":\"linalg-elementwise-v1\""));

    micro.data = NULL;
    micro.length = 0;
    CHECK(llk_frontend_inspect_v1(handle, "micro", &micro, &error) == 0);
    CHECK(micro.data != NULL && micro.length != 0);

    // A stage this frontend cannot produce is an explicit error, not empty text.
    micro.data = NULL;
    micro.length = 0;
    CHECK(llk_frontend_inspect_v1(handle, "asm", &micro, &error) == 1);
    CHECK(error != NULL);
    if (error) {
      LLKString json = llk_frontend_error_json_v1(error);
      CHECK(contains(json, "inspection_unavailable"));
      llk_frontend_error_free_v1(error);
      error = NULL;
    }

    checkRejectedBuffers(handle);
    checkExecution(handle);
    llk_frontend_release_v1(handle);
    handle = NULL;
  }

  // Releasing nothing is a no-op; freeing no error is a no-op.
  llk_frontend_release_v1(NULL);
  llk_frontend_error_free_v1(NULL);

  free(text);
  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("frontend C ABI consumer passed\n");
  return 0;
}
