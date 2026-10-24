//===- Frontend.h - Versioned C ABI for frontend compilation ------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// LLKFrontendCAPI v1. The Python frontend loads this shared library through
// ctypes, so every layout below is part of a frozen ABI and every entry point
// reports failure as a status code plus an owned LLKError.
//
// Ownership rules:
//   * LLKExecutable is opaque and owned by the caller. Release it exactly once
//     with llk_frontend_release_v1. The handle is reference counted internally:
//     a release that lands while another call is in flight drops the owner's
//     reference, the in-flight call keeps the executable alive and finishes,
//     and the last reference frees it. Starting a call on a released handle is
//     undefined, and releasing twice is invalid.
//   * LLKError is owned by the caller and freed with llk_frontend_error_free_v1.
//     Freeing null is a no-op.
//   * Caller buffers are borrowed for the duration of a call only. The ABI
//     never retains a data pointer.
//   * LLKString points into native storage owned by its handle or error. It
//     stays valid until that owner is released and must not be freed by the
//     caller; copy it immediately if it must outlive the owner.
//
// Return convention: 0 on success, 1 on failure. On failure the output handle
// and string are set to null/zero and, when a non-null error slot was passed, an
// owned LLKError describes the failure. No C++ exception crosses this boundary.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CAPI_FRONTEND_H
#define LLK_CAPI_FRONTEND_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(LLK_FRONTEND_CAPI_EXPORTS)
#define LLK_FRONTEND_API __declspec(dllexport)
#elif defined(_WIN32)
#define LLK_FRONTEND_API __declspec(dllimport)
#else
#define LLK_FRONTEND_API __attribute__((visibility("default")))
#endif

/// Only this version exists. A caller compares it with its own expectation and
/// refuses to use a handle from a different layout.
#define LLK_FRONTEND_API_VERSION 1u

typedef struct LLKExecutable LLKExecutable;
typedef struct LLKError LLKError;

/// Borrowed UTF-8 bytes with an explicit length. NUL-terminated for logging,
/// but callers must use `length`, which may include interior NULs.
typedef struct LLKString {
  const char *data;
  size_t length;
} LLKString;

/// Element tags are fixed by the ABI and independent of the native enum's
/// ordinal values. An unknown tag is rejected, never treated as f32.
enum LLKElementType {
  LLK_ELEMENT_F32 = 1,
  LLK_ELEMENT_BF16 = 2,
  LLK_ELEMENT_F16 = 3,
  LLK_ELEMENT_I32 = 4,
  LLK_ELEMENT_I8 = 5
};

/// Mirrors llk::MemRef2D field for field, followed by the declared element tag
/// and allocation extent. `allocated` is the base the caller owns, `aligned` is
/// the first element of the view, and `allocation_bytes` counts bytes from
/// `allocated`. The initial contract accepts rank two, compact row-major,
/// zero-offset descriptors with disjoint outputs.
typedef struct LLKBuffer2D {
  void *allocated;
  void *aligned;
  int64_t offset;
  int64_t sizes[2];
  int64_t strides[2];
  uint32_t element_type;
  uint64_t allocation_bytes;
} LLKBuffer2D;

/// Return the ABI version. Always callable, including before any other entry.
LLK_FRONTEND_API uint32_t llk_frontend_api_version(void);

/// Parse, validate and compile one canonical frontend request. `json` is UTF-8
/// of `length` bytes and need not be NUL-terminated. On success `*out` owns a
/// native executable; on failure `*out` is null and `*error` (when non-null)
/// owns a diagnostic.
LLK_FRONTEND_API int llk_frontend_compile_v1(const char *json, size_t length,
                                             LLKExecutable **out,
                                             LLKError **error);

/// Invoke synchronously with caller-owned buffers. The call validates arity,
/// element type, extent, compact strides, offset, allocation bounds and output
/// disjointness before entering JIT code. Concurrent calls on one handle are
/// serialized; distinct output buffers are required for every call.
LLK_FRONTEND_API int
llk_frontend_invoke_v1(LLKExecutable *handle, const LLKBuffer2D *inputs,
                       size_t input_count, const LLKBuffer2D *outputs,
                       size_t output_count, LLKError **error);

/// Borrowed compilation report JSON, valid until the handle is released.
LLK_FRONTEND_API int llk_frontend_report_v1(LLKExecutable *handle,
                                            LLKString *out, LLKError **error);

/// Borrowed inspection text for `frontend`, `llk` or `micro`, valid until the
/// handle is released. Any other stage fails with `inspection_unavailable`
/// rather than returning empty text.
LLK_FRONTEND_API int llk_frontend_inspect_v1(LLKExecutable *handle,
                                             const char *stage, LLKString *out,
                                             LLKError **error);

/// Borrowed JSON diagnostic: `code`, `message`, `stage` and, when the native
/// error carries one, `file`, `line` and `column`. Valid until the error is
/// freed. A null error yields an empty string.
LLK_FRONTEND_API LLKString llk_frontend_error_json_v1(const LLKError *error);

/// Free an owned error. Null is a no-op; double free is invalid.
LLK_FRONTEND_API void llk_frontend_error_free_v1(LLKError *error);

/// Release an owned executable. Null is a no-op; double release and any call on
/// a released handle are invalid.
LLK_FRONTEND_API void llk_frontend_release_v1(LLKExecutable *handle);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // LLK_CAPI_FRONTEND_H
