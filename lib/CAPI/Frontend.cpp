//===- Frontend.cpp - Versioned C ABI for frontend compilation ----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is the whole C boundary. It owns three things the native service
// does not: the opaque handle, the owned error object, and the translation
// between LLKBuffer2D and the checked InvocationBuffer2D contract. Validation
// itself lives in MappedExecutable, so a buffer that passes here is checked by
// the same code path as a C++ caller.
//
//===----------------------------------------------------------------------===//

#include "LLK/CAPI/Frontend.h"

#include "LLK/Frontend/Compiler.h"
#include "LLK/Frontend/Program.h"
#include "LLK/Runtime/MappedInvocation.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <cstddef>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {
using llk::frontend::Diagnostic;
using llk::frontend::FrontendCompilation;
using llk::frontend::FrontendError;

llvm::Error apiError(llvm::StringRef code, llvm::Twine message) {
  return llk::frontend::frontendError(code, message, {}, "capi");
}

std::optional<llk::InvocationElementType> elementType(uint32_t tag) {
  switch (tag) {
  case LLK_ELEMENT_F32:
    return llk::InvocationElementType::F32;
  case LLK_ELEMENT_BF16:
    return llk::InvocationElementType::BF16;
  case LLK_ELEMENT_F16:
    return llk::InvocationElementType::F16;
  case LLK_ELEMENT_I32:
    return llk::InvocationElementType::I32;
  case LLK_ELEMENT_I8:
    return llk::InvocationElementType::I8;
  }
  return std::nullopt;
}

/// Translate public descriptors into the native checked form. Only the tag
/// mapping happens here; arity, dtype, extent, stride, offset, bounds and
/// disjointness stay with validateMappedInvocation.
llvm::Expected<std::vector<llk::InvocationBuffer2D>>
marshal(const LLKBuffer2D *buffers, size_t count, llvm::StringRef kind) {
  std::vector<llk::InvocationBuffer2D> result;
  if (count == 0)
    return result;
  if (!buffers)
    return apiError("invalid_frontend_buffer",
                    kind + " buffer array is null with a nonzero count");
  result.reserve(count);
  for (size_t i = 0; i != count; ++i) {
    const LLKBuffer2D &buffer = buffers[i];
    std::optional<llk::InvocationElementType> type =
        elementType(buffer.element_type);
    if (!type)
      return apiError("invalid_frontend_buffer",
                      kind + " " + llvm::Twine(i) +
                          " has an unsupported element tag");
    result.push_back({MemRef2D{buffer.allocated, buffer.aligned, buffer.offset,
                               buffer.sizes[0], buffer.sizes[1],
                               buffer.strides[0], buffer.strides[1]},
                      *type, buffer.allocation_bytes});
  }
  return result;
}
} // namespace

/// Owned native diagnostic. `json` is the serialized form handed out by
/// llk_frontend_error_json_v1 and outlives every borrowed string view into it.
struct LLKError {
  explicit LLKError(Diagnostic diagnostic)
      : diagnostic(std::move(diagnostic)), json(serialize(this->diagnostic)) {}

  static std::string serialize(const Diagnostic &diagnostic) {
    llvm::json::Object object{{"code", diagnostic.code},
                              {"message", diagnostic.message},
                              {"stage", diagnostic.stage}};
    if (diagnostic.span) {
      object["file"] = diagnostic.span->file;
      object["line"] = diagnostic.span->line;
      object["column"] = diagnostic.span->column;
    }
    return llvm::formatv("{0}", llvm::json::Value(std::move(object))).str();
  }

  Diagnostic diagnostic;
  std::string json;
};

/// Opaque executable handle. `references` counts the caller's ownership plus
/// every call in flight, so a release that lands while an invocation is running
/// drops the owner's reference instead of deleting the handle under it. A call
/// that begins after the last reference is gone is still invalid: the caller's
/// ownership rule covers that window.
struct LLKExecutable {
  std::shared_ptr<FrontendCompilation> compilation;
  std::mutex lock;
  size_t references = 1;
};

namespace {
LLKError *translate(llvm::Error error) {
  if (!error)
    return nullptr;
  LLKError *result = nullptr;
  llvm::handleAllErrors(
      std::move(error),
      [&](const FrontendError &frontend) {
        result = new LLKError(frontend.diagnostic);
      },
      [&](const llvm::ErrorInfoBase &native) {
        result = new LLKError(Diagnostic{
            "frontend_native_error", native.message(), std::nullopt, "native"});
      });
  return result;
}

/// Store an owned error in the caller's slot. A caller that passes no slot
/// still gets the status code; the diagnostic is dropped rather than leaked.
void report(LLKError **slot, llvm::Error error) {
  LLKError *value = translate(std::move(error));
  if (slot)
    *slot = value;
  else
    delete value;
}

void clear(LLKError **slot) {
  if (slot)
    *slot = nullptr;
}

/// Drop one reference, deleting the handle with the last one. The delete happens
/// after the lock is released, so no thread ever unlocks freed storage.
void dropReference(LLKExecutable *handle) {
  bool dead = false;
  {
    std::lock_guard<std::mutex> guard(handle->lock);
    dead = --handle->references == 0;
  }
  if (dead)
    delete handle;
}

/// One strong reference for the duration of a C entry point. Invocation keeps
/// the compilation alive even if another thread releases the handle meanwhile.
class HandleReference {
public:
  explicit HandleReference(LLKExecutable *handle) : handle(handle) {
    if (!handle)
      return;
    std::lock_guard<std::mutex> guard(handle->lock);
    ++handle->references;
    compilation = handle->compilation;
  }
  ~HandleReference() {
    if (handle)
      dropReference(handle);
  }
  HandleReference(const HandleReference &) = delete;
  HandleReference &operator=(const HandleReference &) = delete;

  explicit operator bool() const { return compilation != nullptr; }
  FrontendCompilation *operator->() const { return compilation.get(); }

private:
  LLKExecutable *handle = nullptr;
  std::shared_ptr<FrontendCompilation> compilation;
};

llvm::Error releasedHandle() {
  return apiError("invalid_frontend_handle",
                  "executable has been released or is null");
}

/// Every entry point reports through llvm::Error, so no exception is expected.
/// A build with exceptions enabled still cannot let one cross into the caller,
/// which would unwind through foreign frames. LLVM is usually built with
/// -fno-exceptions, where this guard compiles to the plain call.
template <typename Body> int guarded(LLKError **slot, Body &&body) {
#if defined(__cpp_exceptions)
  try {
    return body();
  } catch (const std::exception &failure) {
    report(slot, apiError("frontend_native_error", failure.what()));
    return 1;
  } catch (...) {
    report(slot, apiError("frontend_native_error",
                          "unknown native exception at the C boundary"));
    return 1;
  }
#else
  (void)slot;
  return body();
#endif
}
} // namespace

extern "C" {

uint32_t llk_frontend_api_version(void) { return LLK_FRONTEND_API_VERSION; }

int llk_frontend_compile_v1(const char *json, size_t length,
                            LLKExecutable **out, LLKError **error) {
  if (!out)
    return 1;
  *out = nullptr;
  clear(error);
  return guarded(error, [&]() -> int {
    if (!json && length != 0) {
      report(error, apiError("invalid_frontend_argument",
                             "request text is null with a nonzero length"));
      return 1;
    }
    llvm::StringRef text =
        json ? llvm::StringRef(json, length) : llvm::StringRef();
    auto request = llk::frontend::parseRequest(text);
    if (!request) {
      report(error, request.takeError());
      return 1;
    }
    auto compiled = llk::frontend::compileFrontend(*request);
    if (!compiled) {
      report(error, compiled.takeError());
      return 1;
    }
    // A nothrow handle reports OOM instead of aborting, and ownership stays in
    // the local guard until the handle is fully constructed.
    std::unique_ptr<LLKExecutable> handle(new (std::nothrow) LLKExecutable);
    if (!handle) {
      report(error, apiError("frontend_out_of_memory",
                             "executable handle allocation failed"));
      return 1;
    }
    handle->compilation =
        std::make_shared<FrontendCompilation>(std::move(*compiled));
    *out = handle.release();
    return 0;
  });
}

int llk_frontend_invoke_v1(LLKExecutable *handle, const LLKBuffer2D *inputs,
                           size_t input_count, const LLKBuffer2D *outputs,
                           size_t output_count, LLKError **error) {
  clear(error);
  return guarded(error, [&]() -> int {
    HandleReference compilation(handle);
    if (!compilation) {
      report(error, releasedHandle());
      return 1;
    }
    auto nativeInputs = marshal(inputs, input_count, "input");
    if (!nativeInputs) {
      report(error, nativeInputs.takeError());
      return 1;
    }
    auto nativeOutputs = marshal(outputs, output_count, "output");
    if (!nativeOutputs) {
      report(error, nativeOutputs.takeError());
      return 1;
    }
    if (llvm::Error failure =
            compilation->invoke(*nativeInputs, *nativeOutputs)) {
      report(error, std::move(failure));
      return 1;
    }
    return 0;
  });
}

int llk_frontend_report_v1(LLKExecutable *handle, LLKString *out,
                           LLKError **error) {
  clear(error);
  if (!out) {
    report(error,
           apiError("invalid_frontend_argument", "report output is null"));
    return 1;
  }
  out->data = nullptr;
  out->length = 0;
  return guarded(error, [&]() -> int {
    HandleReference compilation(handle);
    if (!compilation) {
      report(error, releasedHandle());
      return 1;
    }
    llvm::StringRef text = compilation->report();
    out->data = text.data();
    out->length = text.size();
    return 0;
  });
}

int llk_frontend_inspect_v1(LLKExecutable *handle, const char *stage,
                            LLKString *out, LLKError **error) {
  clear(error);
  if (!out) {
    report(error,
           apiError("invalid_frontend_argument", "inspection output is null"));
    return 1;
  }
  out->data = nullptr;
  out->length = 0;
  if (!stage) {
    report(error,
           apiError("invalid_frontend_argument", "inspection stage is null"));
    return 1;
  }
  return guarded(error, [&]() -> int {
    HandleReference compilation(handle);
    if (!compilation) {
      report(error, releasedHandle());
      return 1;
    }
    auto text = compilation->inspect(stage);
    if (!text) {
      report(error, text.takeError());
      return 1;
    }
    out->data = text->data();
    out->length = text->size();
    return 0;
  });
}

LLKString llk_frontend_error_json_v1(const LLKError *error) {
  if (!error)
    return LLKString{nullptr, 0};
  return LLKString{error->json.data(), error->json.size()};
}

void llk_frontend_error_free_v1(LLKError *error) { delete error; }

void llk_frontend_release_v1(LLKExecutable *handle) {
  if (handle)
    dropReference(handle);
}

} // extern "C"
