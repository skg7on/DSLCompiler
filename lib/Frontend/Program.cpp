//===- Program.cpp - Strict schema-v1 parsing and typed validation --------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "LLK/Frontend/Program.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <set>

namespace llk::frontend {
char FrontendError::ID;
void FrontendError::log(llvm::raw_ostream &os) const {
  os << diagnostic.code << ": " << diagnostic.message;
  if (diagnostic.span)
    os << " at " << diagnostic.span->file << ':' << diagnostic.span->line << ':'
       << diagnostic.span->column;
}
std::error_code FrontendError::convertToErrorCode() const {
  return llvm::inconvertibleErrorCode();
}
llvm::Error frontendError(llvm::StringRef code, llvm::Twine message,
                          std::optional<SourceSpan> span,
                          llvm::StringRef stage) {
  return llvm::make_error<FrontendError>(
      Diagnostic{code.str(), message.str(), std::move(span), stage.str()});
}
namespace {
using JSON = llvm::json::Value;
using Object = llvm::json::Object;
using Array = llvm::json::Array;
constexpr int64_t maxIndex = std::numeric_limits<int64_t>::max();
constexpr llvm::StringLiteral storageTypes[] = {"f32", "bf16", "f16", "i32",
                                                "i8"};
bool storageType(llvm::StringRef type) {
  return llvm::is_contained(storageTypes, type);
}
bool numericType(llvm::StringRef type) {
  return storageType(type) || type == "index" || type == "i1";
}
bool valueType(llvm::StringRef type) {
  return numericType(type) ||
         (type.starts_with("ptr<") && type.ends_with(">") &&
          storageType(type.drop_front(4).drop_back()));
}
bool identifier(llvm::StringRef name) {
  auto first = [](char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
  };
  return !name.empty() && first(name.front()) &&
         llvm::all_of(name.drop_front(), [&](char c) {
           return first(c) || (c >= '0' && c <= '9');
         });
}
/// LLVM's integer accessor accepts 1.0 and its parser overwrites duplicate keys.
/// Keep lexical number kinds while using LLVM JSON for actual JSON decoding.
class Reader {
public:
  explicit Reader(llvm::StringRef text) : text(text) {}
  std::optional<Diagnostic> diagnostic;
  std::optional<SourceSpan> span;
  std::set<const JSON *> floating;
  bool bad(llvm::Twine message) {
    if (!diagnostic)
      diagnostic =
          Diagnostic{"invalid_frontend_schema", message.str(), span, "parse"};
    return false;
  }
  bool bounded() {
    if (text.size() > 16 * 1024 * 1024)
      return bad("request exceeds 16 MiB");
    unsigned depth = 0;
    bool quoted = false, escaped = false;
    for (char c : text) {
      if (quoted) {
        if (escaped)
          escaped = false;
        else if (c == '\\')
          escaped = true;
        else if (c == '"')
          quoted = false;
      } else if (c == '"')
        quoted = true;
      else if (c == '{' || c == '[') {
        if (++depth > 64)
          return bad("JSON nesting exceeds 64");
      } else if (c == '}' || c == ']') {
        if (!depth)
          return bad("unbalanced JSON");
        --depth;
      }
    }
    return true;
  }
  bool scan(const JSON &value) {
    whitespace();
    if (position >= text.size())
      return bad("truncated JSON");
    char c = text[position];
    if (c == '{') {
      const auto *object = value.getAsObject();
      if (!object)
        return bad("duplicate or inconsistent JSON object");
      ++position;
      whitespace();
      std::set<std::string> keys;
      if (peek('}')) {
        ++position;
        return true;
      }
      while (position < text.size()) {
        size_t start = position;
        if (!stringToken())
          return false;
        auto key = llvm::json::parse(text.slice(start, position));
        if (!key)
          return bad(llvm::toString(key.takeError()));
        auto spelling = key->getAsString();
        if (!spelling)
          return bad("expected object key");
        if (!keys.insert(spelling->str()).second)
          return bad("duplicate JSON key");
        whitespace();
        if (!peek(':'))
          return bad("expected colon");
        ++position;
        const auto *item = object->get(*spelling);
        if (!item || !scan(*item))
          return false;
        whitespace();
        if (peek('}')) {
          ++position;
          return true;
        }
        if (!peek(','))
          return bad("expected object separator");
        ++position;
        whitespace();
      }
      return bad("truncated object");
    }
    if (c == '[') {
      const auto *array = value.getAsArray();
      if (!array)
        return bad("inconsistent JSON array");
      ++position;
      whitespace();
      if (peek(']')) {
        ++position;
        return true;
      }
      size_t index = 0;
      while (position < text.size()) {
        if (index >= array->size() || !scan((*array)[index++]))
          return false;
        whitespace();
        if (peek(']')) {
          ++position;
          return true;
        }
        if (!peek(','))
          return bad("expected array separator");
        ++position;
      }
      return bad("truncated array");
    }
    if (c == '"')
      return stringToken();
    size_t start = position;
    while (position < text.size() && text[position] != ',' &&
           text[position] != '}' && text[position] != ']' &&
           !std::isspace(static_cast<unsigned char>(text[position])))
      ++position;
    auto token = text.slice(start, position);
    if (value.kind() == JSON::Number) {
      if (!jsonNumber(token))
        return bad("invalid JSON number spelling");
      auto number = value.getAsNumber();
      if (!number || !std::isfinite(*number))
        return bad("non-finite JSON number");
      if (token.find_first_of(".eE") != llvm::StringRef::npos)
        floating.insert(&value);
      else {
        int64_t integer;
        if (token.getAsInteger(10, integer))
          return bad("integer exceeds signed 64-bit range");
      }
    }
    return true;
  }
  const Object *record(const JSON *value,
                       std::initializer_list<llvm::StringRef> fields) {
    const auto *object = value ? value->getAsObject() : nullptr;
    if (!object) {
      bad("expected JSON object");
      return nullptr;
    }
    if (object->size() != fields.size()) {
      bad("unknown or missing record fields");
      return nullptr;
    }
    for (auto name : fields)
      if (!object->get(name)) {
        bad("missing record field " + name);
        return nullptr;
      }
    return object;
  }
  bool string(const JSON *value, std::string &out, bool empty = false) {
    auto s = value ? value->getAsString() : std::nullopt;
    if (!s || (!empty && s->empty()) || s->contains('\0'))
      return bad("expected text without NUL");
    out = s->str();
    return true;
  }
  bool integer(const JSON *value, int64_t &out,
               int64_t minimum = std::numeric_limits<int64_t>::min()) {
    if (!value || floating.count(value))
      return bad("expected integer token");
    auto n = value->getAsInteger();
    if (!n || *n < minimum)
      return bad("invalid signed 64-bit integer");
    out = *n;
    return true;
  }
  bool boolean(const JSON *value, bool &out) {
    auto b = value ? value->getAsBoolean() : std::nullopt;
    if (!b)
      return bad("expected boolean");
    out = *b;
    return true;
  }
  bool shape(const JSON *value, std::vector<int64_t> &out, int rank = -1,
             int maxRank = 2) {
    const auto *array = value ? value->getAsArray() : nullptr;
    if (!array || (rank >= 0 && array->size() != unsigned(rank)) ||
        array->size() > unsigned(maxRank))
      return bad("invalid shape rank");
    int64_t product = 1;
    for (const auto &item : *array) {
      int64_t dim;
      if (!integer(&item, dim, 1))
        return false;
      if (product > maxIndex / dim)
        return bad("shape product overflows");
      product *= dim;
      out.push_back(dim);
    }
    return true;
  }
  bool literal(const JSON &value) {
    if (value.getAsObject())
      return bad("constants may not contain objects");
    if (const auto *array = value.getAsArray())
      for (const auto &item : *array)
        if (!literal(item))
          return false;
    return true;
  }
  bool constants(const JSON *value, Object &out) {
    const auto *object = value ? value->getAsObject() : nullptr;
    if (!object)
      return bad("constants must be an object");
    for (const auto &item : *object)
      if (!identifier(item.first.str()) || !literal(item.second))
        return bad("invalid literal constant");
    out = *object;
    return true;
  }
  void canonical(const JSON &value, llvm::raw_ostream &os) const {
    if (const auto *object = value.getAsObject()) {
      std::vector<std::string> keys;
      for (const auto &item : *object)
        keys.push_back(item.first.str());
      llvm::sort(keys);
      os << '{';
      bool first = true;
      for (auto key : keys) {
        if (!first)
          os << ',';
        first = false;
        os << llvm::formatv("{0}", JSON(key)) << ':';
        canonical(*object->get(key), os);
      }
      os << '}';
    } else if (const auto *array = value.getAsArray()) {
      os << '[';
      for (size_t i = 0; i < array->size(); ++i) {
        if (i)
          os << ',';
        canonical((*array)[i], os);
      }
      os << ']';
    } else {
      auto rendered = llvm::formatv("{0}", value).str();
      os << rendered;
      if (floating.count(&value) &&
          rendered.find_first_of(".eE") == std::string::npos)
        os << ".0";
    }
  }

private:
  llvm::StringRef text;
  size_t position = 0;
  void whitespace() {
    while (position < text.size() &&
           std::isspace(static_cast<unsigned char>(text[position])))
      ++position;
  }
  bool peek(char c) const {
    return position < text.size() && text[position] == c;
  }
  static bool jsonNumber(llvm::StringRef token) {
    size_t i = 0;
    if (i < token.size() && token[i] == '-')
      ++i;
    if (i == token.size())
      return false;
    auto digit = [](char c) { return c >= '0' && c <= '9'; };
    if (token[i] == '0')
      ++i;
    else {
      if (token[i] < '1' || token[i] > '9')
        return false;
      while (i < token.size() && digit(token[i]))
        ++i;
    }
    if (i < token.size() && token[i] == '.') {
      size_t start = ++i;
      while (i < token.size() && digit(token[i]))
        ++i;
      if (i == start)
        return false;
    }
    if (i < token.size() && (token[i] == 'e' || token[i] == 'E')) {
      ++i;
      if (i < token.size() && (token[i] == '+' || token[i] == '-'))
        ++i;
      size_t start = i;
      while (i < token.size() && digit(token[i]))
        ++i;
      if (i == start)
        return false;
    }
    return i == token.size();
  }
  bool stringToken() {
    if (!peek('"'))
      return bad("expected string");
    ++position;
    while (position < text.size()) {
      char c = text[position++];
      if (c == '"')
        return true;
      if (c == '\\' && position < text.size())
        ++position;
    }
    return bad("truncated string");
  }
};

bool readSpan(Reader &r, const JSON *value, SourceSpan &out) {
  auto *o = r.record(value, {"file", "line", "column"});
  return o && r.string(o->get("file"), out.file) &&
         r.integer(o->get("line"), out.line, 1) &&
         r.integer(o->get("column"), out.column, 0);
}
bool readValue(Reader &r, const JSON *value, FrontendValue &out) {
  auto *o = r.record(value, {"id", "dtype", "shape"});
  if (!o || !r.string(o->get("id"), out.id) ||
      !r.string(o->get("dtype"), out.dtype) ||
      !r.shape(o->get("shape"), out.shape))
    return false;
  return (identifier(out.id) && valueType(out.dtype)) ||
         r.bad("invalid value identifier or dtype");
}
bool readNumerical(Reader &r, const JSON *value, NumericalContract &out) {
  auto *o =
      r.record(value, {"storage_dtype", "compute_dtype", "accumulator_dtype",
                       "cast_sites", "rounding", "activation_expression",
                       "activation_approximation", "reduction_order",
                       "reassociation", "fma", "subnormal"});
  if (!o || !r.string(o->get("storage_dtype"), out.storageDtype) ||
      !r.string(o->get("compute_dtype"), out.computeDtype) ||
      !r.string(o->get("accumulator_dtype"), out.accumulatorDtype) ||
      !r.string(o->get("rounding"), out.rounding) ||
      !r.string(o->get("activation_expression"), out.activationExpression) ||
      !r.string(o->get("activation_approximation"),
                out.activationApproximation) ||
      !r.string(o->get("reduction_order"), out.reductionOrder) ||
      !r.string(o->get("subnormal"), out.subnormal) ||
      !r.boolean(o->get("reassociation"), out.reassociation) ||
      !r.boolean(o->get("fma"), out.fma))
    return false;
  if (!storageType(out.storageDtype) || !storageType(out.computeDtype) ||
      !storageType(out.accumulatorDtype) || out.rounding != "rtne" ||
      (out.subnormal != "preserve" && out.subnormal != "flush-to-zero") ||
      (out.activationExpression != "none" &&
       out.activationExpression != "x/(1+exp(-x))") ||
      (out.activationApproximation != "none" &&
       out.activationApproximation != "exact" &&
       out.activationApproximation != "native") ||
      (out.reductionOrder != "source" && out.reductionOrder != "unspecified"))
    return r.bad("unsupported numerical contract");
  const auto *sites = o->getArray("cast_sites");
  if (!sites)
    return r.bad("cast_sites must be an array");
  for (const auto &site : *sites) {
    auto *c = r.record(
        &site, {"value_id", "source_dtype", "target_dtype", "rounding"});
    CastSite cast;
    if (!c || !r.string(c->get("value_id"), cast.valueId) ||
        !r.string(c->get("source_dtype"), cast.sourceDtype) ||
        !r.string(c->get("target_dtype"), cast.targetDtype) ||
        !r.string(c->get("rounding"), cast.rounding))
      return false;
    if (!identifier(cast.valueId) || !numericType(cast.sourceDtype) ||
        !numericType(cast.targetDtype) || cast.rounding != "rtne")
      return r.bad("invalid cast metadata");
    out.castSites.push_back(std::move(cast));
  }
  out.metadata = *o;
  return true;
}
struct OpSchema {
  int operands;
  int results;
  std::vector<std::string> attributes;
};
const std::map<std::string, OpSchema> &schemas() {
  static const std::map<std::string, OpSchema> all = {
      {"constant", {0, 1, {"value"}}},
      {"program_id", {0, 1, {"axis"}}},
      {"range", {0, 1, {"start", "end"}}},
      {"address", {3, 1, {}}},
      {"load", {3, 1, {}}},
      {"store", {3, 0, {}}},
      {"add", {2, 1, {}}},
      {"sub", {2, 1, {}}},
      {"neg", {1, 1, {}}},
      {"mul", {2, 1, {}}},
      {"div", {2, 1, {}}},
      {"rem", {2, 1, {}}},
      {"ceildiv", {2, 1, {}}},
      {"compare", {2, 1, {"predicate"}}},
      {"select", {3, 1, {}}},
      {"broadcast", {1, 1, {"dimensions"}}},
      {"expand_dims", {1, 1, {"axis"}}},
      {"cast", {1, 1, {"rounding"}}},
      {"dot", {3, 1, {}}},
      {"silu", {1, 1, {"approximation"}}},
      {"return", {0, 0, {}}},
      {"yield", {-1, 0, {}}},
      {"counted_loop", {-1, -1, {"step"}}}};
  return all;
}
bool readOperation(Reader &r, const JSON *value, FrontendOperation &out) {
  auto *o = r.record(value, {"opcode", "operands", "results", "attributes",
                             "regions", "span"});
  if (!o || !readSpan(r, o->get("span"), out.span))
    return false;
  auto previousSpan = r.span;
  r.span = out.span;
  if (!r.string(o->get("opcode"), out.opcode))
    return false;
  auto found = schemas().find(out.opcode);
  if (found == schemas().end())
    return r.bad("unsupported frontend opcode");
  const auto &schema = found->second;
  const auto *operands = o->getArray("operands");
  const auto *results = o->getArray("results");
  const auto *regions = o->getArray("regions");
  const auto *attrs = o->getObject("attributes");
  if (!operands || !results || !regions || !attrs ||
      (schema.operands >= 0 && operands->size() != unsigned(schema.operands)) ||
      (schema.results >= 0 && results->size() != unsigned(schema.results)) ||
      attrs->size() != schema.attributes.size())
    return r.bad("incorrect opcode fields/arity");
  for (const auto &name : schema.attributes)
    if (!attrs->get(name))
      return r.bad("unknown or missing operation attribute");
  for (const auto &item : *operands) {
    std::string id;
    if (!r.string(&item, id) || !identifier(id))
      return r.bad("invalid operand id");
    out.operands.push_back(std::move(id));
  }
  for (const auto &item : *results) {
    FrontendValue result;
    if (!readValue(r, &item, result))
      return false;
    out.results.push_back(std::move(result));
  }
  for (const auto &item : *attrs) {
    auto name = item.first.str();
    const auto &attribute = item.second;
    if (name == "axis" || name == "start" || name == "end" || name == "step") {
      int64_t number;
      if (!r.integer(&attribute, number))
        return false;
    } else if (name == "dimensions") {
      const auto *array = attribute.getAsArray();
      if (!array)
        return r.bad("broadcast dimensions must be an array");
      for (const auto &dim : *array) {
        int64_t index;
        if (!r.integer(&dim, index, 0))
          return false;
      }
    } else if (name == "value") {
      if (out.results.size() != 1)
        return r.bad("constant requires one result");
      const auto &dtype = out.results[0].dtype;
      if (dtype == "i1") {
        bool flag;
        if (!r.boolean(&attribute, flag))
          return false;
      } else if (dtype == "index" || dtype == "i32" || dtype == "i8") {
        int64_t number;
        if (!r.integer(&attribute, number))
          return false;
      } else if (!attribute.getAsNumber())
        return r.bad("constant requires a number");
    } else {
      std::string spelling;
      if (!r.string(&attribute, spelling))
        return false;
    }
  }
  out.attributes = *attrs;
  if ((out.opcode == "counted_loop" && regions->size() != 1) ||
      (out.opcode != "counted_loop" && !regions->empty()))
    return r.bad("only counted_loop can have a region");
  for (const auto &region : *regions) {
    const auto *body = r.record(&region, {"arguments", "operations"});
    if (!body)
      return false;
    const auto *arguments = body->getArray("arguments");
    const auto *ops = body->getArray("operations");
    if (!arguments || !ops)
      return r.bad("region must have argument/operation arrays");
    FrontendRegion parsed;
    for (const auto &arg : *arguments) {
      FrontendValue v;
      if (!readValue(r, &arg, v))
        return false;
      parsed.arguments.push_back(std::move(v));
    }
    for (const auto &op : *ops) {
      FrontendOperation v;
      if (!readOperation(r, &op, v))
        return false;
      parsed.operations.push_back(std::move(v));
    }
    out.regions.push_back(std::move(parsed));
  }
  r.span = previousSpan;
  return true;
}
bool readProgram(Reader &r, const JSON *value, FrontendProgram &out) {
  const auto *o = r.record(value, {"version", "entry", "tensors", "constexprs",
                                   "captured_constants", "launch", "operations",
                                   "numerical_contract", "source"});
  if (!o || !r.integer(o->get("version"), out.version) || out.version != 1)
    return r.bad("only schema version 1 is supported");
  if (!r.string(o->get("entry"), out.entry) || !identifier(out.entry) ||
      !r.string(o->get("source"), out.source, true) ||
      !r.constants(o->get("constexprs"), out.constexprs) ||
      !r.constants(o->get("captured_constants"), out.capturedConstants) ||
      !readNumerical(r, o->get("numerical_contract"), out.numericalContract))
    return false;
  const auto *launch = r.record(o->get("launch"), {"grid", "block"});
  if (!launch)
    return false;
  std::vector<int64_t> grid, block;
  if (!r.shape(launch->get("grid"), grid, 3, 3) ||
      !r.shape(launch->get("block"), block, 3, 3))
    return false;
  for (size_t i = 0; i != 3; ++i) {
    if (grid[i] > maxIndex / block[i])
      return r.bad("launch extent overflows");
    out.launch.grid[i] = grid[i];
    out.launch.block[i] = block[i];
  }
  const auto *tensors = o->getArray("tensors");
  const auto *operations = o->getArray("operations");
  if (!tensors || !operations)
    return r.bad("program ports/operations must be arrays");
  for (const auto &item : *tensors) {
    const auto *t = r.record(
        &item, {"name", "shape", "dtype", "strides", "direction", "effect"});
    TensorSpec tensor;
    if (!t || !r.string(t->get("name"), tensor.name) ||
        !identifier(tensor.name) ||
        !r.shape(t->get("shape"), tensor.shape, 2) ||
        !r.string(t->get("dtype"), tensor.dtype) ||
        !storageType(tensor.dtype) ||
        !r.shape(t->get("strides"), tensor.strides, 2) ||
        !r.string(t->get("direction"), tensor.direction) ||
        !r.string(t->get("effect"), tensor.effect))
      return r.bad("invalid tensor metadata");
    unsigned bytes = tensor.dtype == "i8"                                ? 1
                     : (tensor.dtype == "bf16" || tensor.dtype == "f16") ? 2
                                                                         : 4;
    if (tensor.strides != std::vector<int64_t>{tensor.shape[1], 1} ||
        tensor.shape[0] > maxIndex / tensor.shape[1] / bytes ||
        !((tensor.direction == "input" && tensor.effect == "read") ||
          (tensor.direction == "output" && tensor.effect == "write")))
      return r.bad("unsupported tensor layout/effect or byte overflow");
    out.tensors.push_back(std::move(tensor));
  }
  for (const auto &item : *operations) {
    FrontendOperation op;
    if (!readOperation(r, &item, op))
      return false;
    out.operations.push_back(std::move(op));
  }
  return true;
}
bool readOptions(Reader &r, const JSON *value, CompileOptions &out) {
  const auto *o = r.record(
      value, {"target", "backend", "search_mode", "max_candidates", "top_k",
              "numerical_contract", "config_root", "config_files"});
  if (!o || !r.string(o->get("target"), out.target) ||
      !r.string(o->get("backend"), out.backend) ||
      !r.string(o->get("search_mode"), out.searchMode) ||
      !r.integer(o->get("max_candidates"), out.maxCandidates, 1) ||
      !r.integer(o->get("top_k"), out.topK, 1) ||
      !readNumerical(r, o->get("numerical_contract"), out.numericalContract) ||
      !r.string(o->get("config_root"), out.configRoot))
    return false;
  if ((out.backend != "reference" && out.backend != "selected-target") ||
      out.searchMode != "deterministic" || out.topK > out.maxCandidates)
    return r.bad("invalid backend or search options");
  const auto *snapshot = o->getObject("config_files");
  if (!snapshot)
    return r.bad("config_files must be an explicit snapshot object");
  for (const auto &file : *snapshot) {
    auto spelling = file.first.str();
    if (spelling.find('\0') != std::string::npos)
      return r.bad("config path contains NUL");
    std::filesystem::path path(spelling);
    if (path.empty() || path.is_absolute())
      return r.bad("config files require relative paths");
    for (const auto &part : path)
      if (part == "..")
        return r.bad("config file escapes its root");
    std::string content;
    if (!r.string(&file.second, content, true))
      return false;
  }
  out.configFiles = *snapshot;
  return true;
}
using Scope = std::map<std::string, FrontendValue>;
std::optional<std::vector<int64_t>>
broadcast(const std::vector<std::vector<int64_t>> &shapes) {
  size_t rank = 0;
  for (const auto &shape : shapes)
    rank = std::max(rank, shape.size());
  std::vector<int64_t> result(rank, 1);
  for (const auto &shape : shapes)
    for (size_t i = 0; i < shape.size(); ++i) {
      size_t axis = rank - shape.size() + i;
      if (result[axis] != 1 && shape[i] != 1 && result[axis] != shape[i])
        return std::nullopt;
      result[axis] = std::max(result[axis], shape[i]);
    }
  return result;
}
class Validator {
public:
  explicit Validator(const FrontendProgram &program) : program(program) {}
  llvm::Error run() {
    Scope scope;
    for (size_t i = 0; i < program.tensors.size(); ++i) {
      const auto &t = program.tensors[i];
      if (!defined.insert(t.name).second)
        return frontendError("invalid_frontend_schema",
                             "duplicate tensor name");
      if (t.dtype != program.numericalContract.storageDtype &&
          !(t.direction == "output" &&
            t.dtype == program.numericalContract.accumulatorDtype))
        return frontendError("invalid_frontend_schema",
                             "tensor/numerical dtype mismatch");
      scope.emplace(t.name, FrontendValue{t.name, "ptr<" + t.dtype + ">", {}});
      origins[t.name] = i;
    }
    auto result = walk(program.operations, scope, false);
    if (!result)
      return result.takeError();
    if (casts != program.numericalContract.castSites)
      return frontendError("invalid_frontend_schema",
                           "ordered casts disagree with numerical contract");
    return llvm::Error::success();
  }

private:
  const FrontendProgram &program;
  std::set<std::string> defined;
  std::map<std::string, size_t> origins;
  std::vector<CastSite> casts;
  llvm::Error bad(const FrontendOperation &op, llvm::Twine message) {
    return frontendError("invalid_frontend_schema", message, op.span);
  }
  static bool same(const FrontendValue &a, const FrontendValue &b) {
    return a.dtype == b.dtype && a.shape == b.shape;
  }
  static bool type(const FrontendValue &a, llvm::StringRef dtype,
                   const std::vector<int64_t> &shape) {
    return a.dtype == dtype && a.shape == shape;
  }
  static int64_t attr(const FrontendOperation &op, llvm::StringRef name) {
    return op.attributes.getInteger(name).value_or(0);
  }
  llvm::Expected<std::vector<FrontendValue>>
  walk(const std::vector<FrontendOperation> &ops, Scope scope, bool region) {
    if (ops.empty() || ops.back().opcode != (region ? "yield" : "return"))
      return frontendError("invalid_frontend_schema",
                           "missing terminal yield/void return");
    for (size_t index = 0; index < ops.size(); ++index) {
      const auto &op = ops[index];
      const auto &code = op.opcode;
      std::vector<FrontendValue> in;
      for (const auto &id : op.operands) {
        auto found = scope.find(id);
        if (found == scope.end())
          return bad(op, "undefined or escaping value " + id);
        in.push_back(found->second);
      }
      const FrontendValue empty;
      const auto &result = op.results.empty() ? empty : op.results.front();
      if (code == "return" || code == "yield") {
        if (index != ops.size() - 1 || code != (region ? "yield" : "return"))
          return bad(op, "misplaced terminator");
        if (region)
          return in;
      } else if (code == "constant") {
        const auto *literal = op.attributes.get("value");
        if (!literal)
          return bad(op, "missing constant");
        if (result.dtype == "i1") {
          if (!literal->getAsBoolean())
            return bad(op, "i1 constant requires bool");
        } else if (result.dtype == "index" || result.dtype == "i32" ||
                   result.dtype == "i8") {
          auto n = literal->getAsInteger();
          if (!n ||
              (result.dtype == "i32" && (*n < INT32_MIN || *n > INT32_MAX)) ||
              (result.dtype == "i8" && (*n < -128 || *n > 127)))
            return bad(op, "integer constant exceeds dtype range");
        } else if (result.dtype != "f32" && result.dtype != "bf16" &&
                   result.dtype != "f16")
          return bad(op, "constant has unsupported dtype");
      } else if (code == "program_id") {
        if (attr(op, "axis") < 0 || attr(op, "axis") > 2 ||
            !type(result, "index", {}))
          return bad(op, "invalid program axis/type");
      } else if (code == "range") {
        auto start = attr(op, "start"), end = attr(op, "end");
        if (end <= start || (start < 0 && end > maxIndex + start) ||
            !type(result, "index", {end - start}))
          return bad(op, "invalid range extent/type");
      } else if (code == "neg") {
        if ((in[0].dtype != "f32" && in[0].dtype != "bf16" &&
             in[0].dtype != "f16") ||
            !same(result, in[0]))
          return bad(op, "neg requires matching floating type/shape");
      } else if (code == "add" || code == "sub" || code == "mul" ||
                 code == "div" || code == "rem" || code == "ceildiv" ||
                 code == "compare") {
        auto dims = broadcast({in[0].shape, in[1].shape});
        if (in[0].dtype != in[1].dtype || in[0].dtype == "i1" ||
            in[0].dtype.starts_with("ptr<") || !dims)
          return bad(op, "invalid numeric arithmetic operands");
        if ((code == "rem" || code == "ceildiv") && in[0].dtype != "index" &&
            in[0].dtype != "i32" && in[0].dtype != "i8")
          return bad(op, "integer operation requires integer values");
        if (!type(result, code == "compare" ? "i1" : in[0].dtype, *dims))
          return bad(op, "arithmetic result type/shape mismatch");
        if (code == "compare") {
          auto predicate = op.attributes.getString("predicate").value_or("");
          if (predicate != "eq" && predicate != "ne" && predicate != "lt" &&
              predicate != "le" && predicate != "gt" && predicate != "ge")
            return bad(op, "unknown comparison predicate");
        }
      } else if (code == "select") {
        auto dims = broadcast({in[0].shape, in[1].shape, in[2].shape});
        if (in[0].dtype != "i1" || in[1].dtype != in[2].dtype ||
            in[1].dtype.starts_with("ptr<") || !dims ||
            !type(result, in[1].dtype, *dims))
          return bad(op, "invalid select types/shapes");
      } else if (code == "expand_dims") {
        int64_t axis = attr(op, "axis");
        auto shape = in[0].shape;
        if (axis < 0 || uint64_t(axis) > shape.size() || shape.size() >= 2)
          return bad(op, "invalid expand_dims axis");
        shape.insert(shape.begin() + axis, 1);
        if (!type(result, in[0].dtype, shape))
          return bad(op, "expand_dims result mismatch");
      } else if (code == "broadcast") {
        const auto *dimensions = op.attributes.getArray("dimensions");
        if (!dimensions || dimensions->size() != in[0].shape.size() ||
            result.dtype != in[0].dtype)
          return bad(op, "invalid broadcast dimensions/type");
        int64_t previous = -1;
        for (size_t i = 0; i < dimensions->size(); ++i) {
          auto d = (*dimensions)[i].getAsInteger();
          if (!d || *d <= previous || uint64_t(*d) >= result.shape.size() ||
              (in[0].shape[i] != 1 && in[0].shape[i] != result.shape[*d]))
            return bad(op, "broadcast axis/extent mismatch");
          previous = *d;
        }
      } else if (code == "address") {
        auto origin = origins.find(in[0].id);
        if (origin == origins.end() ||
            in[0].id != program.tensors[origin->second].name ||
            !in[0].shape.empty() || in[1].dtype != "index" ||
            in[2].dtype != "index")
          return bad(op,
                     "address requires declared base and index coordinates");
        auto dims = broadcast({in[1].shape, in[2].shape});
        if (!dims || !type(result, in[0].dtype, *dims))
          return bad(op, "address result mismatch");
        origins[result.id] = origin->second;
      } else if (code == "load" || code == "store") {
        auto origin = origins.find(in[0].id);
        if (origin == origins.end() || !in[0].dtype.starts_with("ptr<"))
          return bad(op, "memory operation requires declared address");
        const auto &tensor = program.tensors[origin->second];
        const auto &mask = in[code == "load" ? 1 : 2];
        if (!type(mask, "i1", in[0].shape))
          return bad(op, "mask must have exact i1 tile shape");
        if (code == "load") {
          if (tensor.effect != "read" || in[2].dtype != tensor.dtype ||
              (!in[2].shape.empty() && in[2].shape != in[0].shape) ||
              !type(result, tensor.dtype, in[0].shape))
            return bad(op, "load effect/padding/result mismatch");
        } else if (tensor.effect != "write" ||
                   !type(in[1], tensor.dtype, in[0].shape))
          return bad(op, "store effect/value mismatch");
      } else if (code == "cast") {
        if (in[0].dtype.starts_with("ptr<") ||
            result.dtype.starts_with("ptr<") || in[0].shape != result.shape ||
            op.attributes.getString("rounding") != "rtne")
          return bad(op, "invalid explicit cast");
        casts.push_back({result.id, in[0].dtype, result.dtype, "rtne"});
      } else if (code == "dot") {
        if (in[0].shape.size() != 2 || in[1].shape.size() != 2 ||
            in[0].dtype != in[1].dtype ||
            (in[0].dtype != "f32" && in[0].dtype != "bf16") ||
            in[0].shape[1] != in[1].shape[0] ||
            !type(in[2], "f32", {in[0].shape[0], in[1].shape[1]}) ||
            !same(result, in[2]))
          return bad(op, "dot type/shape/accumulator mismatch");
      } else if (code == "silu") {
        if (in[0].dtype != "f32" || !same(result, in[0]) ||
            program.numericalContract.activationExpression != "x/(1+exp(-x))" ||
            op.attributes.getString("approximation") !=
                program.numericalContract.activationApproximation)
          return bad(op, "SiLU must preserve its numerical contract");
      } else if (code == "counted_loop") {
        if (in.size() < 2 || op.results.size() != in.size() - 2 ||
            !type(in[0], "index", {}) || !type(in[1], "index", {}) ||
            attr(op, "step") <= 0)
          return bad(op, "invalid counted-loop bounds/arity/step");
        const auto &body = op.regions[0];
        if (body.arguments.size() != op.results.size() + 1 ||
            !type(body.arguments[0], "index", {}))
          return bad(op, "loop requires typed induction/iter arguments");
        Scope local = scope;
        for (const auto &arg : body.arguments) {
          if (!defined.insert(arg.id).second)
            return bad(op, "duplicate region argument");
          local[arg.id] = arg;
        }
        for (size_t i = 0; i < op.results.size(); ++i)
          if (in[i + 2].dtype.starts_with("ptr<") ||
              !same(body.arguments[i + 1], in[i + 2]) ||
              !same(op.results[i], in[i + 2]))
            return bad(op, "loop init/iter/result mismatch");
        auto yielded = walk(body.operations, std::move(local), true);
        if (!yielded)
          return yielded.takeError();
        if (yielded->size() != op.results.size())
          return bad(op, "loop yield arity mismatch");
        for (size_t i = 0; i < yielded->size(); ++i)
          if (!same((*yielded)[i], op.results[i]))
            return bad(op, "loop yield type mismatch");
      }
      for (const auto &v : op.results) {
        if (!defined.insert(v.id).second)
          return bad(op, "duplicate value id " + v.id);
        scope[v.id] = v;
      }
    }
    return std::vector<FrontendValue>{};
  }
};
} // namespace
llvm::Expected<FrontendRequest> parseRequest(llvm::StringRef text) {
  Reader reader(text);
  if (!reader.bounded())
    return llvm::make_error<FrontendError>(*reader.diagnostic);
  auto json = llvm::json::parse(text);
  if (!json)
    return frontendError("invalid_frontend_schema",
                         llvm::toString(json.takeError()), {}, "parse");
  if (!reader.scan(*json)) {
    if (!reader.diagnostic)
      reader.bad("invalid JSON structure");
    return llvm::make_error<FrontendError>(*reader.diagnostic);
  }
  const auto *object = reader.record(&*json, {"program", "options"});
  auto data = std::make_shared<FrontendRequestData>();
  if (!object || !readProgram(reader, object->get("program"), data->program) ||
      !readOptions(reader, object->get("options"), data->options)) {
    if (!reader.diagnostic)
      reader.bad("invalid request record");
    return llvm::make_error<FrontendError>(*reader.diagnostic);
  }
  if (JSON(Object(data->program.numericalContract.metadata)) !=
      JSON(Object(data->options.numericalContract.metadata)))
    return frontendError("invalid_frontend_schema",
                         "program/options numerical contract mismatch");
  if (auto error = Validator(data->program).run())
    return std::move(error);
  llvm::raw_string_ostream os(data->canonicalJSON);
  reader.canonical(*json, os);
  return FrontendRequest(std::move(data));
}
} // namespace llk::frontend
