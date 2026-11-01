"""Bounded specialization cache keyed by full native request identity.

The cache stores one owning reference per compiled executable and hands out
strong leases. A SHA-256 digest indexes buckets, but membership is always
decided by comparing the complete identity bytes: a digest collision must never
return another request's artifact. Identical concurrent misses share one
compilation; a failed compilation is removed so a later call can retry.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass

from llk._errors import LLKError
from llk.frontend import CompileRequest
from llk.frontend._serialize import serialize_request

from . import _ffi

CACHE_SCHEMA = b"llk-frontend-cache-v1"
DEFAULT_MAX_ENTRIES = 128


@dataclass(frozen=True, slots=True)
class CacheInfo:
    """Process-local counters for one ``CompilationCache``."""

    hits: int
    misses: int
    compilations: int
    evictions: int


class _Entry:
    __slots__ = ("digest", "identity", "state", "kernel", "condition")

    def __init__(self, digest, identity, lock):
        self.digest = digest
        self.identity = identity
        self.state = "pending"
        self.kernel = None
        self.condition = threading.Condition(lock)


def _digest(identity):
    """SHA-256 bucket index; callers still compare the full identity."""
    return hashlib.sha256(identity).hexdigest()


def request_identity(request, environment):
    """Return the full identity bytes for one request and native environment.

    Serialization carries the concrete program (source, captured constants,
    tensors, constexprs, launch, numerical policy) and the search/target
    options, including the target configuration snapshot by content. Runtime
    addresses never serialize.
    """
    if not isinstance(request, CompileRequest):
        raise LLKError(
            "cache requests must be CompileRequest records",
            code="invalid_frontend_schema",
            stage="compile",
        )
    if not isinstance(environment, (bytes, bytearray)):
        raise LLKError(
            "native environment identity must be bytes",
            code="invalid_frontend_schema",
            stage="compile",
        )
    return (
        CACHE_SCHEMA
        + b"\x00"
        + serialize_request(request)
        + b"\x00"
        + bytes(environment)
    )


class CompilationCache:
    """Bounded LRU of compiled executables with deduplicated misses.

    ``environment`` supplies the native environment identity consulted before
    every lookup; it defaults to the loaded ``LLKFrontendCAPI`` identity.
    """

    def __init__(self, max_entries=DEFAULT_MAX_ENTRIES, *, environment=None):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self._max_entries = max_entries
        self._environment = environment or _ffi.native_environment
        self._lock = threading.Lock()
        self._buckets = {}
        self._lru = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._compilations = 0
        self._evictions = 0

    def get_or_compile(self, request, compile_fn):
        """Return a lease for ``request``, calling ``compile_fn(request)`` once.

        The returned object only has to expose ``lease()`` and ``close()``;
        ``compile_request`` satisfies that contract.
        """
        if not callable(compile_fn):
            raise LLKError(
                "compile_fn must be callable",
                code="invalid_frontend_schema",
                stage="compile",
            )
        identity = request_identity(request, self._environment())
        digest = _digest(identity)
        counted_miss = False
        while True:
            with self._lock:
                entry = self._find(digest, identity)
                if entry is not None and entry.state == "ready":
                    if not counted_miss:
                        self._hits += 1
                    self._lru.move_to_end(entry)
                    return entry.kernel.lease()
                if not counted_miss:
                    self._misses += 1
                    counted_miss = True
                if entry is not None and entry.state == "pending":
                    entry.condition.wait_for(lambda: entry.state != "pending")
                    continue
                if entry is not None:
                    self._discard(entry)
                entry = _Entry(digest, identity, self._lock)
                self._buckets.setdefault(digest, []).append(entry)
            try:
                kernel = compile_fn(request)
            except BaseException:
                with self._lock:
                    entry.state = "failed"
                    self._discard(entry)
                    entry.condition.notify_all()
                raise
            failure = None
            with self._lock:
                entry.kernel = kernel
                entry.state = "ready"
                self._compilations += 1
                self._lru[entry] = None
                evicted = self._evict()
                try:
                    lease = kernel.lease()
                except BaseException as error:
                    # A compiler that returns an unusable artifact must not
                    # leave a "ready" success entry behind.
                    failure = error
                    entry.state = "failed"
                    self._discard(entry)
                entry.condition.notify_all()
            # Drop evicted owning references outside the cache lock. Leases
            # already in flight keep their own references.
            for stale in evicted:
                stale.close()
            if failure is not None:
                kernel.close()
                raise failure
            return lease

    @property
    def max_entries(self):
        """Configured bound on retained ready entries."""
        return self._max_entries

    def info(self):
        """Return the current hit/miss/compilation/eviction counters."""
        with self._lock:
            return CacheInfo(
                self._hits, self._misses, self._compilations, self._evictions
            )

    def _find(self, digest, identity):
        for entry in self._buckets.get(digest, ()):
            if entry.identity == identity:
                return entry
        return None

    def _discard(self, entry):
        bucket = self._buckets.get(entry.digest)
        if bucket is not None:
            try:
                bucket.remove(entry)
            except ValueError:  # pragma: no cover - defensive
                pass
            if not bucket:
                del self._buckets[entry.digest]
        self._lru.pop(entry, None)

    def _evict(self):
        evicted = []
        while len(self._lru) > self._max_entries:
            entry, _ = self._lru.popitem(last=False)
            self._discard(entry)
            self._evictions += 1
            if entry.kernel is not None:
                evicted.append(entry.kernel)
                entry.kernel = None
                entry.state = "evicted"
        return evicted
