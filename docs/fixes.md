# hnswgo — Complete Fix Log

This document contains a comprehensive list of all fixes applied to the hnswgo codebase, organized by branch. Each entry describes the problem that existed before the fix, the fix itself, affected files, and the commit that introduced it.

---

## Branch 1: `fix-memory-leaks` (`026e84a`)

### 1.1 Idempotent `Free()` — prevent double-free crash

**Problem:** `Free()` called `C.freeHNSW(idx.index)` unconditionally. Calling `Free()` twice caused a double-free on the C++ side, leading to undefined behavior (crash or heap corruption).

**Fix:** Added a nil-check on `idx.index` before calling `freeHNSW`. Set `idx.index = nil` after freeing, making subsequent calls no-ops.

**Files:** `hnsw.go`

---

### 1.2 `runtime.SetFinalizer` as GC safety net

**Problem:** If a caller forgot to call `Free()`, the C++ `HnswIndex` and its underlying `HierarchicalNSW` and `SpaceInterface` objects would leak permanently. Go's garbage collector has no knowledge of C++ allocations.

**Fix:** Added `runtime.SetFinalizer(idx, (*HnswIndex).Free)` in both `New()` and `Load()`. The finalizer acts as a safety net — the GC will call `Free()` if the caller does not. This does not replace explicit `Free()` calls; finalizers are non-deterministic.

**Files:** `hnsw.go`

---

### 1.3 `searchKnn` exception cleanup — prevent memory leak on error

**Problem:** In `searchKnn`, the `SearchResult` struct and its `label`/`dist` arrays were allocated before the `ParallelFor` search loop. If an exception occurred during search (e.g., "Cannot return results in a contiguous 2D array"), the exception propagated up but the allocated `SearchResult` memory was never freed.

**Fix:** Wrapped the `ParallelFor` calls in a try/catch block. On exception, the catch block frees `searchResult->label`, `searchResult->dist`, and `searchResult` before returning `nullptr`.

**Files:** `hnsw_wrapper.cc`

---

### 1.4 Dead `index = nullptr` in `freeHNSW`

**Problem:** `freeHNSW` contained `index = nullptr;` which only set the local copy of the pointer, not the caller's pointer. This was dead code that gave a false impression of safety.

**Fix:** Removed the dead assignment.

**Files:** `hnsw_wrapper.cc`

---

### 1.5 Nil-check for `cResult` in `SearchKNN`

**Problem:** `SearchKNN` in Go called `C.searchKnn()` and immediately dereferenced the result to read labels and distances. If `searchKnn` returned `nullptr` (on C++ exception), this was a null pointer dereference — immediate crash.

**Fix:** Added a nil-check: if `cResult == nil`, return an error instead of dereferencing.

**Files:** `hnsw.go`

---

## Branch 2: `fix-exception-handling` (`c88fcf0`)

### 2.1 `last_error` field on `HnswIndex` struct

**Problem:** C++ exceptions thrown inside `extern "C"` functions crossed the language boundary into Go, causing undefined behavior (typically `SIGABRT`). There was no mechanism to communicate C++ error messages to Go.

**Fix:** Added a `char *last_error` field to the `HnswIndex` C struct. Added `setError()` and `copyErrorString()` helpers in C++. Every `extern "C"` function that could throw was wrapped in try/catch. On exception, the error message is stored in `last_error` and an error code is returned. Go reads the error via a `lastError()` helper.

**Note:** This `last_error` field was later removed in Branch 8 (`fix-remaining-issues`) in favor of the per-call `char **err` out-parameter pattern, which eliminates a concurrency race condition. See section 8.1.

**Files:** `hnsw_wrapper.h`, `hnsw_wrapper.cc`, `hnsw.go`

---

### 2.2 `char **err` out-parameter for `newIndex` and `loadIndex`

**Problem:** `newIndex` and `loadIndex` create the `HnswIndex` struct itself, so they cannot use the `last_error` field (the struct doesn't exist yet if creation fails). Errors during index creation were silently swallowed — the functions returned `nullptr` with no explanation.

**Fix:** Added a `char **err` out-parameter to both functions. On failure, the error message is written to `*err` (malloc'd string that Go must free). Go reads and frees it with `readCError()`.

**Files:** `hnsw_wrapper.h`, `hnsw_wrapper.cc`, `hnsw.go`

---

### 2.3 Try/catch on all `extern "C"` functions

**Problem:** Multiple `extern "C"` functions (`saveIndex`, `addPoints`, `markDeleted`, `unmarkDeleted`, `resizeIndex`, `getDataByLabel`) had no exception handling. Any C++ exception (from hnswlib internals or `std::bad_alloc`) would propagate through the C boundary into Go — undefined behavior.

**Fix:** Every `extern "C"` function that calls hnswlib methods is now wrapped in try/catch. Functions that previously returned `void` now return `int` (0 = success, 1 = error). The catch block stores the error message and returns the error code.

**Files:** `hnsw_wrapper.h`, `hnsw_wrapper.cc`

---

### 2.4 Go API returns errors consistently

**Problem:** Several Go methods had no error return: `New()` returned `*HnswIndex` (no error), `Save()` returned nothing, `MarkDeleted()` returned nothing, etc. Callers had no way to know if operations failed.

**Fix:** Changed signatures:
- `New()` → `(*HnswIndex, error)`
- `Load()` → `(*HnswIndex, error)`
- `Save()` → `error`
- `MarkDeleted()` → `error`
- `UnmarkDeleted()` → `error`
- `ResizeIndex()` → `error`
- `GetDataByLabel()` → `([]float32, error)`

All methods check the C return code and return a descriptive error on failure.

**Files:** `hnsw.go`, `hnsw_test.go`, `example/example.go`

---

### 2.5 Fixed `getDataByLabel` — local pointer reassignment bug

**Problem:** The C++ `getDataByLabel` function had:
```cpp
float* data = ...;  // caller's buffer
auto vec = getDataByLabel<float>(label);
data = vec.data();  // BUG: reassigns local pointer, caller's buffer unchanged
```
The function appeared to work (compiled, no crash) but the caller's buffer was never written to. The returned vector was always zeros.

**Fix:** Changed to `memcpy(data, vec.data(), sizeof(float) * index->dim)` which correctly copies the vector data into the caller's buffer.

**Files:** `hnsw_wrapper.cc`

---

### 2.6 Fixed Go `GetDataByLabel` — wrong pointer passed to C

**Problem:** Go's `GetDataByLabel` passed `&vec` (pointer to the slice header) instead of `&vec[0]` (pointer to the underlying array's first element) to the C function. This caused the C function to write into Go's slice header memory, corrupting it.

**Fix:** Changed `(*C.float)(unsafe.Pointer(&vec))` to `(*C.float)(unsafe.Pointer(&vec[0]))`.

**Files:** `hnsw.go`

---

### 2.7 Default case for `SpaceType` switch

**Problem:** `New()` and `Load()` accepted any `SpaceType` value. Invalid values fell through the switch without setting `sType`, using whatever `C.l2` was the zero-value. This silently created an L2 index regardless of the invalid input.

**Fix:** Added a `default` case that returns an error: `"unsupported space type"`.

**Files:** `hnsw.go`

---

## Branch 3: `fix-test-coverage` (`9740cf6`)

### 3.1 Implemented `TestGetVectorData` (was empty)

**Problem:** `TestGetVectorData` existed as an empty test function — it compiled but tested nothing. The `GetDataByLabel` function had no test coverage, hiding the bugs fixed in 2.5 and 2.6.

**Fix:** Implemented a full round-trip test: add a known vector with a specific label, retrieve it with `GetDataByLabel`, compare element-by-element.

**Files:** `hnsw_test.go`

---

### 3.2 Rewrote `TestVectorSearch` to be self-contained

**Problem:** `TestVectorSearch` loaded from `./example.data`, an external file that had to be generated separately. The test failed if the file didn't exist, making it non-portable and CI-unfriendly.

**Fix:** Rewrote the test to create an in-memory index, populate it with random vectors, and search — no external file dependency.

**Files:** `hnsw_test.go`

---

### 3.3 Added `TestErrorPaths`

**Problem:** No tests exercised error paths: missing labels, wrong dimensions, nonexistent files.

**Fix:** Added three subtests:
- `GetDataByLabel_missing_label` — retrieves a non-existent label, expects error
- `AddPoints_wrong_dimension` — adds vectors with wrong dimensionality, expects error
- `Load_nonexistent_file` — loads from a non-existent path, expects error

**Files:** `hnsw_test.go`

---

### 3.4 Added `TestConcurrentAccess`

**Problem:** No tests verified that the index was safe for concurrent goroutine use.

**Fix:** Test launches 10 concurrent search goroutines and 5 concurrent add goroutines against the same index. Verifies no panics or data races (via `-race` flag).

**Files:** `hnsw_test.go`

---

### 3.5 Added `TestMultipleSpaceTypes`

**Problem:** Tests only exercised the Cosine space type. L2 and IP were untested.

**Fix:** Table-driven test that creates an index, adds points, and searches for each of L2, IP, and Cosine.

**Files:** `hnsw_test.go`

---

### 3.6 Added `TestDoubleFree`

**Problem:** No test verified that calling `Free()` twice was safe.

**Fix:** Test creates an index, calls `Free()` twice, verifies no panic.

**Files:** `hnsw_test.go`

---

### 3.7 Added `TestUnmarkDeleted`

**Problem:** `UnmarkDeleted` had no test coverage.

**Fix:** Test marks a label as deleted, unmarks it, then verifies it's still retrievable via `GetDataByLabel`.

**Files:** `hnsw_test.go`

---

## Branch 4: `fix-hnswlib-pinning` (`3d24c6b`)

### 4.1 Removed `*.mod` from `.gitignore`

**Problem:** `.gitignore` contained `*.mod` (intended for Fortran module files). This blocked `go.mod` from being committed, meaning the project had no Go module definition. Users couldn't `go get` it, and builds required `GO111MODULE=off`.

**Fix:** Removed the `*.mod` line from `.gitignore`.

**Files:** `.gitignore`

---

### 4.2 Created `go.mod`

**Problem:** No `go.mod` file existed. The project couldn't be used as a Go module dependency.

**Fix:** Created `go.mod` with `module github.com/midhunkrishna/hnswgo` and `go 1.21`.

**Files:** `go.mod`

---

### 4.3 Created `hnswlib/VERSION`

**Problem:** The vendored hnswlib headers had no version tracking. There was no way to know which upstream version was included or when it was last updated.

**Fix:** Created `hnswlib/VERSION` documenting the source (nmslib/hnswlib), approximate version (v0.8.0), and that it includes PR #508 (custom error stream).

**Files:** `hnswlib/VERSION`

---

### 4.4 Updated example import path

**Problem:** The example used the old module path.

**Fix:** Updated import to `github.com/midhunkrishna/hnswgo`.

**Files:** `example/example.go`

---

## Branch 5: `fix-variable-typing` (`2ced1ef`)

### 5.1 Fixed `C.sizeof_ulong` → `C.sizeof_size_t` in `SearchKNN`

**Problem:** `SearchKNN` used `C.sizeof_ulong` as the stride when reading labels from the C `SearchResult` struct. On platforms where `sizeof(unsigned long) != sizeof(size_t)` (e.g., Windows LLP64), this would read labels at wrong offsets, returning garbage data.

**Fix:** Changed to `C.sizeof_size_t` which matches the actual type of `SearchResult.label` (`size_t *`).

**Files:** `hnsw.go`

---

### 5.2 Fixed label dereference to use `C.size_t`

**Problem:** Labels were dereferenced as `*(*uint64)` which assumes `size_t == uint64`. On 32-bit platforms, `size_t` is 4 bytes but `uint64` reads 8 bytes — reading 4 bytes of garbage.

**Fix:** Changed to `uint64(*(*C.size_t)(...))` — dereference as the correct C type, then widen to Go's `uint64`.

**Files:** `hnsw.go`

---

### 5.3 Added `[]uint64` to `[]C.size_t` conversion in `AddPoints`

**Problem:** `AddPoints` passed `[]uint64` labels directly to C as `(*C.size_t)`. If `sizeof(uint64) != sizeof(size_t)` (32-bit platforms), this reinterprets the memory layout incorrectly — labels would be garbled.

**Fix:** Added explicit conversion: create a `[]C.size_t` slice, copy each label, and pass that to C.

**Files:** `hnsw.go`

---

## Branch 6: `fix-predictable-api` (`6c2ad99`)

### 6.1 Added `TestGetDataByLabelRoundTrip`

**Problem:** The core `getDataByLabel` bugs were fixed in Branch 2, but there was no table-driven test verifying correctness across all space types. Cosine space normalizes vectors on insert, so the round-trip behavior differs from L2 and IP.

**Fix:** Table-driven test over L2, IP, and Cosine. For L2/IP, verifies exact round-trip equality. For Cosine, verifies the returned vector is non-zero (but not equal to input, since it's normalized).

**Files:** `hnsw_test.go`

---

### 6.2 Added doc comment explaining Cosine normalization

**Problem:** `GetDataByLabel` with Cosine space returns the *normalized* vector, not the original input. This was undocumented — callers would be surprised that the vector they put in is not the vector they get back.

**Fix:** Added doc comment: "For Cosine space, the returned vector is the normalized version that was stored, not the original input vector."

**Files:** `hnsw.go`

---

## Branch 7: `fix-pointer-safety` (`3ed0b4c`)

### 7.1 Added `sync.RWMutex` for goroutine safety

**Problem:** `HnswIndex` methods accessed the underlying C pointer (`idx.index`) without any synchronization. If one goroutine called `Free()` while another was in `SearchKNN`, the search would dereference a freed pointer — use-after-free, crash, or data corruption.

**Fix:** Added `sync.RWMutex` field `mu` to `HnswIndex`. Read-only methods (`SearchKNN`, `GetDataByLabel`, `Save`, `IndexFileSize`, `GetMaxElements`, `GetCurrentCount`, `GetAllowReplaceDeleted`) take `RLock`. Mutating methods (`AddPoints`, `MarkDeleted`, `UnmarkDeleted`, `ResizeIndex`, `Free`) take full `Lock`.

**Files:** `hnsw.go`

---

### 7.2 Added `ErrIndexClosed` sentinel error

**Problem:** After `Free()`, all methods would crash with a nil pointer dereference when trying to pass `idx.index` to C. There was no way for callers to distinguish "index is closed" from other errors.

**Fix:** Added `var ErrIndexClosed = errors.New("index is closed")`. All methods nil-check `idx.index` under the lock and return `ErrIndexClosed` if the index has been freed.

**Files:** `hnsw.go`

---

### 7.3 Changed `GetMaxElements`, `GetCurrentCount`, `IndexFileSize` to return `(uint64, error)`

**Problem:** These methods returned only a value. After `Free()`, they would crash instead of returning an error.

**Fix:** Changed signatures to `(uint64, error)` with nil-guard returning `ErrIndexClosed`.

**Files:** `hnsw.go`, `hnsw_test.go`

---

### 7.4 Changed `SetEf` to return `error`

**Problem:** `SetEf` returned nothing. After `Free()`, it would crash.

**Fix:** Changed to return `error` with nil-guard.

**Files:** `hnsw.go`

---

### 7.5 Added `TestUseAfterFree`

**Problem:** No test verified that all methods returned `ErrIndexClosed` after `Free()`.

**Fix:** Test calls `Free()`, then invokes every method and asserts `errors.Is(err, ErrIndexClosed)`.

**Files:** `hnsw_test.go`

---

### 7.6 Added `TestConcurrentFreeAndSearch`

**Problem:** No test verified that concurrent `Free()` and `SearchKNN` calls didn't race or crash.

**Fix:** Test launches `Free()` and `SearchKNN` in parallel goroutines, runs with `-race`. Either `SearchKNN` completes normally or gets `ErrIndexClosed` — neither should panic.

**Files:** `hnsw_test.go`

---

## Branch 8: `fix-remaining-issues` (current branch)

This branch addresses all issues identified in the post-fix re-analysis by the adversarial-analyst, code-reviewer, and performance-tuner.

### 8.1 Eliminated `last_error` race condition — switched to `char **err` out-params

**Problem:** The `last_error` field on `HnswIndex` was shared mutable state. Methods like `Save()` and `GetDataByLabel()` held `RLock` (allowing concurrent access) and called `lastError()` on failure, which reads, frees, and clears `last_error`. If two concurrent readers both failed, they would race on `lastError()` — one would read the pointer, the other would free it, then the first would double-free or read freed memory. This was a use-after-free / double-free race condition.

**Fix:** Removed the `last_error` field from the `HnswIndex` C struct entirely. Changed all fallible C functions (`saveIndex`, `addPoints`, `markDeleted`, `unmarkDeleted`, `resizeIndex`, `getDataByLabel`, `searchKnn`, `indexFileSize`) to accept a `char **err` out-parameter. On error, each function writes a malloc'd error string to `*err`. The Go side declares a local `var cErr *C.char`, passes `&cErr`, and reads the result with `readCError()`. Since each goroutine has its own stack-local `cErr`, there is no shared state and no race.

**Files:** `hnsw_wrapper.h`, `hnsw_wrapper.cc`, `hnsw.go`

---

### 8.2 Fixed memory leak in `newIndex`/`loadIndex` catch blocks

**Problem:** In `newIndex` and `loadIndex`, the `HnswIndex` struct and `SpaceInterface` were allocated before the `HierarchicalNSW` constructor. If the constructor threw an exception, the catch block returned `nullptr` but did not free the already-allocated `space` and `index` objects — they leaked.

**Before (leaks on throw):**
```cpp
try {
    HnswIndex *index = new HnswIndex;        // allocated
    hnswlib::SpaceInterface<float> *space;
    space = new hnswlib::L2Space(dim);        // allocated
    auto *appr_alg = new HierarchicalNSW(...); // THROWS
    ...
} catch (const std::exception& e) {
    // index and space are leaked — not accessible here
    if (err) *err = copyErrorString(e.what());
    return nullptr;
}
```

**Fix:** Moved `index` and `space` declarations outside the try block (initialized to `nullptr`), so the catch block can free them:
```cpp
HnswIndex *index = nullptr;
hnswlib::SpaceInterface<float> *space = nullptr;
try {
    ...
} catch (const std::exception& e) {
    delete space;   // safe even if nullptr
    delete index;   // safe even if nullptr
    ...
}
```

**Files:** `hnsw_wrapper.cc`

---

### 8.3 Fixed `norm_array` buffer overflow when `concurrency=0`

**Problem:** In both `addPoints` and `searchKnn`, when the caller passed `concurrency=0`, the `norm_array` (used for Cosine normalization) was allocated as `std::vector<float>(0 * dim)` — size zero. However, the `ParallelFor` function resolved `numThreads=0` to `hardware_concurrency()` internally, spawning real threads. Each thread wrote to `norm_array[threadId * dim]`, causing a heap buffer overflow.

**Before (overflow):**
```cpp
// num_threads = 0 (from caller)
std::vector<float> norm_array(num_threads * dim);  // size 0!
ParallelFor(0, rows, num_threads, [&](size_t row, size_t threadId) {
    // ParallelFor internally resolves 0 → hardware_concurrency()
    // threadId can be 0, 1, 2, ... up to hardware_concurrency()-1
    normalize_vector(dim, ..., norm_array.data() + threadId * dim);
    // OVERFLOW: writing past end of zero-sized array
});
```

**Fix:** Added `resolveThreadCount()` helper that clamps `num_threads` to at least 1 (falling back to `hardware_concurrency()`). Called before `norm_array` allocation, ensuring the array is sized to match the actual thread count.

**Files:** `hnsw_wrapper.cc`

---

### 8.4 `SetEf` changed from `RLock` to `Lock`

**Problem:** `SetEf` modified `ef_` (a field read by `searchKnn` during searches) but only held `RLock`. If `SetEf` ran concurrently with `SearchKNN` (both holding `RLock`), there was a data race on `ef_`: one goroutine writing while another reads.

**Fix:** Changed `SetEf` to use `idx.mu.Lock()` (exclusive lock), preventing concurrent reads during the mutation.

**Files:** `hnsw.go`

---

### 8.5 `searchKnn` error propagation to Go

**Problem:** When `searchKnn` caught a C++ exception, it wrote the error to `stderr` and returned `nullptr`. The Go side received `nil` with no error message, returning the generic "search failed: internal error". The actual error message (e.g., "Cannot return results in a contiguous 2D array") was lost to stderr, invisible to the Go caller.

**Before:**
```cpp
} catch (const std::exception& e) {
    std::cerr << "[hnsw] searchKnn exception: " << e.what() << std::endl;
    ...
    return nullptr;
}
```

**Fix:** Added `char **err` parameter to `searchKnn`. On exception, writes the error message to `*err` instead of stderr. Go reads it with `readCError()`.

**Files:** `hnsw_wrapper.h`, `hnsw_wrapper.cc`, `hnsw.go`

---

### 8.6 Removed `-march=native` from CGo CXXFLAGS

**Problem:** The CGo compiler flags included `-march=native`, which generates code optimized for the build machine's CPU. When cross-compiling or building in CI on a different architecture (e.g., building on x86 for ARM, or building in Docker on a different CPU), the resulting binary could use illegal instructions, causing `SIGILL` crashes.

**Fix:** Removed `-march=native` from the `#cgo CXXFLAGS` directive.

**Files:** `hnsw.go`

---

### 8.7 Added try/catch to `indexFileSize`

**Problem:** `indexFileSize` called `indexFileSize()` on the hnswlib object without exception handling. This function performs file I/O calculations that could throw. An uncaught exception would cross the `extern "C"` boundary — undefined behavior.

**Fix:** Changed signature to `int indexFileSize(HnswIndex *index, size_t *result, char **err)`. Wrapped in try/catch. Returns result via out-parameter, error via `char **err`.

**Files:** `hnsw_wrapper.h`, `hnsw_wrapper.cc`, `hnsw.go`

---

### 8.8 `GetAllowReplaceDeleted` returns `(bool, error)`

**Problem:** `GetAllowReplaceDeleted` returned only `bool`. After `Free()`, it silently returned `false` — the caller couldn't distinguish "replace not allowed" from "index is closed".

**Fix:** Changed return type to `(bool, error)`. Returns `(false, ErrIndexClosed)` when the index is freed.

**Files:** `hnsw.go`, `hnsw_test.go`

---

### 8.9 Removed dead `CustomFilterFunctor` class

**Problem:** `hnsw_wrapper.cc` contained an unused `CustomFilterFunctor` class (lines 99-113 in the old code). It was never instantiated — the commented-out filter function pointer in `searchKnn` referenced it but was disabled. Dead code adds maintenance burden and confusion.

**Fix:** Removed the class entirely.

**Files:** `hnsw_wrapper.cc`

---

### 8.10 Removed dead null-checks after C++ `new`

**Problem:** `searchKnn` had null-checks after `new SearchResult`, `new hnswlib::labeltype[...]`, and `new float[...]`:
```cpp
SearchResult *searchResult = new SearchResult;
if (!searchResult) { return nullptr; }  // dead code
```
In C++, `new` throws `std::bad_alloc` on failure — it never returns `nullptr`. These checks were dead code that gave a false impression of error handling.

**Fix:** Removed the dead null-checks. The `try/catch` already handles `std::bad_alloc`.

**Files:** `hnsw_wrapper.cc`

---

### 8.11 Eliminated `convertTo2DVector` — direct pointer arithmetic

**Problem:** Both `addPoints` and `searchKnn` called `convertTo2DVector()`, which deep-copied the entire flat vector array into a `std::vector<std::vector<float>>`. The Go side had already flattened the 2D slice into a contiguous array — the C++ side was undoing this work. For 10K vectors of dim=400, this allocated ~16MB of temporary heap memory and performed 4M unnecessary float copies.

**Fix:** Eliminated `convertTo2DVector` entirely. Both functions now index directly into the flat array with pointer arithmetic: `flat_vectors + row * dim`. This removes all C++ heap allocation from the hot path and halves peak memory during batch operations.

**Files:** `hnsw_wrapper.cc`

---

### 8.12 `const` correctness on `normalize_vector`

**Problem:** `normalize_vector` took `float *data` (non-const) for the input parameter, even though it never modifies the input. This was semantically incorrect and prevented the compiler from making certain optimizations.

**Fix:** Changed to `const float *data`.

**Files:** `hnsw_wrapper.cc`

---

### 8.13 Bulk-allocated `SearchResult` structs in Go

**Problem:** `SearchKNN` allocated each `SearchResult` individually inside the inner loop (`&r` escapes to heap). For a batch of 1000 queries with topK=100, this created 100,000 small heap objects, putting pressure on the garbage collector.

**Fix:** Pre-allocate a single flat `[]SearchResult` slice of size `rows * topK`. Each result is filled in-place, and pointers into this slice are used for the 2D output. This reduces allocations from O(rows*topK) to O(rows + 1).

**Files:** `hnsw.go`

---

### 8.14 Upgraded compiler optimization from `-O2` to `-O3`

**Problem:** The CGo CXXFLAGS used `-O2`. hnswlib's distance computation inner loops (L2, inner product) benefit significantly from `-O3`, which enables more aggressive auto-vectorization and inlining.

**Fix:** Changed `-O2` to `-O3` in the `#cgo CXXFLAGS` directive.

**Files:** `hnsw.go`

---

### 8.15 Fixed `topK` guard — check `getCurrentCount` not `getMaxElements`

**Problem:** `SearchKNN` checked `if topK > maxElements` to prevent oversized queries. But `maxElements` is the *capacity*, not the *actual element count*. An index with capacity 1000 but only 3 elements would pass this check with topK=10, then fail deep in C++ with a cryptic error: "Cannot return the results in a contiguous 2D array."

**Fix:** Changed to check against `getCurrentCount` (the actual number of elements in the index), with a clearer error message: "topK is larger than the number of elements in the index".

**Files:** `hnsw.go`

---

### 8.16 Clear finalizer in `Free()`

**Problem:** When `Free()` was called explicitly, the `runtime.SetFinalizer` registered during `New()`/`Load()` remained active. The GC would eventually call `Free()` again (safe due to idempotency), but the finalizer kept the `HnswIndex` object alive for an extra GC cycle, delaying memory reclamation.

**Fix:** Added `runtime.SetFinalizer(idx, nil)` inside `Free()` after freeing the C resources. This detaches the finalizer, allowing the GC to collect the Go object immediately.

**Files:** `hnsw.go`

---

### 8.17 Moved `searchKnn` allocations inside try/catch

**Problem:** In `searchKnn`, the `resolveThreadCount()` and vector processing occurred outside the `try` block. If any allocation threw `std::bad_alloc` (e.g., under memory pressure with large batch queries), the exception would propagate through the `extern "C"` boundary — undefined behavior, typically a crash.

**Fix:** Moved all code that can throw inside the `try` block. Only the `searchResult` pointer declaration (initialized to `nullptr`) remains outside, enabling proper cleanup in the catch.

**Files:** `hnsw_wrapper.cc`

---

### 8.18 Added null-guard to `freeResult`

**Problem:** `freeResult` did not check for a null `SearchResult` pointer. If called with `nullptr` (defensive programming), it would crash.

**Fix:** Added `if (!result) return;` at the top.

**Files:** `hnsw_wrapper.cc`

---

### 8.19 New tests for remaining issues

| Test | Covers |
|------|--------|
| `TestSearchKnnConcurrencyZero` | Fix 8.3: Cosine search with concurrency=0 doesn't overflow |
| `TestAddPointsConcurrencyZero` | Fix 8.3: Cosine add with concurrency=0 doesn't overflow |
| `TestSearchKnnTopKGuard` | Fix 8.15: topK > currentCount caught at Go level |
| `TestSearchKnnErrorPropagation` | Fix 8.5: C++ errors propagated to Go, not lost to stderr |
| `TestGetAllowReplaceDeletedClosed` | Fix 8.8: returns ErrIndexClosed on freed index |
| `TestIndexFileSizeTryCatch` | Fix 8.7: IndexFileSize works and returns ErrIndexClosed when freed |

Also updated:
- All `GetAllowReplaceDeleted()` calls updated for `(bool, error)` return
- `TestUseAfterFree` extended to cover `GetAllowReplaceDeleted`

**Files:** `hnsw_test.go`

---

## Summary

| Branch | Commit | Category | Fixes |
|--------|--------|----------|-------|
| `fix-memory-leaks` | `026e84a` | Memory safety | 5 |
| `fix-exception-handling` | `c88fcf0` | Error handling | 7 |
| `fix-test-coverage` | `9740cf6` | Test quality | 7 |
| `fix-hnswlib-pinning` | `3d24c6b` | Build/tooling | 4 |
| `fix-variable-typing` | `2ced1ef` | Type safety | 3 |
| `fix-predictable-api` | `6c2ad99` | API correctness | 2 |
| `fix-pointer-safety` | `3ed0b4c` | Concurrency safety | 6 |
| `fix-remaining-issues` | (current) | All categories | 19 |
| **Total** | | | **53** |

### Test count: 21 tests (14 original + 7 new in this branch)

All tests pass with `-race` flag enabled.
