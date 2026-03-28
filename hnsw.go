package hnswgo

// #cgo CXXFLAGS: -fPIC -pthread -Wall -std=c++11 -O3 -I.
// #cgo LDFLAGS: -pthread
// #cgo CFLAGS: -I./
// #include <stdlib.h>
// #include "hnsw_wrapper.h"
import "C"
import (
	"errors"
	"runtime"
	"sync"
	"unsafe"
)

// ErrIndexClosed is returned when an operation is attempted on a freed index.
var ErrIndexClosed = errors.New("index is closed")

type SpaceType int

const (
	L2 SpaceType = iota
	IP
	Cosine
)

// HnswIndex wraps the C index type and provides a set of useful index manipulation methods.
// All methods are safe for concurrent use from multiple goroutines.
type HnswIndex struct {
	mu    sync.RWMutex
	index *C.HnswIndex
}

// SearchResult is the result returned by search method. Field Distance may be of
// euclidean distance or inner product distance, or cosine distance, depending on the chosen space type.
type SearchResult struct {
	Label    uint64
	Distance float32
}

// readCError converts and frees a C error string. Returns "unknown error" if ptr is nil.
func readCError(cErr *C.char) string {
	if cErr == nil {
		return "unknown error"
	}
	msg := C.GoString(cErr)
	C.free(unsafe.Pointer(cErr))
	return msg
}

// New creates a new HnswIndex with the specified dimension and other parameters. For details please see hnswlib documents.
// When allowReplaceDeleted is set, deleted elements can be replaced with new added ones.
func New(dim, M, efConstruction, randSeed int, maxElements uint64, spaceType SpaceType, allowReplaceDeleted bool) (*HnswIndex, error) {
	var allowReplace int = 0
	if allowReplaceDeleted {
		allowReplace = 1
	}

	var sType C.spaceType = C.l2
	switch spaceType {
	case L2:
		sType = C.l2
	case IP:
		sType = C.ip
	case Cosine:
		sType = C.cosine
	default:
		return nil, errors.New("unsupported space type")
	}

	var cErr *C.char
	cindex := C.newIndex(sType, C.int(dim), C.size_t(maxElements), C.int(M), C.int(efConstruction), C.int(randSeed), C.int(allowReplace), &cErr)
	if cindex == nil {
		return nil, errors.New(readCError(cErr))
	}

	idx := &HnswIndex{
		index: cindex,
	}
	runtime.SetFinalizer(idx, (*HnswIndex).Free)
	return idx, nil
}

// Load loads data from an existing HNSW index file.
func Load(location string, spaceType SpaceType, dim int, maxElements uint64, allowReplaceDeleted bool) (*HnswIndex, error) {
	var allowReplace int = 0
	if allowReplaceDeleted {
		allowReplace = 1
	}

	var sType C.spaceType = C.l2
	switch spaceType {
	case L2:
		sType = C.l2
	case IP:
		sType = C.ip
	case Cosine:
		sType = C.cosine
	default:
		return nil, errors.New("unsupported space type")
	}

	cloc := C.CString(location)
	defer C.free(unsafe.Pointer(cloc))

	var cErr *C.char
	cindex := C.loadIndex(cloc, sType, C.int(dim), C.size_t(maxElements), C.int(allowReplace), &cErr)
	if cindex == nil {
		return nil, errors.New(readCError(cErr))
	}

	idx := &HnswIndex{
		index: cindex,
	}
	runtime.SetFinalizer(idx, (*HnswIndex).Free)
	return idx, nil
}

// SetEf sets the query time accuracy/speed trade-off, defined by the ef parameter (see doc ALGO_PARAMS.md of hnswlib).
// Note that the parameter is currently not saved along with the index, so you need to set it manually after loading.
func (idx *HnswIndex) SetEf(ef int) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if idx.index == nil {
		return ErrIndexClosed
	}
	C.setEf(idx.index, C.size_t(ef))
	return nil
}

// IndexFileSize returns the index file size in bytes.
func (idx *HnswIndex) IndexFileSize() (uint64, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if idx.index == nil {
		return 0, ErrIndexClosed
	}

	var result C.size_t
	var cErr *C.char
	rc := C.indexFileSize(idx.index, &result, &cErr)
	if int(rc) != 0 {
		return 0, errors.New("indexFileSize failed: " + readCError(cErr))
	}
	return uint64(result), nil
}

// Save writes index data to disk.
func (idx *HnswIndex) Save(location string) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if idx.index == nil {
		return ErrIndexClosed
	}

	cloc := C.CString(location)
	defer C.free(unsafe.Pointer(cloc))

	var cErr *C.char
	rc := C.saveIndex(idx.index, cloc, &cErr)
	if int(rc) != 0 {
		return errors.New("save failed: " + readCError(cErr))
	}
	return nil
}

// AddPoints adds points. Updates the point if it is already in the index.
// If replacement of deleted elements is enabled: replaces previously deleted point if any, updating it with new point.
func (idx *HnswIndex) AddPoints(vectors [][]float32, labels []uint64, concurrency int, replaceDeleted bool) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if idx.index == nil {
		return ErrIndexClosed
	}

	var replace int = 0
	if replaceDeleted {
		replace = 1
	}

	if len(vectors) <= 0 || len(labels) <= 0 {
		return errors.New("invalid vector data")
	}

	if len(labels) != len(vectors) {
		return errors.New("unmatched vectors size and labels size")
	}

	if len(vectors[0]) != int(idx.index.dim) {
		return errors.New("unmatched dimensions of vector and index")
	}

	rows := len(vectors)
	flatVectors := flatten2DArray(vectors)

	// Convert uint64 labels to C.size_t for portable type safety.
	cLabels := make([]C.size_t, len(labels))
	for i, l := range labels {
		cLabels[i] = C.size_t(l)
	}

	// A Go []float32 is layout-compatible with a C float[] so we can pass
	// the Go slice directly to the C function as a pointer to its first element.
	var cErr *C.char
	errCode := C.addPoints(idx.index,
		(*C.float)(unsafe.Pointer(&flatVectors[0])),
		C.int(rows),
		(*C.size_t)(unsafe.Pointer(&cLabels[0])),
		C.int(concurrency),
		C.int(replace),
		&cErr)

	if int(errCode) != 0 {
		return errors.New("add point failed: " + readCError(cErr))
	}

	return nil
}

// flatten the vectors to prevent the "cgo argument has Go pointer to unpinned Go pointer" issue.
func flatten2DArray(vectors [][]float32) []float32 {
	rows := len(vectors)
	dim := len(vectors[0])
	flatVectors := make([]float32, 0, rows*dim)

	for _, vector := range vectors {
		flatVectors = append(flatVectors, vector...)
	}

	return flatVectors
}

// SearchKNN does a batch query against the index using the provided vectors. concurrency sets the threads to use for searching.
// For each of the queried vectors, topK SearchResults will be returned if no error occurred.
func (idx *HnswIndex) SearchKNN(vectors [][]float32, topK int, concurrency int) ([][]*SearchResult, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if idx.index == nil {
		return nil, ErrIndexClosed
	}

	if len(vectors) <= 0 {
		return nil, errors.New("invalid vector data")
	}

	if len(vectors[0]) != int(idx.index.dim) {
		return nil, errors.New("unmatched dimensions of vector and index")
	}

	if uint64(topK) > uint64(C.getCurrentCount(idx.index)) {
		return nil, errors.New("topK is larger than the number of elements in the index")
	}

	rows := len(vectors)
	flatVectors := flatten2DArray(vectors)

	var cErr *C.char
	cResult := C.searchKnn(idx.index,
		(*C.float)(unsafe.Pointer(&flatVectors[0])),
		C.int(rows),
		C.int(topK),
		C.int(concurrency),
		&cErr,
	)

	if cResult == nil {
		return nil, errors.New("search failed: " + readCError(cErr))
	}
	defer C.freeResult(cResult)

	allResults := make([]SearchResult, rows*topK)
	results := make([][]*SearchResult, rows)
	for rowID := range results {
		rowTopk := make([]*SearchResult, topK)
		for j := 0; j < topK; j++ {
			flat := rowID*topK + j
			allResults[flat].Label = uint64(*(*C.size_t)(unsafe.Add(unsafe.Pointer(cResult.label), flat*C.sizeof_size_t)))
			allResults[flat].Distance = *(*float32)(unsafe.Add(unsafe.Pointer(cResult.dist), flat*C.sizeof_float))
			rowTopk[j] = &allResults[flat]
		}
		results[rowID] = rowTopk
	}

	return results, nil

}

// GetDataByLabel retrieves the stored vector for the given label.
// For Cosine space, the returned vector is the normalized version that was stored,
// not the original input vector.
func (idx *HnswIndex) GetDataByLabel(label uint64) ([]float32, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if idx.index == nil {
		return nil, ErrIndexClosed
	}

	vec := make([]float32, idx.index.dim)

	var cErr *C.char
	rc := C.getDataByLabel(idx.index, C.size_t(label), (*C.float)(unsafe.Pointer(&vec[0])), &cErr)
	if int(rc) != 0 {
		return nil, errors.New("getDataByLabel failed: " + readCError(cErr))
	}
	return vec, nil
}

// GetAllowReplaceDeleted returns the setting of allowReplaceDeleted.
func (idx *HnswIndex) GetAllowReplaceDeleted() (bool, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if idx.index == nil {
		return false, ErrIndexClosed
	}
	return C.getAllowReplaceDeleted(idx.index) > 0, nil
}

// MarkDeleted marks the element as deleted, so it will be omitted from search results.
func (idx *HnswIndex) MarkDeleted(label uint64) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if idx.index == nil {
		return ErrIndexClosed
	}

	var cErr *C.char
	rc := C.markDeleted(idx.index, C.size_t(label), &cErr)
	if int(rc) != 0 {
		return errors.New("markDeleted failed: " + readCError(cErr))
	}
	return nil
}

// UnmarkDeleted unmarks the element as deleted, so it will not be omitted from search results.
func (idx *HnswIndex) UnmarkDeleted(label uint64) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if idx.index == nil {
		return ErrIndexClosed
	}

	var cErr *C.char
	rc := C.unmarkDeleted(idx.index, C.size_t(label), &cErr)
	if int(rc) != 0 {
		return errors.New("unmarkDeleted failed: " + readCError(cErr))
	}
	return nil
}

// ResizeIndex changes the maximum capacity of the index.
func (idx *HnswIndex) ResizeIndex(newSize uint64) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if idx.index == nil {
		return ErrIndexClosed
	}

	var cErr *C.char
	rc := C.resizeIndex(idx.index, C.size_t(newSize), &cErr)
	if int(rc) != 0 {
		return errors.New("resizeIndex failed: " + readCError(cErr))
	}
	return nil
}

// GetMaxElements returns the current capacity of the index.
func (idx *HnswIndex) GetMaxElements() (uint64, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if idx.index == nil {
		return 0, ErrIndexClosed
	}
	return uint64(C.getMaxElements(idx.index)), nil
}

// GetCurrentCount returns the current number of elements stored in the index.
func (idx *HnswIndex) GetCurrentCount() (uint64, error) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	if idx.index == nil {
		return 0, ErrIndexClosed
	}
	return uint64(C.getCurrentCount(idx.index)), nil
}

// Free releases resources bound to the index. Should be called when index is destroyed on close.
// Safe to call multiple times.
func (idx *HnswIndex) Free() {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if idx.index != nil {
		C.freeHNSW(idx.index)
		idx.index = nil
		runtime.SetFinalizer(idx, nil)
	}
}
