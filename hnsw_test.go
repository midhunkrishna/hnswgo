package hnswgo

import (
	"errors"
	"math"
	"math/rand"
	"os"
	"strings"
	"sync"
	"testing"
)

const testVectorDB = "./test.db"
const (
	dim            = 400
	M              = 20
	efConstruction = 10

	batchSize = 100
)

func newTestIndex(batch int, allowRepaceDeleted bool) (*HnswIndex, error) {
	maxElements := batch * batchSize

	index, err := New(dim, M, efConstruction, 55, uint64(maxElements), Cosine, allowRepaceDeleted)
	if err != nil {
		return nil, err
	}

	for i := 0; i < batch; i++ {
		points, labels := randomPoints(dim, i*batchSize, batchSize)
		if err := index.AddPoints(points, labels, 1, false); err != nil {
			index.Free()
			return nil, err
		}
	}

	return index, nil
}

func mustGetMaxElements(t *testing.T, idx *HnswIndex) uint64 {
	t.Helper()
	v, err := idx.GetMaxElements()
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func mustGetCurrentCount(t *testing.T, idx *HnswIndex) uint64 {
	t.Helper()
	v, err := idx.GetCurrentCount()
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func mustGetAllowReplaceDeleted(t *testing.T, idx *HnswIndex) bool {
	t.Helper()
	v, err := idx.GetAllowReplaceDeleted()
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func TestNewIndex(t *testing.T) {
	var maxElements uint64 = batchSize * 1

	idx, err := newTestIndex(1, true)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Free()

	if mustGetMaxElements(t, idx) != maxElements {
		t.Fail()
	}

	if mustGetAllowReplaceDeleted(t, idx) != true {
		t.Fail()
	}

	if mustGetCurrentCount(t, idx) != maxElements {
		t.Fail()
	}

}

func TestLoadAndSaveIndex(t *testing.T) {
	var maxElements uint64 = batchSize * 1

	// setup
	idx, err := newTestIndex(1, true)
	if err != nil {
		t.Fatal(err)
	}
	if err := idx.Save(testVectorDB); err != nil {
		t.Fatal(err)
	}
	idx.Free()

	index, err := Load(testVectorDB, Cosine, dim, uint64(maxElements), true)
	if err != nil {
		t.Fatal(err)
	}
	if err := index.SetEf(efConstruction); err != nil {
		t.Fatal(err)
	}
	defer index.Free()

	if err := index.Save(testVectorDB); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		deleteDB()
	})
}

func TestResizeIndex(t *testing.T) {
	var maxElements uint64 = batchSize * 1

	idx, err := newTestIndex(1, false)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Free()

	if mustGetMaxElements(t, idx) != maxElements {
		t.Fail()
	}

	if mustGetCurrentCount(t, idx) != maxElements {
		t.Fail()
	}

	if mustGetAllowReplaceDeleted(t, idx) != false {
		t.Fail()
	}

	points, labels := randomPoints(dim, 1*batchSize, batchSize)
	err = idx.AddPoints(points, labels, 1, false)
	if err == nil {
		t.Log(err)
		t.FailNow()
	}

	if err := idx.ResizeIndex(maxElements * 2); err != nil {
		t.Fatal(err)
	}
	if mustGetMaxElements(t, idx) != maxElements*2 {
		t.Fail()
	}

	if mustGetCurrentCount(t, idx) != maxElements {
		t.Fail()
	}

	err = idx.AddPoints(points, labels, 1, false)
	if err != nil {
		t.Log(err)
		t.Fail()
	}
}

func TestReplacePoint(t *testing.T) {
	allowRepaceDeleted := true
	maxElements := 100
	index, err := New(dim, M, efConstruction, 505, uint64(maxElements), Cosine, allowRepaceDeleted)
	if err != nil {
		t.Fatal(err)
	}
	defer index.Free()

	if !mustGetAllowReplaceDeleted(t, index) {
		t.Fail()
	}

	points, labels := randomPoints(dim, 0, maxElements)
	index.AddPoints(points, labels, 1, false)

	if err := index.MarkDeleted(labels[len(labels)-1]); err != nil {
		t.Fatal(err)
	}

	err = index.AddPoints([][]float32{randomPoint(dim)}, []uint64{math.MaxUint64 - 1}, 1, false)
	if err == nil {
		t.Fail()
	}

	err = index.AddPoints([][]float32{randomPoint(dim)}, []uint64{math.MaxUint64 - 1}, 1, true)
	if err != nil {
		t.Fail()
	}

}

func TestVectorSearch(t *testing.T) {
	searchDim := 32
	batchSz := 100
	numBatches := 10
	maxElems := uint64(batchSz * numBatches)

	index, err := New(searchDim, 16, 200, 42, maxElems, Cosine, false)
	if err != nil {
		t.Fatal(err)
	}
	defer index.Free()
	index.SetEf(100)

	for i := 0; i < numBatches; i++ {
		points, labels := randomPoints(searchDim, i*batchSz, batchSz)
		if err := index.AddPoints(points, labels, 1, false); err != nil {
			t.Fatal(err)
		}
	}

	query := genQuery(searchDim, 10)
	topK := 5

	result, err := index.SearchKNN(query, topK, 1)
	if err != nil {
		t.Fatal(err)
	}

	if len(result) != len(query) {
		t.Fatalf("expected %d result rows, got %d", len(query), len(result))
	}

	for i, rv := range result {
		if len(rv) != topK {
			t.Fatalf("row %d: expected %d results, got %d", i, topK, len(rv))
		}
	}
}

func TestGetVectorData(t *testing.T) {
	testDim := 8
	maxElements := uint64(10)
	index, err := New(testDim, 16, 200, 42, maxElements, L2, false)
	if err != nil {
		t.Fatal(err)
	}
	defer index.Free()

	vec := make([]float32, testDim)
	for i := range vec {
		vec[i] = float32(i) * 1.1
	}
	label := uint64(7)

	if err := index.AddPoints([][]float32{vec}, []uint64{label}, 1, false); err != nil {
		t.Fatal(err)
	}

	got, err := index.GetDataByLabel(label)
	if err != nil {
		t.Fatal(err)
	}

	for i := range vec {
		if got[i] != vec[i] {
			t.Errorf("dim %d: got %f, want %f", i, got[i], vec[i])
		}
	}
}

func TestGetDataByLabelRoundTrip(t *testing.T) {
	for _, st := range []struct {
		name      string
		spaceType SpaceType
	}{
		{"L2", L2},
		{"IP", IP},
		{"Cosine", Cosine},
	} {
		t.Run(st.name, func(t *testing.T) {
			testDim := 16
			index, err := New(testDim, 16, 200, 42, 100, st.spaceType, false)
			if err != nil {
				t.Fatal(err)
			}
			defer index.Free()

			vec := make([]float32, testDim)
			for i := range vec {
				vec[i] = float32(i+1) * 0.1
			}
			label := uint64(42)
			if err := index.AddPoints([][]float32{vec}, []uint64{label}, 1, false); err != nil {
				t.Fatal(err)
			}

			got, err := index.GetDataByLabel(label)
			if err != nil {
				t.Fatal(err)
			}

			if len(got) != testDim {
				t.Fatalf("expected len %d, got %d", testDim, len(got))
			}

			if st.spaceType == Cosine {
				// Cosine normalizes on insert, so we verify non-zero values
				// but cannot compare to the original vector directly.
				allZero := true
				for _, v := range got {
					if v != 0 {
						allZero = false
						break
					}
				}
				if allZero {
					t.Error("expected non-zero normalized vector for Cosine space")
				}
			} else {
				for i := range vec {
					if got[i] != vec[i] {
						t.Errorf("dim %d: got %f, want %f", i, got[i], vec[i])
					}
				}
			}
		})
	}
}

func TestErrorPaths(t *testing.T) {
	t.Run("GetDataByLabel_missing_label", func(t *testing.T) {
		index, err := New(8, 16, 200, 42, 10, L2, false)
		if err != nil {
			t.Fatal(err)
		}
		defer index.Free()

		// Add one point so the index is not empty
		if err := index.AddPoints([][]float32{make([]float32, 8)}, []uint64{0}, 1, false); err != nil {
			t.Fatal(err)
		}

		_, err = index.GetDataByLabel(999)
		if err == nil {
			t.Error("expected error for missing label")
		}
	})

	t.Run("AddPoints_wrong_dimension", func(t *testing.T) {
		index, err := New(8, 16, 200, 42, 10, L2, false)
		if err != nil {
			t.Fatal(err)
		}
		defer index.Free()

		wrongDimVec := [][]float32{{1.0, 2.0}} // dim=2, index expects 8
		err = index.AddPoints(wrongDimVec, []uint64{0}, 1, false)
		if err == nil {
			t.Error("expected error for wrong dimensions")
		}
	})

	t.Run("Load_nonexistent_file", func(t *testing.T) {
		_, err := Load("/tmp/nonexistent_hnswgo_test.data", L2, 8, 100, false)
		if err == nil {
			t.Error("expected error for nonexistent file")
		}
	})
}

func TestConcurrentAccess(t *testing.T) {
	testDim := 8
	index, err := New(testDim, 16, 200, 42, 1000, L2, false)
	if err != nil {
		t.Fatal(err)
	}
	defer index.Free()

	// Pre-populate
	points, labels := randomPoints(testDim, 0, 100)
	if err := index.AddPoints(points, labels, 1, false); err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup

	// Concurrent searches
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			query := genQuery(testDim, 5)
			_, err := index.SearchKNN(query, 3, 1)
			if err != nil {
				t.Error(err)
			}
		}()
	}

	// Concurrent adds
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(batch int) {
			defer wg.Done()
			pts, lbls := randomPoints(testDim, 100+batch*10, 10)
			_ = index.AddPoints(pts, lbls, 1, false)
		}(i)
	}

	wg.Wait()
}

func TestMultipleSpaceTypes(t *testing.T) {
	for _, st := range []struct {
		name      string
		spaceType SpaceType
	}{
		{"L2", L2},
		{"IP", IP},
		{"Cosine", Cosine},
	} {
		t.Run(st.name, func(t *testing.T) {
			index, err := New(8, 16, 200, 42, 100, st.spaceType, false)
			if err != nil {
				t.Fatal(err)
			}
			defer index.Free()

			points, labels := randomPoints(8, 0, 50)
			if err := index.AddPoints(points, labels, 1, false); err != nil {
				t.Fatal(err)
			}

			query := genQuery(8, 3)
			results, err := index.SearchKNN(query, 5, 1)
			if err != nil {
				t.Fatal(err)
			}

			if len(results) != 3 {
				t.Fatalf("expected 3 result rows, got %d", len(results))
			}
		})
	}
}

func TestDoubleFree(t *testing.T) {
	index, err := New(8, 16, 200, 42, 10, L2, false)
	if err != nil {
		t.Fatal(err)
	}

	index.Free()
	index.Free() // should not panic
}

func TestUnmarkDeleted(t *testing.T) {
	testDim := 8
	index, err := New(testDim, 16, 200, 42, 100, L2, false)
	if err != nil {
		t.Fatal(err)
	}
	defer index.Free()

	points, labels := randomPoints(testDim, 0, 50)
	if err := index.AddPoints(points, labels, 1, false); err != nil {
		t.Fatal(err)
	}

	if err := index.MarkDeleted(0); err != nil {
		t.Fatal(err)
	}

	if err := index.UnmarkDeleted(0); err != nil {
		t.Fatal(err)
	}

	// Should be retrievable again
	vec, err := index.GetDataByLabel(0)
	if err != nil {
		t.Fatal("expected to find undeleted label 0:", err)
	}
	if len(vec) != testDim {
		t.Fatalf("expected dim %d, got %d", testDim, len(vec))
	}
}

func TestUseAfterFree(t *testing.T) {
	index, err := New(8, 16, 200, 42, 100, L2, false)
	if err != nil {
		t.Fatal(err)
	}

	// Add some data before freeing
	pts, lbls := randomPoints(8, 0, 10)
	if err := index.AddPoints(pts, lbls, 1, false); err != nil {
		t.Fatal(err)
	}

	index.Free()

	// All methods should return ErrIndexClosed, not panic
	_, err = index.SearchKNN([][]float32{make([]float32, 8)}, 1, 1)
	if !errors.Is(err, ErrIndexClosed) {
		t.Errorf("SearchKNN: expected ErrIndexClosed, got %v", err)
	}

	err = index.AddPoints([][]float32{make([]float32, 8)}, []uint64{0}, 1, false)
	if !errors.Is(err, ErrIndexClosed) {
		t.Errorf("AddPoints: expected ErrIndexClosed, got %v", err)
	}

	_, err = index.GetDataByLabel(0)
	if !errors.Is(err, ErrIndexClosed) {
		t.Errorf("GetDataByLabel: expected ErrIndexClosed, got %v", err)
	}

	err = index.MarkDeleted(0)
	if !errors.Is(err, ErrIndexClosed) {
		t.Errorf("MarkDeleted: expected ErrIndexClosed, got %v", err)
	}

	err = index.UnmarkDeleted(0)
	if !errors.Is(err, ErrIndexClosed) {
		t.Errorf("UnmarkDeleted: expected ErrIndexClosed, got %v", err)
	}

	err = index.ResizeIndex(200)
	if !errors.Is(err, ErrIndexClosed) {
		t.Errorf("ResizeIndex: expected ErrIndexClosed, got %v", err)
	}

	err = index.Save("/tmp/test_closed.db")
	if !errors.Is(err, ErrIndexClosed) {
		t.Errorf("Save: expected ErrIndexClosed, got %v", err)
	}

	err = index.SetEf(100)
	if !errors.Is(err, ErrIndexClosed) {
		t.Errorf("SetEf: expected ErrIndexClosed, got %v", err)
	}

	_, err = index.GetMaxElements()
	if !errors.Is(err, ErrIndexClosed) {
		t.Errorf("GetMaxElements: expected ErrIndexClosed, got %v", err)
	}

	_, err = index.GetCurrentCount()
	if !errors.Is(err, ErrIndexClosed) {
		t.Errorf("GetCurrentCount: expected ErrIndexClosed, got %v", err)
	}

	_, err = index.IndexFileSize()
	if !errors.Is(err, ErrIndexClosed) {
		t.Errorf("IndexFileSize: expected ErrIndexClosed, got %v", err)
	}

	_, err = index.GetAllowReplaceDeleted()
	if !errors.Is(err, ErrIndexClosed) {
		t.Errorf("GetAllowReplaceDeleted: expected ErrIndexClosed, got %v", err)
	}
}

func TestConcurrentFreeAndSearch(t *testing.T) {
	index, err := New(8, 16, 200, 42, 100, L2, false)
	if err != nil {
		t.Fatal(err)
	}

	points, labels := randomPoints(8, 0, 50)
	if err := index.AddPoints(points, labels, 1, false); err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		query := genQuery(8, 1)
		index.SearchKNN(query, 3, 1) // may get ErrIndexClosed, that's fine
	}()

	go func() {
		defer wg.Done()
		index.Free()
	}()

	wg.Wait()
	// Success = no panic or data race
}

// TestSearchKnnConcurrencyZero verifies that passing concurrency=0 with Cosine
// space does not cause a buffer overflow in the norm_array allocation.
func TestSearchKnnConcurrencyZero(t *testing.T) {
	index, err := New(8, 16, 200, 42, 100, Cosine, false)
	if err != nil {
		t.Fatal(err)
	}
	defer index.Free()

	points, labels := randomPoints(8, 0, 50)
	if err := index.AddPoints(points, labels, 1, false); err != nil {
		t.Fatal(err)
	}

	query := genQuery(8, 5)
	results, err := index.SearchKNN(query, 3, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 5 {
		t.Fatalf("expected 5 result rows, got %d", len(results))
	}
}

// TestAddPointsConcurrencyZero verifies that passing concurrency=0 with Cosine
// space does not cause a buffer overflow in the norm_array allocation.
func TestAddPointsConcurrencyZero(t *testing.T) {
	index, err := New(8, 16, 200, 42, 100, Cosine, false)
	if err != nil {
		t.Fatal(err)
	}
	defer index.Free()

	points, labels := randomPoints(8, 0, 50)
	if err := index.AddPoints(points, labels, 0, false); err != nil {
		t.Fatal(err)
	}

	count, err := index.GetCurrentCount()
	if err != nil {
		t.Fatal(err)
	}
	if count != 50 {
		t.Fatalf("expected 50 elements, got %d", count)
	}
}

// TestSearchKnnTopKGuard verifies that SearchKNN rejects topK > currentCount
// at the Go level with a clear error message.
func TestSearchKnnTopKGuard(t *testing.T) {
	index, err := New(8, 16, 200, 42, 100, L2, false)
	if err != nil {
		t.Fatal(err)
	}
	defer index.Free()

	// Add only 3 points
	points, labels := randomPoints(8, 0, 3)
	if err := index.AddPoints(points, labels, 1, false); err != nil {
		t.Fatal(err)
	}

	// Search for topK=10 but only 3 elements exist — Go guard catches this
	query := genQuery(8, 1)
	_, err = index.SearchKNN(query, 10, 1)
	if err == nil {
		t.Fatal("expected error when topK > number of indexed elements")
	}
	if !strings.Contains(err.Error(), "larger than the number of elements") {
		t.Errorf("expected topK guard message, got: %s", err.Error())
	}
}

// TestSearchKnnErrorPropagation verifies that C++ search errors are propagated
// to Go with a meaningful message instead of being lost to stderr.
func TestSearchKnnErrorPropagation(t *testing.T) {
	index, err := New(8, 4, 8, 42, 100, L2, false)
	if err != nil {
		t.Fatal(err)
	}
	defer index.Free()

	// Add 10 points with minimal M and efConstruction
	points, labels := randomPoints(8, 0, 10)
	if err := index.AddPoints(points, labels, 1, false); err != nil {
		t.Fatal(err)
	}

	// Set ef=1 (very small) — may cause search to return fewer than topK results
	if err := index.SetEf(1); err != nil {
		t.Fatal(err)
	}

	// This may or may not trigger the C++ error depending on graph connectivity.
	// The point is: if searchKnn fails, the error message should be meaningful.
	query := genQuery(8, 1)
	_, err = index.SearchKNN(query, 10, 1)
	if err != nil {
		// If it fails, the error should contain the C++ message, not "unknown error"
		if strings.Contains(err.Error(), "unknown error") {
			t.Errorf("expected meaningful error message, got: %s", err.Error())
		}
	}
	// If it succeeds, that's also fine — the test validates the error path works
}

// TestGetAllowReplaceDeletedClosed verifies that GetAllowReplaceDeleted returns
// ErrIndexClosed on a freed index instead of silently returning false.
func TestGetAllowReplaceDeletedClosed(t *testing.T) {
	index, err := New(8, 16, 200, 42, 10, L2, false)
	if err != nil {
		t.Fatal(err)
	}
	index.Free()

	_, err = index.GetAllowReplaceDeleted()
	if !errors.Is(err, ErrIndexClosed) {
		t.Errorf("expected ErrIndexClosed, got %v", err)
	}
}

// TestIndexFileSizeTryCatch verifies that IndexFileSize is wrapped in try/catch
// and returns proper errors. Also tests the closed-index path.
func TestIndexFileSizeTryCatch(t *testing.T) {
	index, err := New(8, 16, 200, 42, 100, L2, false)
	if err != nil {
		t.Fatal(err)
	}

	// IndexFileSize on a valid index should succeed
	points, labels := randomPoints(8, 0, 10)
	if err := index.AddPoints(points, labels, 1, false); err != nil {
		t.Fatal(err)
	}

	size, err := index.IndexFileSize()
	if err != nil {
		t.Fatal(err)
	}
	if size == 0 {
		t.Error("expected non-zero index file size")
	}

	// IndexFileSize on a closed index should return ErrIndexClosed
	index.Free()
	_, err = index.IndexFileSize()
	if !errors.Is(err, ErrIndexClosed) {
		t.Errorf("expected ErrIndexClosed, got %v", err)
	}
}

func randomPoints(dim int, startLabel int, batchSize int) ([][]float32, []uint64) {
	points := make([][]float32, batchSize)
	labels := make([]uint64, 0)

	for i := 0; i < batchSize; i++ {
		v := make([]float32, dim)
		for i := range v {
			v[i] = rand.Float32()
		}
		points[i] = v
		labels = append(labels, uint64(startLabel+i))
	}

	return points, labels
}

func genQuery(dim int, size int) [][]float32 {
	points := make([][]float32, size)

	for i := 0; i < size; i++ {
		v := make([]float32, dim)
		for i := range v {
			v[i] = rand.Float32()
		}
		points[i] = v
	}

	return points
}

func pathExists(path string) bool {
	stat, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		return false
	}

	if err == nil || stat != nil {
		return true
	}

	return false
}

func deleteDB() error {
	if pathExists(testVectorDB) {
		return os.Remove(testVectorDB)
	}

	return nil
}

func randomPoint(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}
