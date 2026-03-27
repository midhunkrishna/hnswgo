package hnswgo

import (
	"errors"
	"math"
	"math/rand"
	"os"
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

func TestNewIndex(t *testing.T) {
	var maxElements uint64 = batchSize * 1

	idx, err := newTestIndex(1, true)
	if err != nil {
		t.Fatal(err)
	}
	defer idx.Free()

	if idx.GetMaxElements() != maxElements {
		t.Fail()
	}

	if idx.GetAllowReplaceDeleted() != true {
		t.Fail()
	}

	if idx.GetCurrentCount() != maxElements {
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
	index.SetEf(efConstruction)
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

	if idx.GetMaxElements() != maxElements {
		t.Fail()
	}

	if idx.GetCurrentCount() != maxElements {
		t.Fail()
	}

	if idx.GetAllowReplaceDeleted() != false {
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
	if idx.GetMaxElements() != maxElements*2 {
		t.Fail()
	}

	if idx.GetCurrentCount() != maxElements {
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

	if !index.GetAllowReplaceDeleted() {
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
