# hnswgo

[![Tests](https://github.com/midhunkrishna/hnswgo/actions/workflows/test.yml/badge.svg)](https://github.com/midhunkrishna/hnswgo/actions/workflows/test.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/midhunkrishna/hnswgo.svg)](https://pkg.go.dev/github.com/midhunkrishna/hnswgo)
[![Go Report Card](https://goreportcard.com/badge/github.com/midhunkrishna/hnswgo)](https://goreportcard.com/report/github.com/midhunkrishna/hnswgo)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Go bindings for [hnswlib](https://github.com/nmslib/hnswlib) — a fast, production-ready library for approximate nearest neighbor (ANN) search using Hierarchical Navigable Small World graphs.

## Features

- **Three distance metrics** — L2 (Euclidean), Inner Product, and Cosine similarity
- **Thread-safe** — all operations protected by `sync.RWMutex`, safe for concurrent goroutines
- **Batch operations** — parallel insert and search with configurable concurrency
- **Persistence** — save and load indices to/from disk
- **Dynamic indices** — resize capacity, mark/unmark deletions at runtime
- **Zero dependencies** — only Go stdlib + vendored hnswlib C++ headers

## Prerequisites

- Go 1.21+
- CGO enabled (`CGO_ENABLED=1`)
- C/C++ compiler with C++11 support (gcc or clang)

## Installation

```bash
go get github.com/midhunkrishna/hnswgo
```

## Quick Start

```go
package main

import (
	"fmt"
	"math/rand"

	"github.com/midhunkrishna/hnswgo"
)

func main() {
	// Create an index
	index, err := hnswgo.New(
		128,             // vector dimension
		16,              // M — max connections per node
		200,             // efConstruction — build-time accuracy
		42,              // random seed
		10000,           // max elements
		hnswgo.Cosine,   // distance metric
		false,           // allow replace deleted
	)
	if err != nil {
		panic(err)
	}
	defer index.Free()

	// Insert vectors
	vectors := make([][]float32, 1000)
	labels := make([]uint64, 1000)
	for i := range vectors {
		vectors[i] = make([]float32, 128)
		for j := range vectors[i] {
			vectors[i][j] = rand.Float32()
		}
		labels[i] = uint64(i)
	}

	if err := index.AddPoints(vectors, labels, 4, false); err != nil {
		panic(err)
	}

	// Search
	query := [][]float32{vectors[0]} // find nearest neighbors of first vector
	results, err := index.SearchKNN(query, 5, 1)
	if err != nil {
		panic(err)
	}

	for _, r := range results[0] {
		fmt.Printf("label: %d, distance: %f\n", r.Label, r.Distance)
	}

	// Save to disk
	if err := index.Save("my_index.bin"); err != nil {
		panic(err)
	}
}
```

See [`example/example.go`](example/example.go) for a more complete example.

## API

### Creating and Loading

```go
// Create a new index
index, err := hnswgo.New(dim, M, efConstruction, randSeed, maxElements, spaceType, allowReplaceDeleted)

// Load from disk
index, err := hnswgo.Load(path, spaceType, dim, maxElements, allowReplaceDeleted)

// Release resources (idempotent, safe to call multiple times)
err := index.Free()
```

### Inserting and Deleting

```go
// Batch insert with concurrent workers
err := index.AddPoints(vectors, labels, concurrency, replaceDeleted)

// Soft-delete / restore
err := index.MarkDeleted(label)
err := index.UnmarkDeleted(label)
```

### Searching

```go
// Batch KNN search — returns one result slice per query vector
results, err := index.SearchKNN(queryVectors, topK, concurrency)

// Retrieve a stored vector by label
vector, err := index.GetDataByLabel(label)
```

> **Note:** In Cosine space, `GetDataByLabel` returns the normalized vector, not the original input.

### Configuration

```go
// Set query-time accuracy/speed tradeoff (not persisted — set after Load)
err := index.SetEf(ef)

// Resize index capacity
err := index.ResizeIndex(newSize)

// Save index to disk
err := index.Save(path)
```

### Index Info

```go
count, err := index.GetCurrentCount()
capacity, err := index.GetMaxElements()
replaceable, err := index.GetAllowReplaceDeleted()
fileSize, err := index.IndexFileSize()
```

All methods return `hnswgo.ErrIndexClosed` after `Free()` has been called.

## Parameters

### Index Construction

| Parameter | Type | Description |
|---|---|---|
| `dim` | `int` | Vector dimensionality |
| `M` | `int` | Max connections per node. Higher = better recall, more memory. See [ALGO_PARAMS.md](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md) |
| `efConstruction` | `int` | Build-time search width. Higher = better index quality, slower builds. See [ALGO_PARAMS.md](https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md) |
| `randSeed` | `int` | Random seed for reproducibility |
| `maxElements` | `uint64` | Maximum index capacity (can be resized later) |
| `allowReplaceDeleted` | `bool` | Allow new inserts to reuse slots of deleted elements |

### Query-Time

| Parameter | Type | Description |
|---|---|---|
| `ef` | `int` | Search accuracy/speed tradeoff. Must be >= `topK`. Higher = better recall, slower queries |
| `topK` | `int` | Number of nearest neighbors to return |
| `concurrency` | `int` | Number of parallel workers for batch operations |

### Distance Metrics

| SpaceType | Metric | Typical Use |
|---|---|---|
| `hnswgo.L2` | Euclidean distance | General-purpose, geometric data |
| `hnswgo.IP` | Inner product | When vectors are pre-normalized |
| `hnswgo.Cosine` | Cosine similarity | Text/image embeddings, semantic search |

## Testing

```bash
go test -v -race ./...
```

The test suite covers core operations, error handling, edge cases, concurrency safety (with `-race`), and round-trip persistence across all space types.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-change`)
3. Commit your changes
4. Push to your branch (`git push origin feature/my-change`)
5. Open a Pull Request

Please ensure tests pass with the race detector enabled (`go test -race ./...`) before submitting.

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## Acknowledgments

- Forked from [oligo/hnswgo](https://github.com/oligo/hnswgo) — the original Go bindings for hnswlib
- [hnswlib](https://github.com/nmslib/hnswlib) by Yury Malkov and others — the underlying C++ ANN library
- Malkov, Yu A., and D. A. Yashunin. "[Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs](https://arxiv.org/abs/1603.09320)." IEEE Transactions on Pattern Analysis and Machine Intelligence, 2018.
