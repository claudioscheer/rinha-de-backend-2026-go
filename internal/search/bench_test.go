package search

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
)

// BenchmarkFraudScore_Real exercises the hot path against the production
// references.bin (~3 M vectors). Skipped automatically when the blob is not
// present in the working tree.
func BenchmarkFraudScore_Real(b *testing.B) {
	dir, ok := findResources()
	if !ok {
		b.Skip("resources/references.bin not available; run cmd/compile-dataset first")
	}
	ds, err := dataset.Load(dir)
	if err != nil {
		b.Fatalf("load: %v", err)
	}
	b.Cleanup(func() { _ = ds.Close() })
	b.Logf("dataset: %d records", ds.Size())

	var q [dataset.VectorDim]uint8
	for i := range q {
		q[i] = 128
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = FraudScore(ds, q)
	}
}

func findResources() (string, bool) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Join(filepath.Dir(file), "..", "..")
	bin := filepath.Join(root, "resources", "references.bin")
	if _, err := os.Stat(bin); err != nil {
		return "", false
	}
	return filepath.Join(root, "resources"), true
}
