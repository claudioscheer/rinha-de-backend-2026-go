package search

import (
	"testing"

	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
)

// Build a dataset from a small list of vectors + labels, padding each vector
// to VectorDim (14) with zeros so every test gets a valid layout.
func buildDataset(vecs [][]float32, labels []bool) *dataset.Dataset {
	flat := make([]float32, 0, len(vecs)*dataset.VectorDim)
	for _, v := range vecs {
		row := make([]float32, dataset.VectorDim)
		copy(row, v)
		flat = append(flat, row...)
	}
	return &dataset.Dataset{
		Vectors: flat,
		Frauds:  labels,
	}
}

// Five identical-distance fraud neighbors → score 1.0.
func TestFraudScore_AllFraud(t *testing.T) {
	ds := buildDataset(
		[][]float32{{0}, {0}, {0}, {0}, {0}},
		[]bool{true, true, true, true, true},
	)
	var q [dataset.VectorDim]float32
	got := FraudScore(ds, q)
	if got != 1.0 {
		t.Fatalf("got %v want 1.0", got)
	}
}

// Five legit neighbors → score 0.0.
func TestFraudScore_AllLegit(t *testing.T) {
	ds := buildDataset(
		[][]float32{{0}, {0}, {0}, {0}, {0}},
		[]bool{false, false, false, false, false},
	)
	var q [dataset.VectorDim]float32
	got := FraudScore(ds, q)
	if got != 0.0 {
		t.Fatalf("got %v want 0.0", got)
	}
}

// 3 frauds out of 5 → 0.6.
func TestFraudScore_Mixed(t *testing.T) {
	ds := buildDataset(
		[][]float32{{0}, {0}, {0}, {0}, {0}},
		[]bool{true, true, true, false, false},
	)
	var q [dataset.VectorDim]float32
	got := FraudScore(ds, q)
	if got != 0.6 {
		t.Fatalf("got %v want 0.6", got)
	}
}

// With more than K neighbors, only the K nearest contribute. Place 5 close
// fraud points and 5 far legit points; result must be 1.0 (all 5 nearest
// are frauds), proving that far points are excluded.
func TestFraudScore_PicksNearest(t *testing.T) {
	ds := buildDataset(
		[][]float32{
			{0.0}, {0.0}, {0.0}, {0.0}, {0.0}, // close to query
			{10}, {10}, {10}, {10}, {10}, // far from query
		},
		[]bool{
			true, true, true, true, true,
			false, false, false, false, false,
		},
	)
	var q [dataset.VectorDim]float32
	got := FraudScore(ds, q)
	if got != 1.0 {
		t.Fatalf("got %v want 1.0 (5 nearest are frauds)", got)
	}
}

// Symmetric case: closest 5 are legit even though there are nearby frauds
// further out.
func TestFraudScore_PicksNearestLegit(t *testing.T) {
	ds := buildDataset(
		[][]float32{
			{0.1}, {0.1}, {0.1}, {0.1}, {0.1}, // close to query
			{0.0}, {0.0}, // even closer but only 2 of them — wait, they'd be picked
		},
		[]bool{false, false, false, false, false, true, true},
	)
	// With 7 points, the 5 nearest are the 2 frauds at 0.0 and 3 of the
	// legits at 0.1, so 2 frauds out of 5 = 0.4.
	var q [dataset.VectorDim]float32
	got := FraudScore(ds, q)
	if got != 0.4 {
		t.Fatalf("got %v want 0.4 (2 frauds in 5 nearest)", got)
	}
}
