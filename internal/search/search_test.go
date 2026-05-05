package search

import (
	"testing"

	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
)

// Build a dataset from a small list of float vectors + labels. Each vector
// is quantized to uint8 and padded to VectorDim (14) with zeros.
func buildDataset(vecs [][]float32, labels []bool) *dataset.Dataset {
	flat := make([]uint8, 0, len(vecs)*dataset.VectorDim)
	for _, v := range vecs {
		row := make([]float32, dataset.VectorDim)
		copy(row, v)
		for _, x := range row {
			flat = append(flat, dataset.Quantize(x))
		}
	}
	frauds := make([]uint8, (len(labels)+7)/8)
	for i, fr := range labels {
		if fr {
			frauds[i>>3] |= 1 << uint(i&7)
		}
	}
	return dataset.NewForTest(flat, frauds, len(labels))
}

func quantQuery(v []float32) [dataset.VectorDim]uint8 {
	var q [dataset.VectorDim]uint8
	for i := 0; i < dataset.VectorDim; i++ {
		if i < len(v) {
			q[i] = dataset.Quantize(v[i])
		} else {
			q[i] = dataset.Quantize(0)
		}
	}
	return q
}

func TestFraudScore_AllFraud(t *testing.T) {
	ds := buildDataset(
		[][]float32{{0}, {0}, {0}, {0}, {0}},
		[]bool{true, true, true, true, true},
	)
	q := quantQuery(nil)
	got := FraudScore(ds, q)
	if got != 1.0 {
		t.Fatalf("got %v want 1.0", got)
	}
}

func TestFraudScore_AllLegit(t *testing.T) {
	ds := buildDataset(
		[][]float32{{0}, {0}, {0}, {0}, {0}},
		[]bool{false, false, false, false, false},
	)
	q := quantQuery(nil)
	got := FraudScore(ds, q)
	if got != 0.0 {
		t.Fatalf("got %v want 0.0", got)
	}
}

func TestFraudScore_Mixed(t *testing.T) {
	ds := buildDataset(
		[][]float32{{0}, {0}, {0}, {0}, {0}},
		[]bool{true, true, true, false, false},
	)
	q := quantQuery(nil)
	got := FraudScore(ds, q)
	if got != 0.6 {
		t.Fatalf("got %v want 0.6", got)
	}
}

// With more than K neighbors, only the K nearest contribute. Fraud points
// near origin must be picked over far legit points.
func TestFraudScore_PicksNearest(t *testing.T) {
	ds := buildDataset(
		[][]float32{
			{0}, {0}, {0}, {0}, {0}, // close to query
			{1}, {1}, {1}, {1}, {1}, // far from query (clamped to 1)
		},
		[]bool{
			true, true, true, true, true,
			false, false, false, false, false,
		},
	)
	q := quantQuery(nil)
	got := FraudScore(ds, q)
	if got != 1.0 {
		t.Fatalf("got %v want 1.0 (5 nearest are frauds)", got)
	}
}

// 7 records: 5 legits slightly farther, 2 frauds closer. Top 5 nearest
// includes both frauds + 3 legits → 2/5 = 0.4.
func TestFraudScore_PicksNearestLegit(t *testing.T) {
	ds := buildDataset(
		[][]float32{
			{0.1}, {0.1}, {0.1}, {0.1}, {0.1},
			{0.0}, {0.0},
		},
		[]bool{false, false, false, false, false, true, true},
	)
	q := quantQuery(nil)
	got := FraudScore(ds, q)
	if got != 0.4 {
		t.Fatalf("got %v want 0.4 (2 frauds in 5 nearest)", got)
	}
}
