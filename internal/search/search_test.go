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

// TestFraudScore_IVF exercises the IVF code path with a hand-built index.
// Two clusters, well-separated. A query near one cluster centroid must pick
// the K nearest records inside that cluster, not the other.
func TestFraudScore_IVF(t *testing.T) {
	// Cluster 0 (records 0..4): all near +1, all fraud.
	// Cluster 1 (records 5..9): all near -1, all legit.
	// Centroid 0 ≈ +1, Centroid 1 ≈ -1.
	vecs := make([]uint8, 10*dataset.VectorDim)
	frauds := make([]uint8, (10+7)/8)
	for i := 0; i < 5; i++ {
		row := vecs[i*dataset.VectorDim : (i+1)*dataset.VectorDim]
		for d := range row {
			row[d] = dataset.Quantize(1)
		}
		frauds[i>>3] |= 1 << uint(i&7) // fraud
	}
	for i := 5; i < 10; i++ {
		row := vecs[i*dataset.VectorDim : (i+1)*dataset.VectorDim]
		for d := range row {
			row[d] = dataset.Quantize(-1)
		}
	}

	centroids := make([]uint8, 2*dataset.VectorDim)
	for d := 0; d < dataset.VectorDim; d++ {
		centroids[d] = dataset.Quantize(1)
		centroids[dataset.VectorDim+d] = dataset.Quantize(-1)
	}
	offsets := []uint32{0, 5, 10}

	ds := dataset.NewForTestIVF(vecs, frauds, centroids, offsets, 10, 2)

	// Query near cluster 0 → all 5 nearest are fraud.
	var q [dataset.VectorDim]uint8
	for d := range q {
		q[d] = dataset.Quantize(1)
	}
	if got := FraudScore(ds, q); got != 1.0 {
		t.Fatalf("near-fraud query: got %v, want 1.0", got)
	}

	// Query near cluster 1 → all 5 nearest are legit.
	for d := range q {
		q[d] = dataset.Quantize(-1)
	}
	if got := FraudScore(ds, q); got != 0.0 {
		t.Fatalf("near-legit query: got %v, want 0.0", got)
	}
}

// TestFraudScore_IVF_NprobeOverflow checks the safety clamp when the index
// has fewer clusters than the configured Nprobe.
func TestFraudScore_IVF_NprobeOverflow(t *testing.T) {
	// 1 cluster, 5 records, all fraud. nprobe is clamped to 1.
	vecs := make([]uint8, 5*dataset.VectorDim)
	frauds := make([]uint8, 1)
	for i := 0; i < 5; i++ {
		frauds[0] |= 1 << uint(i)
	}
	centroids := make([]uint8, dataset.VectorDim)
	for d := range centroids {
		centroids[d] = dataset.Quantize(0)
	}
	offsets := []uint32{0, 5}
	ds := dataset.NewForTestIVF(vecs, frauds, centroids, offsets, 5, 1)

	var q [dataset.VectorDim]uint8
	if got := FraudScore(ds, q); got != 1.0 {
		t.Fatalf("got %v, want 1.0", got)
	}
}
