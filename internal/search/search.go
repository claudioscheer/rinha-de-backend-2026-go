package search

import (
	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
)

const K = 5

// FraudScore returns the fraction of frauds among the K nearest neighbors of
// query in the reference dataset, using squared Euclidean distance with a
// brute-force scan.
func FraudScore(ds *dataset.Dataset, query [dataset.VectorDim]float32) float64 {
	type neighbor struct {
		dist  float32
		fraud bool
	}

	// Bounded array, K is small.
	var top [K]neighbor
	for i := range top {
		top[i].dist = -1 // sentinel: empty slot
	}
	worstIdx := 0
	filled := 0

	vectors := ds.Vectors
	frauds := ds.Frauds
	n := len(frauds)

	for i := 0; i < n; i++ {
		base := i * dataset.VectorDim
		var d float32
		for j := 0; j < dataset.VectorDim; j++ {
			diff := vectors[base+j] - query[j]
			d += diff * diff
		}

		if filled < K {
			top[filled] = neighbor{dist: d, fraud: frauds[i]}
			filled++
			if filled == K {
				worstIdx = 0
				for k := 1; k < K; k++ {
					if top[k].dist > top[worstIdx].dist {
						worstIdx = k
					}
				}
			}
			continue
		}

		if d < top[worstIdx].dist {
			top[worstIdx] = neighbor{dist: d, fraud: frauds[i]}
			worstIdx = 0
			for k := 1; k < K; k++ {
				if top[k].dist > top[worstIdx].dist {
					worstIdx = k
				}
			}
		}
	}

	count := 0
	for _, t := range top {
		if t.fraud {
			count++
		}
	}
	return float64(count) / float64(K)
}
