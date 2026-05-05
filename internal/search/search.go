package search

import (
	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
)

const K = 5

// FraudScore returns the fraction of frauds among the K nearest neighbors of
// query in the reference dataset, using squared Euclidean distance with a
// brute-force scan over uint8-quantized vectors.
//
// Distances stay in int32: max per-dim diff is 255, so per-dim squared diff
// fits in int32 (255² = 65025), and the sum of 14 such terms maxes out at
// ~910k — well below the int32 ceiling.
func FraudScore(ds *dataset.Dataset, query [dataset.VectorDim]uint8) float64 {
	type neighbor struct {
		dist int32
		idx  int32
	}

	// Pre-fill all K slots with a sentinel distance so the scan has a
	// single uniform path: every record either replaces the worst slot or
	// is discarded. Avoids a separate "filled < K" branch in the hot loop.
	const sentinelDist int32 = 1 << 30
	var top [K]neighbor
	for i := range top {
		top[i].dist = sentinelDist
	}
	worstIdx := 0

	q0 := int32(query[0])
	q1 := int32(query[1])
	q2 := int32(query[2])
	q3 := int32(query[3])
	q4 := int32(query[4])
	q5 := int32(query[5])
	q6 := int32(query[6])
	q7 := int32(query[7])
	q8 := int32(query[8])
	q9 := int32(query[9])
	q10 := int32(query[10])
	q11 := int32(query[11])
	q12 := int32(query[12])
	q13 := int32(query[13])

	vectors := ds.Vectors
	n := ds.Size()
	worstDist := top[0].dist

	for i := 0; i < n; i++ {
		// Reslice to a fixed-size array pointer: the compiler can elide
		// bounds checks on v[0]..v[13].
		v := (*[dataset.VectorDim]uint8)(vectors[i*dataset.VectorDim:])
		d0 := int32(v[0]) - q0
		d1 := int32(v[1]) - q1
		d2 := int32(v[2]) - q2
		d3 := int32(v[3]) - q3
		d4 := int32(v[4]) - q4
		d5 := int32(v[5]) - q5
		d6 := int32(v[6]) - q6
		d7 := int32(v[7]) - q7
		d8 := int32(v[8]) - q8
		d9 := int32(v[9]) - q9
		d10 := int32(v[10]) - q10
		d11 := int32(v[11]) - q11
		d12 := int32(v[12]) - q12
		d13 := int32(v[13]) - q13
		d := d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
			d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13

		if d < worstDist {
			top[worstIdx] = neighbor{dist: d, idx: int32(i)}
			worstIdx = 0
			for k := 1; k < K; k++ {
				if top[k].dist > top[worstIdx].dist {
					worstIdx = k
				}
			}
			worstDist = top[worstIdx].dist
		}
	}

	count := 0
	for k := 0; k < K; k++ {
		if top[k].dist < sentinelDist && ds.IsFraud(int(top[k].idx)) {
			count++
		}
	}
	return float64(count) / float64(K)
}
