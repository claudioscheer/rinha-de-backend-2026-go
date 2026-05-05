package search

import (
	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
)

const (
	// K is the number of nearest neighbors used in the majority vote.
	K = 5
	// Nprobe is how many of the closest IVF clusters are scanned per query.
	// Higher values raise recall and latency. With 1024 production clusters
	// this scans ~0.8 % of the dataset.
	Nprobe = 8
)

const sentinelDist int32 = 1 << 30

type neighbor struct {
	dist int32
	idx  int32
}

// FraudScore returns the fraction of frauds among the K nearest neighbors of
// query in the reference dataset. When the dataset carries an IVF index
// (NumClusters > 0), only the closest Nprobe clusters are scanned.
func FraudScore(ds *dataset.Dataset, query [dataset.VectorDim]uint8) float64 {
	var top [K]neighbor
	for i := range top {
		top[i].dist = sentinelDist
	}
	worstIdx := 0
	worstDist := top[0].dist

	if ds.NumClusters == 0 {
		scanRange(ds.Vectors, 0, ds.Size(), &query, &top, &worstIdx, &worstDist)
	} else {
		scanIVF(ds, &query, &top, &worstIdx, &worstDist)
	}

	count := 0
	for k := range top {
		if top[k].dist < sentinelDist && ds.IsFraud(int(top[k].idx)) {
			count++
		}
	}
	return float64(count) / float64(K)
}

// scanIVF ranks the centroids and scans the top Nprobe clusters into the
// running K-NN state.
func scanIVF(ds *dataset.Dataset, query *[dataset.VectorDim]uint8, top *[K]neighbor, worstIdxP *int, worstDistP *int32) {
	nc := ds.NumClusters
	nprobe := Nprobe
	if nprobe > nc {
		nprobe = nc
	}

	var topC [Nprobe]neighbor
	for i := 0; i < nprobe; i++ {
		topC[i].dist = sentinelDist
	}
	worstC := 0
	worstCDist := topC[0].dist

	centroids := ds.Centroids
	for c := 0; c < nc; c++ {
		cv := (*[dataset.VectorDim]uint8)(centroids[c*dataset.VectorDim:])
		d := dist14(cv, query)
		if d < worstCDist {
			topC[worstC] = neighbor{dist: d, idx: int32(c)}
			worstC = 0
			for k := 1; k < nprobe; k++ {
				if topC[k].dist > topC[worstC].dist {
					worstC = k
				}
			}
			worstCDist = topC[worstC].dist
		}
	}

	offsets := ds.ClusterOffsets
	vectors := ds.Vectors
	for p := 0; p < nprobe; p++ {
		c := int(topC[p].idx)
		start := int(offsets[c])
		end := int(offsets[c+1])
		scanRange(vectors, start, end, query, top, worstIdxP, worstDistP)
	}
}

// scanRange runs the K-NN inner loop over vectors[start..end). The K-NN
// state is taken by pointer so multiple ranges (one per IVF probe) can extend
// the same set of nearest neighbors.
//
// Distances stay in int32: max per-dim diff is 255, per-dim squared diff fits
// in int32 (255² = 65 025), and the sum of 14 such terms maxes out at ~910 k
// — well below the int32 ceiling.
func scanRange(vectors []uint8, start, end int, query *[dataset.VectorDim]uint8, top *[K]neighbor, worstIdxP *int, worstDistP *int32) {
	worstIdx := *worstIdxP
	worstDist := *worstDistP

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

	for i := start; i < end; i++ {
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

	*worstIdxP = worstIdx
	*worstDistP = worstDist
}

// dist14 returns the squared Euclidean distance between two 14-dim uint8
// vectors. Used only on the cold centroid-ranking path; the hot per-vector
// loop in scanRange inlines its own distance computation.
func dist14(a, b *[dataset.VectorDim]uint8) int32 {
	d0 := int32(a[0]) - int32(b[0])
	d1 := int32(a[1]) - int32(b[1])
	d2 := int32(a[2]) - int32(b[2])
	d3 := int32(a[3]) - int32(b[3])
	d4 := int32(a[4]) - int32(b[4])
	d5 := int32(a[5]) - int32(b[5])
	d6 := int32(a[6]) - int32(b[6])
	d7 := int32(a[7]) - int32(b[7])
	d8 := int32(a[8]) - int32(b[8])
	d9 := int32(a[9]) - int32(b[9])
	d10 := int32(a[10]) - int32(b[10])
	d11 := int32(a[11]) - int32(b[11])
	d12 := int32(a[12]) - int32(b[12])
	d13 := int32(a[13]) - int32(b[13])
	return d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
		d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13
}
