package search

import (
	"sort"

	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
)

const (
	// K is the number of nearest neighbors used in the majority vote.
	K = 5
	// DefaultNprobe is how many of the closest IVF clusters are scanned before
	// the provisional decision is inspected.
	DefaultNprobe = 12
	// DefaultMaxNprobe is the adaptive ceiling used when adaptive probing is
	// enabled. Production keeps this equal to DefaultNprobe to stay near the
	// p99 scoring ceiling.
	DefaultMaxNprobe = 24
	// MaxProbeLimit bounds stack storage for centroid ranking. Higher values
	// raise stack usage in the hot path.
	MaxProbeLimit = 128
)

const sentinelDist int64 = 1 << 62

// Options controls the IVF recall/latency tradeoff. The zero value maps to
// DefaultOptions.
type Options struct {
	// Nprobe is the initial number of closest clusters to scan.
	Nprobe int
	// MaxNprobe is the optional adaptive ceiling. Ignored when Adaptive is
	// false. Values above MaxProbeLimit are clamped.
	MaxNprobe int
	// Adaptive scans up to MaxNprobe only when the first pass lands on the
	// 2-vs-3 fraud boundary.
	Adaptive bool
}

var DefaultOptions = Options{
	Nprobe:    DefaultNprobe,
	MaxNprobe: DefaultMaxNprobe,
	Adaptive:  true,
}

type neighbor struct {
	dist int64
	idx  int32
}

// FraudScore returns the fraction of frauds among the K nearest neighbors of
// query in the reference dataset. When the dataset carries an IVF index
// (NumClusters > 0), it uses DefaultOptions.
func FraudScore(ds *dataset.Dataset, query [dataset.VectorDim]uint8) float64 {
	return FraudScoreWithOptions(ds, query, DefaultOptions)
}

// FraudScoreWithOptions returns the fraction of frauds among the K nearest
// neighbors of query using the supplied IVF options.
func FraudScoreWithOptions(ds *dataset.Dataset, query [dataset.VectorDim]uint8, opts Options) float64 {
	if ds.Vectors16 != nil {
		return FraudScore16WithOptions(ds, promoteQuery16(query), opts)
	}
	var top [K]neighbor
	for i := range top {
		top[i].dist = sentinelDist
	}
	worstIdx := 0
	worstDist := top[0].dist

	if ds.NumClusters == 0 {
		scanRange(ds.Vectors, 0, ds.Size(), &query, &top, &worstIdx, &worstDist)
	} else {
		scanIVF(ds, &query, &top, &worstIdx, &worstDist, opts)
	}

	count := fraudCount(ds, &top)
	return float64(count) / float64(K)
}

// FraudScore16WithOptions is the high-precision variant used by 16-bit
// compiled blobs. If the dataset is only 8-bit, the query is demoted and the
// regular path is used.
func FraudScore16WithOptions(ds *dataset.Dataset, query [dataset.VectorDim]uint16, opts Options) float64 {
	if ds.Vectors16 == nil {
		return FraudScoreWithOptions(ds, demoteQuery8(query), opts)
	}

	var top [K]neighbor
	for i := range top {
		top[i].dist = sentinelDist
	}
	worstIdx := 0
	worstDist := top[0].dist

	if ds.NumClusters == 0 {
		scanRange16(ds.Vectors16, 0, ds.Size(), &query, &top, &worstIdx, &worstDist)
	} else {
		scanIVF16(ds, &query, &top, &worstIdx, &worstDist, opts)
	}

	count := fraudCount(ds, &top)
	return float64(count) / float64(K)
}

func promoteQuery16(query [dataset.VectorDim]uint8) [dataset.VectorDim]uint16 {
	var out [dataset.VectorDim]uint16
	for i, v := range query {
		out[i] = uint16(v) * 257
	}
	return out
}

func demoteQuery8(query [dataset.VectorDim]uint16) [dataset.VectorDim]uint8 {
	var out [dataset.VectorDim]uint8
	for i, v := range query {
		out[i] = uint8((uint32(v) + 128) / 257)
	}
	return out
}

func fraudCount(ds *dataset.Dataset, top *[K]neighbor) int {
	count := 0
	for k := range top {
		if top[k].dist < sentinelDist && ds.IsFraud(int(top[k].idx)) {
			count++
		}
	}
	return count
}

// scanIVF ranks centroids, scans the initial probe set, and optionally extends
// the scan when the provisional vote is on the decision boundary.
func scanIVF(ds *dataset.Dataset, query *[dataset.VectorDim]uint8, top *[K]neighbor, worstIdxP *int, worstDistP *int64, opts Options) {
	nc := ds.NumClusters
	nprobe, maxNprobe, adaptive := normalizeOptions(nc, opts)

	var topC [MaxProbeLimit]neighbor
	for i := 0; i < maxNprobe; i++ {
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
			for k := 1; k < maxNprobe; k++ {
				if topC[k].dist > topC[worstC].dist {
					worstC = k
				}
			}
			worstCDist = topC[worstC].dist
		}
	}
	if maxNprobe > nprobe {
		sort.Slice(topC[:maxNprobe], func(i, j int) bool {
			return topC[i].dist < topC[j].dist
		})
	}

	offsets := ds.ClusterOffsets
	vectors := ds.Vectors
	for p := 0; p < nprobe; p++ {
		c := int(topC[p].idx)
		start := int(offsets[c])
		end := int(offsets[c+1])
		scanRange(vectors, start, end, query, top, worstIdxP, worstDistP)
	}
	if !adaptive || nprobe == maxNprobe {
		return
	}
	frauds := fraudCount(ds, top)
	if frauds != 2 && frauds != 3 {
		return
	}
	for p := nprobe; p < maxNprobe; p++ {
		c := int(topC[p].idx)
		start := int(offsets[c])
		end := int(offsets[c+1])
		scanRange(vectors, start, end, query, top, worstIdxP, worstDistP)
	}
}

func normalizeOptions(numClusters int, opts Options) (nprobe, maxNprobe int, adaptive bool) {
	if opts.Nprobe <= 0 {
		opts.Nprobe = DefaultNprobe
	}
	if opts.MaxNprobe <= 0 {
		opts.MaxNprobe = opts.Nprobe
	}
	if opts.Nprobe > MaxProbeLimit {
		opts.Nprobe = MaxProbeLimit
	}
	if opts.MaxNprobe > MaxProbeLimit {
		opts.MaxNprobe = MaxProbeLimit
	}
	if opts.Nprobe > opts.MaxNprobe {
		opts.MaxNprobe = opts.Nprobe
	}
	if opts.Nprobe > numClusters {
		opts.Nprobe = numClusters
	}
	if opts.MaxNprobe > numClusters {
		opts.MaxNprobe = numClusters
	}
	return opts.Nprobe, opts.MaxNprobe, opts.Adaptive && opts.MaxNprobe > opts.Nprobe
}

// scanRange runs the K-NN inner loop over vectors[start..end). The K-NN
// state is taken by pointer so multiple ranges (one per IVF probe) can extend
// the same set of nearest neighbors.
//
// Distances stay in int32: max per-dim diff is 255, per-dim squared diff fits
// in int32 (255² = 65 025), and the sum of 14 such terms maxes out at ~910 k
// — well below the int32 ceiling.
func scanRange(vectors []uint8, start, end int, query *[dataset.VectorDim]uint8, top *[K]neighbor, worstIdxP *int, worstDistP *int64) {
	worstIdx := *worstIdxP
	worstDist := *worstDistP

	q0 := int64(query[0])
	q1 := int64(query[1])
	q2 := int64(query[2])
	q3 := int64(query[3])
	q4 := int64(query[4])
	q5 := int64(query[5])
	q6 := int64(query[6])
	q7 := int64(query[7])
	q8 := int64(query[8])
	q9 := int64(query[9])
	q10 := int64(query[10])
	q11 := int64(query[11])
	q12 := int64(query[12])
	q13 := int64(query[13])

	for i := start; i < end; i++ {
		v := (*[dataset.VectorDim]uint8)(vectors[i*dataset.VectorDim:])
		d0 := int64(v[0]) - q0
		d1 := int64(v[1]) - q1
		d2 := int64(v[2]) - q2
		d3 := int64(v[3]) - q3
		d4 := int64(v[4]) - q4
		d5 := int64(v[5]) - q5
		d6 := int64(v[6]) - q6
		d7 := int64(v[7]) - q7
		d8 := int64(v[8]) - q8
		d9 := int64(v[9]) - q9
		d10 := int64(v[10]) - q10
		d11 := int64(v[11]) - q11
		d12 := int64(v[12]) - q12
		d13 := int64(v[13]) - q13
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
func dist14(a, b *[dataset.VectorDim]uint8) int64 {
	d0 := int64(a[0]) - int64(b[0])
	d1 := int64(a[1]) - int64(b[1])
	d2 := int64(a[2]) - int64(b[2])
	d3 := int64(a[3]) - int64(b[3])
	d4 := int64(a[4]) - int64(b[4])
	d5 := int64(a[5]) - int64(b[5])
	d6 := int64(a[6]) - int64(b[6])
	d7 := int64(a[7]) - int64(b[7])
	d8 := int64(a[8]) - int64(b[8])
	d9 := int64(a[9]) - int64(b[9])
	d10 := int64(a[10]) - int64(b[10])
	d11 := int64(a[11]) - int64(b[11])
	d12 := int64(a[12]) - int64(b[12])
	d13 := int64(a[13]) - int64(b[13])
	return d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
		d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13
}

func scanIVF16(ds *dataset.Dataset, query *[dataset.VectorDim]uint16, top *[K]neighbor, worstIdxP *int, worstDistP *int64, opts Options) {
	nc := ds.NumClusters
	nprobe, maxNprobe, adaptive := normalizeOptions(nc, opts)

	var topC [MaxProbeLimit]neighbor
	for i := 0; i < maxNprobe; i++ {
		topC[i].dist = sentinelDist
	}
	worstC := 0
	worstCDist := topC[0].dist

	centroids := ds.Centroids16
	for c := 0; c < nc; c++ {
		cv := (*[dataset.VectorDim]uint16)(centroids[c*dataset.VectorDim:])
		d := dist14u16(cv, query)
		if d < worstCDist {
			topC[worstC] = neighbor{dist: d, idx: int32(c)}
			worstC = 0
			for k := 1; k < maxNprobe; k++ {
				if topC[k].dist > topC[worstC].dist {
					worstC = k
				}
			}
			worstCDist = topC[worstC].dist
		}
	}
	if maxNprobe > nprobe {
		sort.Slice(topC[:maxNprobe], func(i, j int) bool {
			return topC[i].dist < topC[j].dist
		})
	}

	offsets := ds.ClusterOffsets
	vectors := ds.Vectors16
	for p := 0; p < nprobe; p++ {
		c := int(topC[p].idx)
		start := int(offsets[c])
		end := int(offsets[c+1])
		scanRange16(vectors, start, end, query, top, worstIdxP, worstDistP)
	}
	if !adaptive || nprobe == maxNprobe {
		return
	}
	frauds := fraudCount(ds, top)
	if frauds != 2 && frauds != 3 {
		return
	}
	for p := nprobe; p < maxNprobe; p++ {
		c := int(topC[p].idx)
		start := int(offsets[c])
		end := int(offsets[c+1])
		scanRange16(vectors, start, end, query, top, worstIdxP, worstDistP)
	}
}

func scanRange16(vectors []uint16, start, end int, query *[dataset.VectorDim]uint16, top *[K]neighbor, worstIdxP *int, worstDistP *int64) {
	worstIdx := *worstIdxP
	worstDist := *worstDistP

	q0 := int64(query[0])
	q1 := int64(query[1])
	q2 := int64(query[2])
	q3 := int64(query[3])
	q4 := int64(query[4])
	q5 := int64(query[5])
	q6 := int64(query[6])
	q7 := int64(query[7])
	q8 := int64(query[8])
	q9 := int64(query[9])
	q10 := int64(query[10])
	q11 := int64(query[11])
	q12 := int64(query[12])
	q13 := int64(query[13])

	for i := start; i < end; i++ {
		v := (*[dataset.VectorDim]uint16)(vectors[i*dataset.VectorDim:])
		d0 := int64(v[0]) - q0
		d1 := int64(v[1]) - q1
		d2 := int64(v[2]) - q2
		d3 := int64(v[3]) - q3
		d4 := int64(v[4]) - q4
		d5 := int64(v[5]) - q5
		d6 := int64(v[6]) - q6
		d7 := int64(v[7]) - q7
		d8 := int64(v[8]) - q8
		d9 := int64(v[9]) - q9
		d10 := int64(v[10]) - q10
		d11 := int64(v[11]) - q11
		d12 := int64(v[12]) - q12
		d13 := int64(v[13]) - q13
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

func dist14u16(a, b *[dataset.VectorDim]uint16) int64 {
	d0 := int64(a[0]) - int64(b[0])
	d1 := int64(a[1]) - int64(b[1])
	d2 := int64(a[2]) - int64(b[2])
	d3 := int64(a[3]) - int64(b[3])
	d4 := int64(a[4]) - int64(b[4])
	d5 := int64(a[5]) - int64(b[5])
	d6 := int64(a[6]) - int64(b[6])
	d7 := int64(a[7]) - int64(b[7])
	d8 := int64(a[8]) - int64(b[8])
	d9 := int64(a[9]) - int64(b[9])
	d10 := int64(a[10]) - int64(b[10])
	d11 := int64(a[11]) - int64(b[11])
	d12 := int64(a[12]) - int64(b[12])
	d13 := int64(a[13]) - int64(b[13])
	return d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6 +
		d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13
}
