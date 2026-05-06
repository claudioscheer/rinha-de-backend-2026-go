package dataset

import (
	"math/rand/v2"
	"runtime"
	"sync"
)

const (
	kmeansIters    = 15
	kmeansSeedHi   = 0x12345678
	kmeansSeedLo   = 0xdeadbeef
	kmeansMaxProcs = 8
)

// chooseNumClusters picks the IVF cluster count. Datasets below the threshold
// are stored flat (NumClusters=0) so tiny test fixtures don't pay the build
// cost and the search package falls back to brute force.
func chooseNumClusters(count int) int {
	switch {
	case count < 1024:
		return 0
	case count < 16384:
		return 64
	default:
		return 1024
	}
}

// kmeans runs Lloyd's algorithm on uint8 reference vectors and returns
// quantized centroids plus per-vector cluster assignments. Centroid means are
// accumulated in float64 for numerical stability and quantized back at the
// end. The assignment loop is parallelized across NumCPU goroutines.
func kmeans(vectors []uint8, count, numClusters int) ([]uint8, []uint16) {
	rng := rand.New(rand.NewPCG(kmeansSeedHi, kmeansSeedLo))

	centroids := make([]float32, numClusters*VectorDim)
	for c := 0; c < numClusters; c++ {
		idx := rng.IntN(count)
		for d := 0; d < VectorDim; d++ {
			centroids[c*VectorDim+d] = float32(vectors[idx*VectorDim+d])
		}
	}

	assignments := make([]uint16, count)

	workers := runtime.GOMAXPROCS(0)
	if workers > kmeansMaxProcs {
		workers = kmeansMaxProcs
	}
	if workers < 1 {
		workers = 1
	}

	type partial struct {
		sums   []float64
		counts []uint32
	}
	parts := make([]partial, workers)
	for i := range parts {
		parts[i].sums = make([]float64, numClusters*VectorDim)
		parts[i].counts = make([]uint32, numClusters)
	}

	for iter := 0; iter < kmeansIters; iter++ {
		for w := 0; w < workers; w++ {
			for j := range parts[w].sums {
				parts[w].sums[j] = 0
			}
			for j := range parts[w].counts {
				parts[w].counts[j] = 0
			}
		}

		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func(w int) {
				defer wg.Done()
				start := w * count / workers
				end := (w + 1) * count / workers
				ws := parts[w].sums
				wc := parts[w].counts
				for i := start; i < end; i++ {
					best := nearestCentroid(centroids, numClusters, vectors[i*VectorDim:i*VectorDim+VectorDim])
					assignments[i] = uint16(best)
					base := best * VectorDim
					for d := 0; d < VectorDim; d++ {
						ws[base+d] += float64(vectors[i*VectorDim+d])
					}
					wc[best]++
				}
			}(w)
		}
		wg.Wait()

		for c := 0; c < numClusters; c++ {
			total := uint32(0)
			for w := 0; w < workers; w++ {
				total += parts[w].counts[c]
			}
			if total == 0 {
				// Empty cluster: re-seed from a random vector so it
				// gets contested in the next iteration.
				idx := rng.IntN(count)
				for d := 0; d < VectorDim; d++ {
					centroids[c*VectorDim+d] = float32(vectors[idx*VectorDim+d])
				}
				continue
			}
			inv := 1.0 / float64(total)
			for d := 0; d < VectorDim; d++ {
				sum := float64(0)
				for w := 0; w < workers; w++ {
					sum += parts[w].sums[c*VectorDim+d]
				}
				centroids[c*VectorDim+d] = float32(sum * inv)
			}
		}
	}

	out := make([]uint8, numClusters*VectorDim)
	for i, v := range centroids {
		x := int(v + 0.5)
		if x < 0 {
			x = 0
		} else if x > 255 {
			x = 255
		}
		out[i] = uint8(x)
	}
	return out, assignments
}

// nearestCentroid returns the index of the centroid minimizing squared
// Euclidean distance to query (read as uint8).
func nearestCentroid(centroids []float32, numClusters int, query []byte) int {
	bestIdx := 0
	bestDist := float32(1e20)
	for c := 0; c < numClusters; c++ {
		base := c * VectorDim
		d := float32(0)
		for j := 0; j < VectorDim; j++ {
			diff := centroids[base+j] - float32(query[j])
			d += diff * diff
		}
		if d < bestDist {
			bestDist = d
			bestIdx = c
		}
	}
	return bestIdx
}

// reorderByCluster bucket-sorts vectors and the fraud bitmap by cluster id and
// returns the new layout plus the prefix-sum offsets used by the search.
func reorderByCluster(vectors, frauds []uint8, count, numClusters int, assignments []uint16) ([]uint8, []uint8, []uint32) {
	offsets := make([]uint32, numClusters+1)
	for i := 0; i < count; i++ {
		offsets[assignments[i]+1]++
	}
	for c := 0; c < numClusters; c++ {
		offsets[c+1] += offsets[c]
	}

	pos := make([]uint32, numClusters)
	copy(pos, offsets[:numClusters])

	newVectors := make([]uint8, count*VectorDim)
	newFrauds := make([]uint8, fraudBytes(count))

	for i := 0; i < count; i++ {
		c := assignments[i]
		p := int(pos[c])
		copy(newVectors[p*VectorDim:(p+1)*VectorDim], vectors[i*VectorDim:(i+1)*VectorDim])
		if frauds[i>>3]&(1<<uint(i&7)) != 0 {
			newFrauds[p>>3] |= 1 << uint(p&7)
		}
		pos[c]++
	}

	return newVectors, newFrauds, offsets
}

func buildCentroids16(vectors []uint16, count, numClusters int, assignments []uint16) []uint16 {
	sums := make([]uint64, numClusters*VectorDim)
	counts := make([]uint32, numClusters)
	for i := 0; i < count; i++ {
		c := int(assignments[i])
		base := c * VectorDim
		for d := 0; d < VectorDim; d++ {
			sums[base+d] += uint64(vectors[i*VectorDim+d])
		}
		counts[c]++
	}

	out := make([]uint16, numClusters*VectorDim)
	for c := 0; c < numClusters; c++ {
		if counts[c] == 0 {
			continue
		}
		half := uint64(counts[c] / 2)
		base := c * VectorDim
		for d := 0; d < VectorDim; d++ {
			out[base+d] = uint16((sums[base+d] + half) / uint64(counts[c]))
		}
	}
	return out
}

func reorderByCluster16(vectors []uint16, frauds []uint8, count, numClusters int, assignments []uint16) ([]uint16, []uint8, []uint32) {
	offsets := make([]uint32, numClusters+1)
	for i := 0; i < count; i++ {
		offsets[assignments[i]+1]++
	}
	for c := 0; c < numClusters; c++ {
		offsets[c+1] += offsets[c]
	}

	pos := make([]uint32, numClusters)
	copy(pos, offsets[:numClusters])

	newVectors := make([]uint16, count*VectorDim)
	newFrauds := make([]uint8, fraudBytes(count))

	for i := 0; i < count; i++ {
		c := assignments[i]
		p := int(pos[c])
		copy(newVectors[p*VectorDim:(p+1)*VectorDim], vectors[i*VectorDim:(i+1)*VectorDim])
		if frauds[i>>3]&(1<<uint(i&7)) != 0 {
			newFrauds[p>>3] |= 1 << uint(p&7)
		}
		pos[c]++
	}

	return newVectors, newFrauds, offsets
}
