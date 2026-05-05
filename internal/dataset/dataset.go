package dataset

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"syscall"
)

const VectorDim = 14

// binaryMagic identifies the pre-compiled references.bin format produced by
// cmd/compile-dataset. Bumping this invalidates older blobs.
const binaryMagic uint32 = 0x52313237 // "R127"

// binaryHeaderSize is the on-disk header preceding the index + vector + fraud
// blocks: magic, count, num_clusters, reserved (each 4 bytes).
const binaryHeaderSize = 16

type Normalization struct {
	MaxAmount            float64 `json:"max_amount"`
	MaxInstallments      float64 `json:"max_installments"`
	AmountVsAvgRatio     float64 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float64 `json:"max_minutes"`
	MaxKm                float64 `json:"max_km"`
	MaxTxCount24h        float64 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float64 `json:"max_merchant_avg_amount"`
}

// Dataset is the in-memory reference base used by the K-NN search.
//
// Vectors are quantized to uint8 (linear map [-1, 1] → [0, 255]) to cut memory
// 4× versus float32. Frauds is a packed bitmap (1 bit per record). The bulk
// of the data is kept in Vectors, which can be mmap-backed when loaded from a
// pre-compiled blob — that lets two API replicas share the same physical
// pages via the kernel's page cache.
//
// When NumClusters > 0, the dataset carries an inverted-file (IVF) index:
// vectors are sorted by cluster id, ClusterOffsets[c..c+1] gives the slice of
// records assigned to cluster c, and Centroids holds one quantized centroid
// per cluster. Search ranks centroids and only scans the closest few buckets.
type Dataset struct {
	Norm    Normalization
	MccRisk map[string]float32

	// Vectors stores quantized vectors flattened: index i lives at
	// Vectors[i*VectorDim : (i+1)*VectorDim]. With an IVF index, vectors
	// are stored in cluster-id order.
	Vectors []uint8
	// Frauds is a packed bitmap. Bit i is 1 when record i is fraud. The
	// ordering matches Vectors.
	Frauds []uint8

	// Centroids holds quantized cluster centroids:
	// Centroids[c*VectorDim : (c+1)*VectorDim]. Nil when NumClusters == 0.
	Centroids []uint8
	// ClusterOffsets has length NumClusters+1. Records of cluster c live
	// at indices [ClusterOffsets[c], ClusterOffsets[c+1]). Heap-allocated
	// so the hot path doesn't decode uint32s out of the mmap on every
	// query.
	ClusterOffsets []uint32
	NumClusters    int

	count int

	// mmap is non-nil when Vectors/Frauds/Centroids slice into an mmap'd
	// region.
	mmap []byte
}

func (d *Dataset) Size() int { return d.count }

// IsFraud reports whether record i is labeled as fraud.
func (d *Dataset) IsFraud(i int) bool {
	return d.Frauds[i>>3]&(1<<uint(i&7)) != 0
}

// Vector returns the i-th reference vector as a slice into the underlying
// storage. The result aliases the dataset and must not be modified.
func (d *Dataset) Vector(i int) []uint8 {
	return d.Vectors[i*VectorDim : (i+1)*VectorDim]
}

// NewForTest constructs a heap-only Dataset around caller-supplied buffers.
// Intended only for unit tests that need a controlled fixture. The returned
// dataset has no IVF index (NumClusters == 0), so search falls back to brute
// force.
func NewForTest(vectors, frauds []uint8, count int) *Dataset {
	return &Dataset{
		Vectors: vectors,
		Frauds:  frauds,
		count:   count,
	}
}

// NewForTestIVF constructs a heap-only Dataset with an IVF index. The caller
// must guarantee vectors and frauds are already in cluster-id order, with
// offsets[c..c+1] delimiting cluster c and offsets[numClusters] == count.
func NewForTestIVF(vectors, frauds, centroids []uint8, offsets []uint32, count, numClusters int) *Dataset {
	return &Dataset{
		Vectors:        vectors,
		Frauds:         frauds,
		Centroids:      centroids,
		ClusterOffsets: offsets,
		NumClusters:    numClusters,
		count:          count,
	}
}

// Close releases any mmap-backed memory. Safe to call on heap-only datasets.
func (d *Dataset) Close() error {
	if d.mmap != nil {
		err := syscall.Munmap(d.mmap)
		d.mmap = nil
		d.Vectors = nil
		d.Frauds = nil
		return err
	}
	return nil
}

// Quantize maps a float in [-1, 1] to uint8 in [0, 255]. Out-of-range values
// are clamped. -1 → 0, 0 → 128, 1 → 255. The +0.5 produces round-half-up
// because the uint8 cast truncates toward zero on a non-negative input.
func Quantize(v float32) uint8 {
	if v < -1 {
		v = -1
	} else if v > 1 {
		v = 1
	}
	return uint8((v+1)*127.5 + 0.5)
}

func Load(dir string) (*Dataset, error) {
	norm, err := loadNormalization(filepath.Join(dir, "normalization.json"))
	if err != nil {
		return nil, err
	}

	mccRisk, err := loadMccRisk(filepath.Join(dir, "mcc_risk.json"))
	if err != nil {
		return nil, err
	}

	binPath := filepath.Join(dir, "references.bin")
	if ds, err := loadBinary(binPath); err == nil {
		ds.Norm = norm
		ds.MccRisk = mccRisk
		return ds, nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("load binary references: %w", err)
	}

	// Fallback: decode JSON. Slower and uses more peak memory, but keeps
	// `go test` green when compile-dataset hasn't been run.
	vectors, frauds, count, err := loadJSON(filepath.Join(dir, "references.json.gz"))
	if err != nil {
		return nil, err
	}
	return &Dataset{
		Norm:    norm,
		MccRisk: mccRisk,
		Vectors: vectors,
		Frauds:  frauds,
		count:   count,
	}, nil
}

func loadNormalization(path string) (Normalization, error) {
	var n Normalization
	data, err := os.ReadFile(path)
	if err != nil {
		return n, fmt.Errorf("read normalization: %w", err)
	}
	if err := json.Unmarshal(data, &n); err != nil {
		return n, fmt.Errorf("parse normalization: %w", err)
	}
	return n, nil
}

func loadMccRisk(path string) (map[string]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read mcc_risk: %w", err)
	}
	raw := map[string]float64{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parse mcc_risk: %w", err)
	}
	out := make(map[string]float32, len(raw))
	for k, v := range raw {
		out[k] = float32(v)
	}
	return out, nil
}

// fraudBytes returns the bitmap byte count that holds `count` fraud bits.
func fraudBytes(count int) int { return (count + 7) / 8 }

// openGzipped opens a gzipped file and returns a ReadCloser whose Close
// closes both the gzip reader and the underlying file.
func openGzipped(path string) (io.ReadCloser, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	gz, err := gzip.NewReader(f)
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	return gzipReadCloser{gz: gz, f: f}, nil
}

type gzipReadCloser struct {
	gz *gzip.Reader
	f  *os.File
}

func (g gzipReadCloser) Read(p []byte) (int, error) { return g.gz.Read(p) }
func (g gzipReadCloser) Close() error {
	gzErr := g.gz.Close()
	fErr := g.f.Close()
	if gzErr != nil {
		return gzErr
	}
	return fErr
}

// loadBinary mmaps the pre-compiled references.bin and returns a Dataset
// whose Vectors/Frauds/Centroids slices alias the mmap region. The caller
// must Close the dataset at shutdown to release the mapping.
//
// Layout:
//
//	header[16]                                            magic | count | nc | reserved
//	if nc > 0:
//	  centroids[nc*VectorDim]                             quantized centroids
//	  cluster_offsets[(nc+1)*4]                           uint32 LE prefix sums
//	vectors[count*VectorDim]                              uint8 vectors (cluster-sorted when nc>0)
//	frauds[(count+7)/8]                                   packed fraud bitmap
func loadBinary(path string) (*Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat: %w", err)
	}
	size := int(st.Size())
	if size < binaryHeaderSize {
		return nil, fmt.Errorf("references.bin too small: %d bytes", size)
	}

	buf, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}

	magic := binary.LittleEndian.Uint32(buf[0:4])
	if magic != binaryMagic {
		_ = syscall.Munmap(buf)
		return nil, fmt.Errorf("bad magic 0x%08x, want 0x%08x", magic, binaryMagic)
	}
	count := int(binary.LittleEndian.Uint32(buf[4:8]))
	nc := int(binary.LittleEndian.Uint32(buf[8:12]))

	pos := binaryHeaderSize
	var centroids []uint8
	var offsets []uint32
	if nc > 0 {
		centBytes := nc * VectorDim
		if pos+centBytes > size {
			_ = syscall.Munmap(buf)
			return nil, fmt.Errorf("truncated centroids: %d bytes available, %d needed", size-pos, centBytes)
		}
		centroids = buf[pos : pos+centBytes]
		pos += centBytes

		offsetsBytes := (nc + 1) * 4
		if pos+offsetsBytes > size {
			_ = syscall.Munmap(buf)
			return nil, fmt.Errorf("truncated cluster offsets")
		}
		offsets = make([]uint32, nc+1)
		for i := 0; i <= nc; i++ {
			offsets[i] = binary.LittleEndian.Uint32(buf[pos+i*4:])
		}
		if int(offsets[nc]) != count {
			_ = syscall.Munmap(buf)
			return nil, fmt.Errorf("cluster offsets[%d]=%d does not match count=%d", nc, offsets[nc], count)
		}
		pos += offsetsBytes
	}

	vecBytes := count * VectorDim
	if pos+vecBytes > size {
		_ = syscall.Munmap(buf)
		return nil, fmt.Errorf("truncated vectors: %d bytes available, %d needed", size-pos, vecBytes)
	}
	vectors := buf[pos : pos+vecBytes]
	pos += vecBytes

	fb := fraudBytes(count)
	if pos+fb > size {
		_ = syscall.Munmap(buf)
		return nil, fmt.Errorf("truncated frauds: %d bytes available, %d needed", size-pos, fb)
	}
	frauds := buf[pos : pos+fb]

	return &Dataset{
		Vectors:        vectors,
		Frauds:         frauds,
		Centroids:      centroids,
		ClusterOffsets: offsets,
		NumClusters:    nc,
		count:          count,
		mmap:           buf,
	}, nil
}

// loadJSON streams a gzipped JSON array of `{"vector": [...], "label": "..."}`
// records and quantizes them in place. Used as a fallback when no binary blob
// is available (e.g. from tests).
func loadJSON(path string) ([]uint8, []uint8, int, error) {
	r, err := openGzipped(path)
	if err != nil {
		return nil, nil, 0, fmt.Errorf("open references: %w", err)
	}
	defer r.Close()
	return decodeJSON(r)
}

// decodeJSON consumes a JSON array of reference records from r.
func decodeJSON(r io.Reader) ([]uint8, []uint8, int, error) {
	dec := json.NewDecoder(bufio.NewReaderSize(r, 1<<20))

	tok, err := dec.Token()
	if err != nil {
		return nil, nil, 0, fmt.Errorf("read array start: %w", err)
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '[' {
		return nil, nil, 0, fmt.Errorf("expected array start, got %v", tok)
	}

	const estimated = 3_000_000
	vectors := make([]uint8, 0, estimated*VectorDim)
	frauds := make([]uint8, 0, fraudBytes(estimated))
	count := 0

	var rec referenceRecord
	for dec.More() {
		rec.Vector = rec.Vector[:0]
		rec.Label = ""
		if err := dec.Decode(&rec); err != nil {
			return nil, nil, 0, fmt.Errorf("decode record: %w", err)
		}
		if len(rec.Vector) != VectorDim {
			return nil, nil, 0, fmt.Errorf("unexpected vector dim %d", len(rec.Vector))
		}
		for _, v := range rec.Vector {
			vectors = append(vectors, Quantize(v))
		}
		if count%8 == 0 {
			frauds = append(frauds, 0)
		}
		if rec.Label == "fraud" {
			frauds[count>>3] |= 1 << uint(count&7)
		}
		count++
	}

	return vectors, frauds, count, nil
}

type referenceRecord struct {
	Vector []float32 `json:"vector"`
	Label  string    `json:"label"`
}

// writeBlob serializes a (possibly IVF-indexed) reference blob into the
// on-disk format consumed by loadBinary. centroids/offsets must be set
// consistently with numClusters: nil and zero for a flat layout, populated
// otherwise.
func writeBlob(w io.Writer, vectors, frauds, centroids []uint8, offsets []uint32, count, numClusters int) error {
	if len(vectors) != count*VectorDim {
		return fmt.Errorf("vectors length %d, want %d", len(vectors), count*VectorDim)
	}
	if len(frauds) != fraudBytes(count) {
		return fmt.Errorf("frauds length %d, want %d", len(frauds), fraudBytes(count))
	}
	if numClusters > 0 {
		if len(centroids) != numClusters*VectorDim {
			return fmt.Errorf("centroids length %d, want %d", len(centroids), numClusters*VectorDim)
		}
		if len(offsets) != numClusters+1 {
			return fmt.Errorf("offsets length %d, want %d", len(offsets), numClusters+1)
		}
		if int(offsets[numClusters]) != count {
			return fmt.Errorf("offsets[%d]=%d, want count=%d", numClusters, offsets[numClusters], count)
		}
	}

	bw := bufio.NewWriterSize(w, 1<<20)
	var hdr [binaryHeaderSize]byte
	binary.LittleEndian.PutUint32(hdr[0:4], binaryMagic)
	binary.LittleEndian.PutUint32(hdr[4:8], uint32(count))
	binary.LittleEndian.PutUint32(hdr[8:12], uint32(numClusters))
	if _, err := bw.Write(hdr[:]); err != nil {
		return err
	}
	if numClusters > 0 {
		if _, err := bw.Write(centroids); err != nil {
			return err
		}
		var off [4]byte
		for i := 0; i <= numClusters; i++ {
			binary.LittleEndian.PutUint32(off[:], offsets[i])
			if _, err := bw.Write(off[:]); err != nil {
				return err
			}
		}
	}
	if _, err := bw.Write(vectors); err != nil {
		return err
	}
	if _, err := bw.Write(frauds); err != nil {
		return err
	}
	return bw.Flush()
}

// CompileFromJSONGz reads a gzipped references file, builds an IVF index
// (when the dataset is large enough), reorders vectors by cluster id, and
// writes the compact binary blob the API mmaps at startup. Exported for use
// by cmd/compile-dataset.
func CompileFromJSONGz(in, out string) error {
	r, err := openGzipped(in)
	if err != nil {
		return err
	}
	defer r.Close()

	vectors, frauds, count, err := decodeJSON(r)
	if err != nil {
		return err
	}

	numClusters := chooseNumClusters(count)
	var centroids []uint8
	var offsets []uint32
	if numClusters > 0 {
		var assignments []uint16
		centroids, assignments = kmeans(vectors, count, numClusters)
		vectors, frauds, offsets = reorderByCluster(vectors, frauds, count, numClusters, assignments)
	}

	tmp := out + ".tmp"
	o, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if err := writeBlob(o, vectors, frauds, centroids, offsets, count, numClusters); err != nil {
		_ = o.Close()
		_ = os.Remove(tmp)
		return err
	}
	if err := o.Close(); err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, out)
}
