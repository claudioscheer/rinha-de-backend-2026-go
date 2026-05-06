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
	"unsafe"
)

const VectorDim = 14
const (
	Precision8  = 8
	Precision16 = 16
)

// binaryMagic identifies the pre-compiled references.bin format produced by
// cmd/compile-dataset. Bumping this invalidates older blobs.
const binaryMagic uint32 = 0x52313237 // "R127"

// binaryHeaderSize is the on-disk header preceding the index + vector + fraud
// blocks: magic, count, num_clusters, precision (each 4 bytes). Older blobs
// wrote zero in the precision slot, which is treated as Precision8.
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
// Vectors are quantized to either uint8 or uint16 to cut memory versus float32.
// Frauds is a packed bitmap (1 bit per record). The bulk of the data is kept
// in Vectors/Vectors16, which can be mmap-backed when loaded from a
// pre-compiled blob — that lets two API replicas share the same physical pages
// via the kernel's page cache.
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
	// are stored in cluster-id order. Used when Precision == Precision8.
	Vectors []uint8
	// Vectors16 is the uint16 variant used by high-recall compiled blobs.
	Vectors16 []uint16
	// Frauds is a packed bitmap. Bit i is 1 when record i is fraud. The
	// ordering matches Vectors/Vectors16.
	Frauds []uint8

	// Centroids holds quantized cluster centroids:
	// Centroids[c*VectorDim : (c+1)*VectorDim]. Nil when NumClusters == 0 or
	// Precision == Precision16.
	Centroids []uint8
	// Centroids16 is the uint16 variant used when Precision == Precision16.
	Centroids16 []uint16
	// ClusterOffsets has length NumClusters+1. Records of cluster c live
	// at indices [ClusterOffsets[c], ClusterOffsets[c+1]). Heap-allocated
	// so the hot path doesn't decode uint32s out of the mmap on every
	// query.
	ClusterOffsets []uint32
	NumClusters    int
	Precision      int

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
	if d.Vectors == nil && d.Vectors16 != nil {
		out := make([]uint8, VectorDim)
		row := d.Vectors16[i*VectorDim : (i+1)*VectorDim]
		for j, v := range row {
			out[j] = uint8((uint32(v) + 128) / 257)
		}
		return out
	}
	return d.Vectors[i*VectorDim : (i+1)*VectorDim]
}

// NewForTest constructs a heap-only Dataset around caller-supplied buffers.
// Intended only for unit tests that need a controlled fixture. The returned
// dataset has no IVF index (NumClusters == 0), so search falls back to brute
// force.
func NewForTest(vectors, frauds []uint8, count int) *Dataset {
	return &Dataset{
		Vectors:   vectors,
		Frauds:    frauds,
		Precision: Precision8,
		count:     count,
	}
}

// NewForTest16 constructs a heap-only Precision16 Dataset for tests.
func NewForTest16(vectors []uint16, frauds []uint8, count int) *Dataset {
	return &Dataset{
		Vectors16: vectors,
		Frauds:    frauds,
		Precision: Precision16,
		count:     count,
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
		Precision:      Precision8,
		count:          count,
	}
}

// NewForTestIVF16 constructs a heap-only Precision16 Dataset with an IVF index.
func NewForTestIVF16(vectors []uint16, frauds []uint8, centroids []uint16, offsets []uint32, count, numClusters int) *Dataset {
	return &Dataset{
		Vectors16:      vectors,
		Frauds:         frauds,
		Centroids16:    centroids,
		ClusterOffsets: offsets,
		NumClusters:    numClusters,
		Precision:      Precision16,
		count:          count,
	}
}

// Close releases any mmap-backed memory. Safe to call on heap-only datasets.
func (d *Dataset) Close() error {
	if d.mmap != nil {
		err := syscall.Munmap(d.mmap)
		d.mmap = nil
		d.Vectors = nil
		d.Vectors16 = nil
		d.Frauds = nil
		d.Centroids = nil
		d.Centroids16 = nil
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

// Quantize16 maps a float in [-1, 1] to uint16 in [0, 65535].
func Quantize16(v float32) uint16 {
	if v < -1 {
		v = -1
	} else if v > 1 {
		v = 1
	}
	return uint16((v+1)*32767.5 + 0.5)
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
		Norm:      norm,
		MccRisk:   mccRisk,
		Vectors:   vectors,
		Frauds:    frauds,
		Precision: Precision8,
		count:     count,
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
//	vectors[count*VectorDim*bytes_per_component]          quantized vectors (cluster-sorted when nc>0)
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
	precision := int(binary.LittleEndian.Uint32(buf[12:16]))
	if precision == 0 {
		precision = Precision8
	}
	if precision != Precision8 && precision != Precision16 {
		_ = syscall.Munmap(buf)
		return nil, fmt.Errorf("unsupported vector precision %d", precision)
	}

	pos := binaryHeaderSize
	var centroids []uint8
	var centroids16 []uint16
	var offsets []uint32
	if nc > 0 {
		centBytes := nc * VectorDim * bytesPerComponent(precision)
		if pos+centBytes > size {
			_ = syscall.Munmap(buf)
			return nil, fmt.Errorf("truncated centroids: %d bytes available, %d needed", size-pos, centBytes)
		}
		if precision == Precision16 {
			centroids16 = uint16Slice(buf[pos : pos+centBytes])
		} else {
			centroids = buf[pos : pos+centBytes]
		}
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

	vecBytes := count * VectorDim * bytesPerComponent(precision)
	if pos+vecBytes > size {
		_ = syscall.Munmap(buf)
		return nil, fmt.Errorf("truncated vectors: %d bytes available, %d needed", size-pos, vecBytes)
	}
	var vectors []uint8
	var vectors16 []uint16
	if precision == Precision16 {
		vectors16 = uint16Slice(buf[pos : pos+vecBytes])
	} else {
		vectors = buf[pos : pos+vecBytes]
	}
	pos += vecBytes

	fb := fraudBytes(count)
	if pos+fb > size {
		_ = syscall.Munmap(buf)
		return nil, fmt.Errorf("truncated frauds: %d bytes available, %d needed", size-pos, fb)
	}
	frauds := buf[pos : pos+fb]

	return &Dataset{
		Vectors:        vectors,
		Vectors16:      vectors16,
		Frauds:         frauds,
		Centroids:      centroids,
		Centroids16:    centroids16,
		ClusterOffsets: offsets,
		NumClusters:    nc,
		Precision:      precision,
		count:          count,
		mmap:           buf,
	}, nil
}

func bytesPerComponent(precision int) int {
	if precision == Precision16 {
		return 2
	}
	return 1
}

func uint16Slice(buf []byte) []uint16 {
	if len(buf) == 0 {
		return nil
	}
	return unsafe.Slice((*uint16)(unsafe.Pointer(&buf[0])), len(buf)/2)
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

func decodeJSONBoth(r io.Reader) ([]uint8, []uint16, []uint8, int, error) {
	dec := json.NewDecoder(bufio.NewReaderSize(r, 1<<20))

	tok, err := dec.Token()
	if err != nil {
		return nil, nil, nil, 0, fmt.Errorf("read array start: %w", err)
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '[' {
		return nil, nil, nil, 0, fmt.Errorf("expected array start, got %v", tok)
	}

	const estimated = 3_000_000
	vectors8 := make([]uint8, 0, estimated*VectorDim)
	vectors16 := make([]uint16, 0, estimated*VectorDim)
	frauds := make([]uint8, 0, fraudBytes(estimated))
	count := 0

	var rec referenceRecord
	for dec.More() {
		rec.Vector = rec.Vector[:0]
		rec.Label = ""
		if err := dec.Decode(&rec); err != nil {
			return nil, nil, nil, 0, fmt.Errorf("decode record: %w", err)
		}
		if len(rec.Vector) != VectorDim {
			return nil, nil, nil, 0, fmt.Errorf("unexpected vector dim %d", len(rec.Vector))
		}
		for _, v := range rec.Vector {
			vectors8 = append(vectors8, Quantize(v))
			vectors16 = append(vectors16, Quantize16(v))
		}
		if count%8 == 0 {
			frauds = append(frauds, 0)
		}
		if rec.Label == "fraud" {
			frauds[count>>3] |= 1 << uint(count&7)
		}
		count++
	}

	return vectors8, vectors16, frauds, count, nil
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
	binary.LittleEndian.PutUint32(hdr[12:16], Precision8)
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

func writeBlob16(w io.Writer, vectors, centroids []uint16, frauds []uint8, offsets []uint32, count, numClusters int) error {
	if len(vectors) != count*VectorDim {
		return fmt.Errorf("vectors16 length %d, want %d", len(vectors), count*VectorDim)
	}
	if len(frauds) != fraudBytes(count) {
		return fmt.Errorf("frauds length %d, want %d", len(frauds), fraudBytes(count))
	}
	if numClusters > 0 {
		if len(centroids) != numClusters*VectorDim {
			return fmt.Errorf("centroids16 length %d, want %d", len(centroids), numClusters*VectorDim)
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
	binary.LittleEndian.PutUint32(hdr[12:16], Precision16)
	if _, err := bw.Write(hdr[:]); err != nil {
		return err
	}
	if numClusters > 0 {
		if err := writeUint16s(bw, centroids); err != nil {
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
	if err := writeUint16s(bw, vectors); err != nil {
		return err
	}
	if _, err := bw.Write(frauds); err != nil {
		return err
	}
	return bw.Flush()
}

func writeUint16s(w io.Writer, values []uint16) error {
	var buf [8192]byte
	for len(values) > 0 {
		n := len(values)
		if n > len(buf)/2 {
			n = len(buf) / 2
		}
		for i := 0; i < n; i++ {
			binary.LittleEndian.PutUint16(buf[i*2:], values[i])
		}
		if _, err := w.Write(buf[:n*2]); err != nil {
			return err
		}
		values = values[n:]
	}
	return nil
}

type CompileOptions struct {
	Precision int
	Clusters  int
}

// CompileFromJSONGz reads a gzipped references file, builds an IVF index
// (when the dataset is large enough), reorders vectors by cluster id, and
// writes the compact binary blob the API mmaps at startup. Exported for use
// by cmd/compile-dataset.
func CompileFromJSONGz(in, out string) error {
	return CompileFromJSONGzWithOptions(in, out, CompileOptions{Precision: Precision8})
}

func CompileFromJSONGzWithOptions(in, out string, opts CompileOptions) error {
	if opts.Precision == 0 {
		opts.Precision = Precision8
	}
	if opts.Precision != Precision8 && opts.Precision != Precision16 {
		return fmt.Errorf("unsupported precision %d", opts.Precision)
	}

	r, err := openGzipped(in)
	if err != nil {
		return err
	}
	defer r.Close()

	var vectors []uint8
	var vectors16 []uint16
	var frauds []uint8
	var count int
	if opts.Precision == Precision16 {
		vectors, vectors16, frauds, count, err = decodeJSONBoth(r)
		if err != nil {
			return err
		}
	} else {
		vectors, frauds, count, err = decodeJSON(r)
		if err != nil {
			return err
		}
	}

	numClusters := chooseNumClusters(count)
	if opts.Clusters > 0 {
		numClusters = opts.Clusters
	}
	if numClusters > count {
		numClusters = count
	}
	var centroids []uint8
	var centroids16 []uint16
	var offsets []uint32
	if numClusters > 0 {
		var assignments []uint16
		centroids, assignments = kmeans(vectors, count, numClusters)
		if opts.Precision == Precision16 {
			centroids16 = buildCentroids16(vectors16, count, numClusters, assignments)
			vectors16, frauds, offsets = reorderByCluster16(vectors16, frauds, count, numClusters, assignments)
		} else {
			vectors, frauds, offsets = reorderByCluster(vectors, frauds, count, numClusters, assignments)
		}
	}

	tmp := out + ".tmp"
	o, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if opts.Precision == Precision16 {
		err = writeBlob16(o, vectors16, centroids16, frauds, offsets, count, numClusters)
	} else {
		err = writeBlob(o, vectors, frauds, centroids, offsets, count, numClusters)
	}
	if err != nil {
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
