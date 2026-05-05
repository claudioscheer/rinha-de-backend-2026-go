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
const binaryMagic uint32 = 0x52313236 // "R126"

// binaryHeaderSize is the on-disk header preceding the vector + fraud blocks.
const binaryHeaderSize = 8

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
type Dataset struct {
	Norm    Normalization
	MccRisk map[string]float32

	// Vectors stores quantized vectors flattened: index i lives at
	// Vectors[i*VectorDim : (i+1)*VectorDim].
	Vectors []uint8
	// Frauds is a packed bitmap. Bit i is 1 when record i is fraud.
	Frauds []uint8
	count  int

	// mmap is non-nil when Vectors/Frauds slice into an mmap'd region.
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
// Intended only for unit tests that need a controlled fixture.
func NewForTest(vectors, frauds []uint8, count int) *Dataset {
	return &Dataset{
		Vectors: vectors,
		Frauds:  frauds,
		count:   count,
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
	if vectors, frauds, count, mmapBuf, err := loadBinary(binPath); err == nil {
		return &Dataset{
			Norm:    norm,
			MccRisk: mccRisk,
			Vectors: vectors,
			Frauds:  frauds,
			count:   count,
			mmap:    mmapBuf,
		}, nil
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

// loadBinary mmaps the pre-compiled references.bin. The slices it returns
// alias the mmap region; the caller must keep mmapBuf alive for their
// lifetime and Munmap it at shutdown.
func loadBinary(path string) ([]uint8, []uint8, int, []byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, 0, nil, err
	}
	defer f.Close()

	st, err := f.Stat()
	if err != nil {
		return nil, nil, 0, nil, fmt.Errorf("stat: %w", err)
	}
	size := int(st.Size())
	if size < binaryHeaderSize {
		return nil, nil, 0, nil, fmt.Errorf("references.bin too small: %d bytes", size)
	}

	buf, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, nil, 0, nil, fmt.Errorf("mmap: %w", err)
	}

	magic := binary.LittleEndian.Uint32(buf[0:4])
	if magic != binaryMagic {
		_ = syscall.Munmap(buf)
		return nil, nil, 0, nil, fmt.Errorf("bad magic 0x%08x, want 0x%08x", magic, binaryMagic)
	}
	count := int(binary.LittleEndian.Uint32(buf[4:8]))

	vecBytes := count * VectorDim
	fb := fraudBytes(count)
	expected := binaryHeaderSize + vecBytes + fb
	if size < expected {
		_ = syscall.Munmap(buf)
		return nil, nil, 0, nil, fmt.Errorf("truncated references.bin: have %d bytes, need %d", size, expected)
	}

	vectors := buf[binaryHeaderSize : binaryHeaderSize+vecBytes]
	frauds := buf[binaryHeaderSize+vecBytes : binaryHeaderSize+vecBytes+fb]
	return vectors, frauds, count, buf, nil
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

// WriteBinary serializes vectors + frauds bitmap into the on-disk format
// consumed by loadBinary. Used by cmd/compile-dataset.
func WriteBinary(w io.Writer, vectors, frauds []uint8, count int) error {
	if len(vectors) != count*VectorDim {
		return fmt.Errorf("vectors length %d, want %d", len(vectors), count*VectorDim)
	}
	if len(frauds) != fraudBytes(count) {
		return fmt.Errorf("frauds length %d, want %d", len(frauds), fraudBytes(count))
	}

	bw := bufio.NewWriterSize(w, 1<<20)
	var hdr [binaryHeaderSize]byte
	binary.LittleEndian.PutUint32(hdr[0:4], binaryMagic)
	binary.LittleEndian.PutUint32(hdr[4:8], uint32(count))
	if _, err := bw.Write(hdr[:]); err != nil {
		return err
	}
	if _, err := bw.Write(vectors); err != nil {
		return err
	}
	if _, err := bw.Write(frauds); err != nil {
		return err
	}
	return bw.Flush()
}

// CompileFromJSONGz reads a gzipped references file and writes the compact
// binary form. Exported for use by cmd/compile-dataset.
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

	tmp := out + ".tmp"
	o, err := os.Create(tmp)
	if err != nil {
		return err
	}
	if err := WriteBinary(o, vectors, frauds, count); err != nil {
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
