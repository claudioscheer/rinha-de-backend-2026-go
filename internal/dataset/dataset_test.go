package dataset

import (
	"bytes"
	"compress/gzip"
	"os"
	"path/filepath"
	"testing"
)

func writeGzipReferences(t *testing.T, path, content string) {
	t.Helper()
	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	if _, err := gz.Write([]byte(content)); err != nil {
		t.Fatalf("gzip write: %v", err)
	}
	if err := gz.Close(); err != nil {
		t.Fatalf("gzip close: %v", err)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("write file: %v", err)
	}
}

func writeFixtureDir(t *testing.T, refContent string) string {
	t.Helper()
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "normalization.json"), []byte(`{
      "max_amount": 10000,
      "max_installments": 12,
      "amount_vs_avg_ratio": 10,
      "max_minutes": 1440,
      "max_km": 1000,
      "max_tx_count_24h": 20,
      "max_merchant_avg_amount": 10000
    }`), 0o644); err != nil {
		t.Fatalf("write normalization: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "mcc_risk.json"), []byte(`{
      "5411": 0.15,
      "7802": 0.75
    }`), 0o644); err != nil {
		t.Fatalf("write mcc_risk: %v", err)
	}
	writeGzipReferences(t, filepath.Join(dir, "references.json.gz"), refContent)
	return dir
}

func TestLoad_HappyPath(t *testing.T) {
	dir := writeFixtureDir(t, `[
      {"vector":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"legit"},
      {"vector":[1,1,1,1,1,1,1,1,1,1,1,1,1,1],"label":"fraud"},
      {"vector":[0.5,0.5,0.5,0.5,0.5,-1,-1,0.5,0.5,0,1,0,0.5,0.5],"label":"legit"}
    ]`)

	ds, err := Load(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { _ = ds.Close() })
	if got := ds.Size(); got != 3 {
		t.Fatalf("Size() = %d, want 3", got)
	}
	if ds.Norm.MaxAmount != 10000 {
		t.Errorf("MaxAmount = %v, want 10000", ds.Norm.MaxAmount)
	}
	if ds.MccRisk["5411"] != 0.15 {
		t.Errorf("mcc 5411 = %v, want 0.15", ds.MccRisk["5411"])
	}
	if !ds.IsFraud(1) || ds.IsFraud(0) || ds.IsFraud(2) {
		t.Errorf("IsFraud values wrong: 0=%v 1=%v 2=%v", ds.IsFraud(0), ds.IsFraud(1), ds.IsFraud(2))
	}

	// Vector(i) must return a contiguous, correctly-shaped slice.
	v0 := ds.Vector(0)
	v1 := ds.Vector(1)
	v2 := ds.Vector(2)
	if len(v0) != VectorDim || len(v1) != VectorDim || len(v2) != VectorDim {
		t.Fatalf("vector lengths %d %d %d, want %d", len(v0), len(v1), len(v2), VectorDim)
	}
	// 1.0 quantizes to 255, -1.0 quantizes to 0.
	for i := 0; i < VectorDim; i++ {
		if v1[i] != 255 {
			t.Errorf("vector 1[%d] = %v, want 255 (quantized 1.0)", i, v1[i])
		}
	}
	if v2[5] != 0 || v2[6] != 0 {
		t.Errorf("expected quantized -1 (=0) at indices 5 and 6, got %v %v", v2[5], v2[6])
	}
}

func TestLoad_MissingNormalization(t *testing.T) {
	dir := t.TempDir()
	if _, err := Load(dir); err == nil {
		t.Fatal("expected error when normalization.json is missing")
	}
}

func TestLoad_BadVectorDim(t *testing.T) {
	dir := writeFixtureDir(t, `[{"vector":[1,2,3],"label":"legit"}]`)
	if _, err := Load(dir); err == nil {
		t.Fatal("expected error for vector with wrong dimension")
	}
}

func TestLoad_BadJSON(t *testing.T) {
	dir := writeFixtureDir(t, `not json at all`)
	if _, err := Load(dir); err == nil {
		t.Fatal("expected error for malformed references")
	}
}

func TestLoad_EmptyArray(t *testing.T) {
	dir := writeFixtureDir(t, `[]`)
	ds, err := Load(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { _ = ds.Close() })
	if ds.Size() != 0 {
		t.Errorf("Size() = %d, want 0", ds.Size())
	}
}

// Round-trip: compile a JSON dataset to the binary form, then load it via the
// mmap path. Verifies the on-disk format is consumed correctly.
func TestLoad_BinaryRoundTrip(t *testing.T) {
	dir := writeFixtureDir(t, `[
      {"vector":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"legit"},
      {"vector":[1,1,1,1,1,1,1,1,1,1,1,1,1,1],"label":"fraud"}
    ]`)
	if err := CompileFromJSONGz(
		filepath.Join(dir, "references.json.gz"),
		filepath.Join(dir, "references.bin"),
	); err != nil {
		t.Fatalf("compile: %v", err)
	}
	// Remove the gz so the loader is forced down the mmap path.
	if err := os.Remove(filepath.Join(dir, "references.json.gz")); err != nil {
		t.Fatalf("remove gz: %v", err)
	}

	ds, err := Load(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	t.Cleanup(func() { _ = ds.Close() })
	if ds.Size() != 2 {
		t.Fatalf("Size() = %d, want 2", ds.Size())
	}
	if ds.IsFraud(0) || !ds.IsFraud(1) {
		t.Errorf("IsFraud wrong after round-trip: 0=%v 1=%v", ds.IsFraud(0), ds.IsFraud(1))
	}
	for i := 0; i < VectorDim; i++ {
		if ds.Vector(0)[i] != Quantize(0) {
			t.Errorf("vec0[%d] = %v, want %v", i, ds.Vector(0)[i], Quantize(0))
		}
		if ds.Vector(1)[i] != Quantize(1) {
			t.Errorf("vec1[%d] = %v, want %v", i, ds.Vector(1)[i], Quantize(1))
		}
	}
}

func TestQuantize(t *testing.T) {
	cases := []struct {
		in   float32
		want uint8
	}{
		{-2, 0}, // clamped
		{-1, 0},
		{0, 128},
		{1, 255},
		{2, 255}, // clamped
	}
	for _, c := range cases {
		if got := Quantize(c.in); got != c.want {
			t.Errorf("Quantize(%v) = %d, want %d", c.in, got, c.want)
		}
	}
}

// The real resources directory must always parse — guards against typos in
// normalization.json or mcc_risk.json that would otherwise only surface in
// production startup.
func TestLoad_RealNormalizationAndMcc(t *testing.T) {
	if _, err := os.Stat("../../resources/normalization.json"); os.IsNotExist(err) {
		t.Skip("resources not present in this checkout")
	}
	norm, err := loadNormalization("../../resources/normalization.json")
	if err != nil {
		t.Fatalf("load normalization: %v", err)
	}
	if norm.MaxAmount <= 0 || norm.MaxKm <= 0 {
		t.Errorf("normalization looks bogus: %+v", norm)
	}
	mcc, err := loadMccRisk("../../resources/mcc_risk.json")
	if err != nil {
		t.Fatalf("load mcc_risk: %v", err)
	}
	if v, ok := mcc["5411"]; !ok || v == 0 {
		t.Errorf("mcc_risk[5411] missing or zero: %v", v)
	}
}
