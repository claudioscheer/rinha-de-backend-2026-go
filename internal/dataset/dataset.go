package dataset

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

const VectorDim = 14

type Normalization struct {
	MaxAmount             float64 `json:"max_amount"`
	MaxInstallments       float64 `json:"max_installments"`
	AmountVsAvgRatio      float64 `json:"amount_vs_avg_ratio"`
	MaxMinutes            float64 `json:"max_minutes"`
	MaxKm                 float64 `json:"max_km"`
	MaxTxCount24h         float64 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount  float64 `json:"max_merchant_avg_amount"`
}

type Dataset struct {
	Norm    Normalization
	MccRisk map[string]float32

	// Vectors stored as a flat slice for memory locality.
	// Index i lives at Vectors[i*VectorDim : (i+1)*VectorDim].
	Vectors []float32
	// Frauds[i] is true when the i-th reference vector is fraudulent.
	Frauds []bool
}

func (d *Dataset) Size() int {
	return len(d.Frauds)
}

func (d *Dataset) Vector(i int) []float32 {
	return d.Vectors[i*VectorDim : (i+1)*VectorDim]
}

type referenceRecord struct {
	Vector []float32 `json:"vector"`
	Label  string    `json:"label"`
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

	vectors, frauds, err := loadReferences(filepath.Join(dir, "references.json.gz"))
	if err != nil {
		return nil, err
	}

	return &Dataset{
		Norm:    norm,
		MccRisk: mccRisk,
		Vectors: vectors,
		Frauds:  frauds,
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

func loadReferences(path string) ([]float32, []bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("open references: %w", err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, nil, fmt.Errorf("gzip reader: %w", err)
	}
	defer gz.Close()

	dec := json.NewDecoder(gz)

	// Expect a top-level JSON array.
	tok, err := dec.Token()
	if err != nil {
		return nil, nil, fmt.Errorf("read array start: %w", err)
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '[' {
		return nil, nil, fmt.Errorf("expected array start, got %v", tok)
	}

	// Pre-size for ~3M records to avoid repeated growth.
	const estimated = 3_000_000
	vectors := make([]float32, 0, estimated*VectorDim)
	frauds := make([]bool, 0, estimated)

	for dec.More() {
		var rec referenceRecord
		if err := dec.Decode(&rec); err != nil {
			return nil, nil, fmt.Errorf("decode record: %w", err)
		}
		if len(rec.Vector) != VectorDim {
			return nil, nil, fmt.Errorf("unexpected vector dim %d", len(rec.Vector))
		}
		vectors = append(vectors, rec.Vector...)
		frauds = append(frauds, rec.Label == "fraud")
	}

	return vectors, frauds, nil
}
