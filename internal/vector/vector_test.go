package vector

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
)

func testDataset() *dataset.Dataset {
	return &dataset.Dataset{
		Norm: dataset.Normalization{
			MaxAmount:            10000,
			MaxInstallments:      12,
			AmountVsAvgRatio:     10,
			MaxMinutes:           1440,
			MaxKm:                1000,
			MaxTxCount24h:        20,
			MaxMerchantAvgAmount: 10000,
		},
		MccRisk: map[string]float32{
			"5411": 0.15,
			"7802": 0.75,
		},
	}
}

// Example from docs/en/DETECTION_RULES.md (the "Flow overview" legitimate
// transaction).
func TestVectorize_LegitExample(t *testing.T) {
	raw := `{
      "id": "tx-1329056812",
      "transaction":      { "amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z" },
      "customer":         { "avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"] },
      "merchant":         { "id": "MERC-016", "mcc": "5411", "avg_amount": 60.25 },
      "terminal":         { "is_online": false, "card_present": true, "km_from_home": 29.23 },
      "last_transaction": null
    }`
	var p Payload
	if err := json.Unmarshal([]byte(raw), &p); err != nil {
		t.Fatalf("decode: %v", err)
	}

	got := Vectorize(&p, testDataset())
	want := [dataset.VectorDim]float32{
		0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006,
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-3 {
			t.Errorf("dim %d: got %v want %v", i, got[i], want[i])
		}
	}
}

// With last_transaction populated, indices 5 and 6 should be normalized
// (and not -1).
func TestVectorize_WithLastTransaction(t *testing.T) {
	raw := `{
      "id": "tx-1",
      "transaction":      { "amount": 100, "installments": 1, "requested_at": "2026-03-11T20:23:35Z" },
      "customer":         { "avg_amount": 100, "tx_count_24h": 1, "known_merchants": ["MERC-001"] },
      "merchant":         { "id": "MERC-001", "mcc": "5411", "avg_amount": 100 },
      "terminal":         { "is_online": true, "card_present": false, "km_from_home": 10 },
      "last_transaction": { "timestamp": "2026-03-11T19:23:35Z", "km_from_current": 50 }
    }`
	var p Payload
	if err := json.Unmarshal([]byte(raw), &p); err != nil {
		t.Fatalf("decode: %v", err)
	}

	got := Vectorize(&p, testDataset())
	// 60 minutes / 1440 = 0.04166...
	if math.Abs(float64(got[5])-60.0/1440.0) > 1e-4 {
		t.Errorf("dim 5 minutes_since_last_tx: got %v", got[5])
	}
	// 50 km / 1000 = 0.05
	if math.Abs(float64(got[6])-0.05) > 1e-4 {
		t.Errorf("dim 6 km_from_last_tx: got %v", got[6])
	}
	if got[9] != 1 {
		t.Errorf("dim 9 is_online: got %v want 1", got[9])
	}
	if got[10] != 0 {
		t.Errorf("dim 10 card_present: got %v want 0", got[10])
	}
	if got[11] != 0 {
		t.Errorf("dim 11 unknown_merchant: got %v want 0 (merchant is known)", got[11])
	}
}

// Values above the max should clamp to 1.0.
func TestVectorize_Clamps(t *testing.T) {
	raw := `{
      "id": "tx-big",
      "transaction":      { "amount": 99999999, "installments": 99, "requested_at": "2026-03-11T00:00:00Z" },
      "customer":         { "avg_amount": 1, "tx_count_24h": 9999, "known_merchants": [] },
      "merchant":         { "id": "MERC-X", "mcc": "5411", "avg_amount": 99999999 },
      "terminal":         { "is_online": false, "card_present": true, "km_from_home": 99999 },
      "last_transaction": null
    }`
	var p Payload
	if err := json.Unmarshal([]byte(raw), &p); err != nil {
		t.Fatalf("decode: %v", err)
	}

	got := Vectorize(&p, testDataset())
	for _, idx := range []int{0, 1, 2, 7, 8, 13} {
		if got[idx] != 1 {
			t.Errorf("dim %d should clamp to 1.0, got %v", idx, got[idx])
		}
	}
}

// avg_amount == 0 must not divide by zero; the spec doesn't say what to do,
// but the implementation must be safe.
func TestVectorize_ZeroAvgAmount(t *testing.T) {
	raw := `{
      "id": "tx-zero",
      "transaction":      { "amount": 100, "installments": 1, "requested_at": "2026-03-11T00:00:00Z" },
      "customer":         { "avg_amount": 0, "tx_count_24h": 0, "known_merchants": [] },
      "merchant":         { "id": "MERC-X", "mcc": "5411", "avg_amount": 50 },
      "terminal":         { "is_online": false, "card_present": false, "km_from_home": 0 },
      "last_transaction": null
    }`
	var p Payload
	if err := json.Unmarshal([]byte(raw), &p); err != nil {
		t.Fatalf("decode: %v", err)
	}

	got := Vectorize(&p, testDataset())
	if math.IsNaN(float64(got[2])) || math.IsInf(float64(got[2]), 0) {
		t.Errorf("dim 2 must be finite when avg_amount is 0, got %v", got[2])
	}
}

// MCCs that are not in mcc_risk.json must default to 0.5.
func TestVectorize_UnknownMcc(t *testing.T) {
	raw := `{
      "id": "tx-mcc",
      "transaction":      { "amount": 1, "installments": 1, "requested_at": "2026-03-11T00:00:00Z" },
      "customer":         { "avg_amount": 1, "tx_count_24h": 0, "known_merchants": [] },
      "merchant":         { "id": "MERC-X", "mcc": "9999", "avg_amount": 1 },
      "terminal":         { "is_online": false, "card_present": false, "km_from_home": 0 },
      "last_transaction": null
    }`
	var p Payload
	if err := json.Unmarshal([]byte(raw), &p); err != nil {
		t.Fatalf("decode: %v", err)
	}

	got := Vectorize(&p, testDataset())
	if got[12] != 0.5 {
		t.Errorf("dim 12 unknown mcc default: got %v want 0.5", got[12])
	}
}

// Mon=0..Sun=6 mapping. 2026-03-09 is a Monday.
func TestVectorize_WeekdayMapping(t *testing.T) {
	cases := []struct {
		date string
		want float32
	}{
		{"2026-03-09T12:00:00Z", 0.0},      // Mon
		{"2026-03-10T12:00:00Z", 1.0 / 6.0}, // Tue
		{"2026-03-13T12:00:00Z", 4.0 / 6.0}, // Fri
		{"2026-03-15T12:00:00Z", 1.0},      // Sun
	}
	for _, c := range cases {
		raw := `{
          "id": "tx",
          "transaction":      { "amount": 1, "installments": 1, "requested_at": "` + c.date + `" },
          "customer":         { "avg_amount": 1, "tx_count_24h": 0, "known_merchants": [] },
          "merchant":         { "id": "M", "mcc": "5411", "avg_amount": 1 },
          "terminal":         { "is_online": false, "card_present": false, "km_from_home": 0 },
          "last_transaction": null
        }`
		var p Payload
		if err := json.Unmarshal([]byte(raw), &p); err != nil {
			t.Fatalf("decode: %v", err)
		}
		got := Vectorize(&p, testDataset())
		if math.Abs(float64(got[4]-c.want)) > 1e-4 {
			t.Errorf("date %s: got dim4 %v want %v", c.date, got[4], c.want)
		}
	}
}

// Example from docs/en/DETECTION_RULES.md (the fraudulent transaction).
func TestVectorize_FraudExample(t *testing.T) {
	raw := `{
      "id": "tx-3330991687",
      "transaction":      { "amount": 9505.97, "installments": 10, "requested_at": "2026-03-14T05:15:12Z" },
      "customer":         { "avg_amount": 81.28, "tx_count_24h": 20, "known_merchants": ["MERC-008", "MERC-007", "MERC-005"] },
      "merchant":         { "id": "MERC-068", "mcc": "7802", "avg_amount": 54.86 },
      "terminal":         { "is_online": false, "card_present": true, "km_from_home": 952.27 },
      "last_transaction": null
    }`
	var p Payload
	if err := json.Unmarshal([]byte(raw), &p); err != nil {
		t.Fatalf("decode: %v", err)
	}

	got := Vectorize(&p, testDataset())
	want := [dataset.VectorDim]float32{
		0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055,
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > 1e-3 {
			t.Errorf("dim %d: got %v want %v", i, got[i], want[i])
		}
	}
}
