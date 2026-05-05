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
