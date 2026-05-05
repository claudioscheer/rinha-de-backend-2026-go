package handler

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
)

// makeDataset builds a 5-record dataset where every reference is the zero
// vector. Setting fraud=true makes the score 1.0; fraud=false makes it 0.0.
func makeDataset(allFraud bool) *dataset.Dataset {
	flat := make([]uint8, 5*dataset.VectorDim)
	frauds := make([]uint8, 1)
	if allFraud {
		frauds[0] = 0b00011111 // 5 frauds
	}
	ds := dataset.NewForTest(flat, frauds, 5)
	ds.Norm = dataset.Normalization{
		MaxAmount:            10000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     10,
		MaxMinutes:           1440,
		MaxKm:                1000,
		MaxTxCount24h:        20,
		MaxMerchantAvgAmount: 10000,
	}
	ds.MccRisk = map[string]float32{}
	return ds
}

func allFraudDataset() *dataset.Dataset { return makeDataset(true) }
func allLegitDataset() *dataset.Dataset { return makeDataset(false) }

func TestReady(t *testing.T) {
	h := New(allFraudDataset())
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	h.Ready(rr, req)
	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rr.Code)
	}
}

func TestFraudScore_ReturnsApprovedFalse(t *testing.T) {
	h := New(allFraudDataset())
	body := strings.NewReader(`{
      "id": "tx-1",
      "transaction":      { "amount": 100, "installments": 1, "requested_at": "2026-03-11T20:23:35Z" },
      "customer":         { "avg_amount": 100, "tx_count_24h": 1, "known_merchants": ["MERC-001"] },
      "merchant":         { "id": "MERC-001", "mcc": "5411", "avg_amount": 100 },
      "terminal":         { "is_online": false, "card_present": true, "km_from_home": 1 },
      "last_transaction": null
    }`)
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/fraud-score", body)
	req.Header.Set("Content-Type", "application/json")
	h.FraudScore(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rr.Code)
	}
	if ct := rr.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", ct)
	}

	var resp struct {
		Approved   bool    `json:"approved"`
		FraudScore float64 `json:"fraud_score"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Approved {
		t.Errorf("approved = true, want false (all-fraud dataset)")
	}
	if resp.FraudScore != 1.0 {
		t.Errorf("fraud_score = %v, want 1.0", resp.FraudScore)
	}
}

func TestFraudScore_ReturnsApprovedTrue(t *testing.T) {
	h := New(allLegitDataset())
	body := strings.NewReader(`{
      "id": "tx-1",
      "transaction":      { "amount": 100, "installments": 1, "requested_at": "2026-03-11T20:23:35Z" },
      "customer":         { "avg_amount": 100, "tx_count_24h": 1, "known_merchants": ["MERC-001"] },
      "merchant":         { "id": "MERC-001", "mcc": "5411", "avg_amount": 100 },
      "terminal":         { "is_online": false, "card_present": true, "km_from_home": 1 },
      "last_transaction": null
    }`)
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/fraud-score", body)
	h.FraudScore(rr, req)

	var resp struct {
		Approved   bool    `json:"approved"`
		FraudScore float64 `json:"fraud_score"`
	}
	if err := json.Unmarshal(rr.Body.Bytes(), &resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !resp.Approved {
		t.Errorf("approved = false, want true (all-legit dataset)")
	}
	if resp.FraudScore != 0.0 {
		t.Errorf("fraud_score = %v, want 0.0", resp.FraudScore)
	}
}

func TestFraudScore_RejectsInvalidJSON(t *testing.T) {
	h := New(allFraudDataset())
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/fraud-score", strings.NewReader(`{not json`))
	h.FraudScore(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rr.Code)
	}
}

// Routes are registered with method-prefixed patterns; verify the mux
// rejects the wrong method instead of falling through.
func TestRouting_MethodMatters(t *testing.T) {
	h := New(allFraudDataset())
	mux := http.NewServeMux()
	mux.HandleFunc("GET /ready", h.Ready)
	mux.HandleFunc("POST /fraud-score", h.FraudScore)

	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	cases := []struct {
		method string
		path   string
		want   int
	}{
		{http.MethodGet, "/ready", http.StatusOK},
		{http.MethodPost, "/ready", http.StatusMethodNotAllowed},
		{http.MethodGet, "/fraud-score", http.StatusMethodNotAllowed},
		{http.MethodGet, "/missing", http.StatusNotFound},
	}
	for _, c := range cases {
		req, err := http.NewRequest(c.method, srv.URL+c.path, nil)
		if err != nil {
			t.Fatalf("build request: %v", err)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatalf("%s %s: %v", c.method, c.path, err)
		}
		_, _ = io.Copy(io.Discard, resp.Body)
		_ = resp.Body.Close()
		if resp.StatusCode != c.want {
			t.Errorf("%s %s: status = %d, want %d", c.method, c.path, resp.StatusCode, c.want)
		}
	}
}

// Response shape contract: the JSON keys must be exactly approved and
// fraud_score, with the documented types.
func TestFraudScore_ResponseShape(t *testing.T) {
	h := New(allLegitDataset())
	body := strings.NewReader(`{
      "id": "tx-1",
      "transaction":      { "amount": 1, "installments": 1, "requested_at": "2026-03-11T20:23:35Z" },
      "customer":         { "avg_amount": 1, "tx_count_24h": 0, "known_merchants": [] },
      "merchant":         { "id": "M", "mcc": "5411", "avg_amount": 1 },
      "terminal":         { "is_online": false, "card_present": true, "km_from_home": 0 },
      "last_transaction": null
    }`)
	rr := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/fraud-score", body)
	h.FraudScore(rr, req)

	var raw map[string]any
	if err := json.Unmarshal(rr.Body.Bytes(), &raw); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if _, ok := raw["approved"].(bool); !ok {
		t.Errorf("approved missing or not a bool: %#v", raw["approved"])
	}
	if _, ok := raw["fraud_score"].(float64); !ok {
		t.Errorf("fraud_score missing or not a number: %#v", raw["fraud_score"])
	}
	for k := range raw {
		if k != "approved" && k != "fraud_score" {
			t.Errorf("unexpected response field: %q", k)
		}
	}
}
