package handler_test

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/handler"
)

// writeFixture produces a real on-disk dataset (gzipped references +
// normalization + mcc_risk) the loader can ingest, then returns the dir.
func writeFixture(t *testing.T, refsJSON string) string {
	t.Helper()
	dir := t.TempDir()

	must := func(path, content string) {
		if err := os.WriteFile(filepath.Join(dir, path), []byte(content), 0o644); err != nil {
			t.Fatalf("write %s: %v", path, err)
		}
	}
	must("normalization.json", `{
      "max_amount": 10000,
      "max_installments": 12,
      "amount_vs_avg_ratio": 10,
      "max_minutes": 1440,
      "max_km": 1000,
      "max_tx_count_24h": 20,
      "max_merchant_avg_amount": 10000
    }`)
	must("mcc_risk.json", `{"5411": 0.15, "7802": 0.75}`)

	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	if _, err := gz.Write([]byte(refsJSON)); err != nil {
		t.Fatalf("gzip write: %v", err)
	}
	if err := gz.Close(); err != nil {
		t.Fatalf("gzip close: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "references.json.gz"), buf.Bytes(), 0o644); err != nil {
		t.Fatalf("write refs: %v", err)
	}

	return dir
}

// End-to-end: load fixtures from disk, spin up the mux, hit the endpoints
// over HTTP, and verify both /ready and /fraud-score behave per the spec.
func TestEndToEnd_LowRiskApproved(t *testing.T) {
	// 5 legit references near origin → score 0, approved=true.
	dir := writeFixture(t, `[
      {"vector":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"legit"},
      {"vector":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"legit"},
      {"vector":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"legit"},
      {"vector":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"legit"},
      {"vector":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"legit"}
    ]`)
	ds, err := dataset.Load(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	h := handler.New(ds)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /ready", h.Ready)
	mux.HandleFunc("POST /fraud-score", h.FraudScore)

	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	// /ready
	resp, err := http.Get(srv.URL + "/ready")
	if err != nil {
		t.Fatalf("GET /ready: %v", err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Errorf("/ready status = %d, want 200", resp.StatusCode)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()

	// /fraud-score
	body := strings.NewReader(`{
      "id": "tx-e2e-1",
      "transaction":      { "amount": 1, "installments": 1, "requested_at": "2026-03-11T20:23:35Z" },
      "customer":         { "avg_amount": 1, "tx_count_24h": 0, "known_merchants": ["MERC-001"] },
      "merchant":         { "id": "MERC-001", "mcc": "5411", "avg_amount": 1 },
      "terminal":         { "is_online": false, "card_present": true, "km_from_home": 0 },
      "last_transaction": null
    }`)
	resp, err = http.Post(srv.URL+"/fraud-score", "application/json", body)
	if err != nil {
		t.Fatalf("POST /fraud-score: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("/fraud-score status = %d, want 200", resp.StatusCode)
	}
	var out struct {
		Approved   bool    `json:"approved"`
		FraudScore float64 `json:"fraud_score"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if !out.Approved || out.FraudScore != 0 {
		t.Errorf("got %+v, want approved=true score=0", out)
	}
}

func TestEndToEnd_HighRiskDenied(t *testing.T) {
	// 5 fraud references → score 1, approved=false.
	dir := writeFixture(t, `[
      {"vector":[1,1,1,1,1,1,1,1,1,1,1,1,1,1],"label":"fraud"},
      {"vector":[1,1,1,1,1,1,1,1,1,1,1,1,1,1],"label":"fraud"},
      {"vector":[1,1,1,1,1,1,1,1,1,1,1,1,1,1],"label":"fraud"},
      {"vector":[1,1,1,1,1,1,1,1,1,1,1,1,1,1],"label":"fraud"},
      {"vector":[1,1,1,1,1,1,1,1,1,1,1,1,1,1],"label":"fraud"}
    ]`)
	ds, err := dataset.Load(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	h := handler.New(ds)
	mux := http.NewServeMux()
	mux.HandleFunc("POST /fraud-score", h.FraudScore)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	body := strings.NewReader(`{
      "id": "tx-e2e-2",
      "transaction":      { "amount": 9999, "installments": 12, "requested_at": "2026-03-14T05:15:12Z" },
      "customer":         { "avg_amount": 1, "tx_count_24h": 20, "known_merchants": [] },
      "merchant":         { "id": "MERC-X", "mcc": "7802", "avg_amount": 9999 },
      "terminal":         { "is_online": true, "card_present": false, "km_from_home": 999 },
      "last_transaction": null
    }`)
	resp, err := http.Post(srv.URL+"/fraud-score", "application/json", body)
	if err != nil {
		t.Fatalf("POST /fraud-score: %v", err)
	}
	defer resp.Body.Close()

	var out struct {
		Approved   bool    `json:"approved"`
		FraudScore float64 `json:"fraud_score"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.Approved || out.FraudScore != 1.0 {
		t.Errorf("got %+v, want approved=false score=1.0", out)
	}
}

// Threshold boundary: with K=5 majority vote, 2 frauds out of 5 → score 0.4 →
// approved=true (minority of neighbors are fraud).
func TestEndToEnd_ThresholdExact(t *testing.T) {
	dir := writeFixture(t, `[
      {"vector":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"fraud"},
      {"vector":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"fraud"},
      {"vector":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"legit"},
      {"vector":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"legit"},
      {"vector":[0,0,0,0,0,0,0,0,0,0,0,0,0,0],"label":"legit"}
    ]`)
	ds, err := dataset.Load(dir)
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	h := handler.New(ds)
	mux := http.NewServeMux()
	mux.HandleFunc("POST /fraud-score", h.FraudScore)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)

	resp, err := http.Post(srv.URL+"/fraud-score", "application/json", strings.NewReader(`{
      "id": "tx-edge",
      "transaction":      { "amount": 1, "installments": 1, "requested_at": "2026-03-11T00:00:00Z" },
      "customer":         { "avg_amount": 1, "tx_count_24h": 0, "known_merchants": [] },
      "merchant":         { "id": "M", "mcc": "5411", "avg_amount": 1 },
      "terminal":         { "is_online": false, "card_present": true, "km_from_home": 0 },
      "last_transaction": null
    }`))
	if err != nil {
		t.Fatalf("POST: %v", err)
	}
	defer resp.Body.Close()

	var out struct {
		Approved   bool    `json:"approved"`
		FraudScore float64 `json:"fraud_score"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if out.FraudScore != 0.4 {
		t.Fatalf("score = %v, want 0.4", out.FraudScore)
	}
	if !out.Approved {
		t.Errorf("approved = false at score 0.4, but majority-vote threshold is < 0.5")
	}
}
