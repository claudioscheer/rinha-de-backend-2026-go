package handler

import (
	"encoding/json"
	"net/http"

	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/search"
	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/vector"
)

type Handler struct {
	ds *dataset.Dataset
}

func New(ds *dataset.Dataset) *Handler {
	return &Handler{ds: ds}
}

type response struct {
	Approved   bool    `json:"approved"`
	FraudScore float64 `json:"fraud_score"`
}

func (h *Handler) Ready(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func (h *Handler) FraudScore(w http.ResponseWriter, r *http.Request) {
	var p vector.Payload
	if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
		http.Error(w, "invalid payload", http.StatusBadRequest)
		return
	}

	q := vector.Vectorize(&p, h.ds)
	score := search.FraudScore(h.ds, q)

	resp := response{
		Approved:   score < 0.6,
		FraudScore: score,
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}
