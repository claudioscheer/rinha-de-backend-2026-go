package handler

import (
	"encoding/json"
	"net/http"
	"strconv"
	"sync"

	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/search"
	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/vector"
)

type Handler struct {
	ds   *dataset.Dataset
	opts search.Options
}

func New(ds *dataset.Dataset) *Handler {
	return NewWithOptions(ds, search.DefaultOptions)
}

func NewWithOptions(ds *dataset.Dataset, opts search.Options) *Handler {
	return &Handler{ds: ds, opts: opts}
}

var (
	payloadPool = sync.Pool{
		New: func() any { return &vector.Payload{} },
	}
	bufPool = sync.Pool{
		New: func() any {
			b := make([]byte, 0, 64)
			return &b
		},
	}
)

func (h *Handler) Ready(w http.ResponseWriter, _ *http.Request) {
	w.WriteHeader(http.StatusOK)
}

func (h *Handler) FraudScore(w http.ResponseWriter, r *http.Request) {
	p := payloadPool.Get().(*vector.Payload)
	p.Reset()
	defer payloadPool.Put(p)

	if err := json.NewDecoder(r.Body).Decode(p); err != nil {
		http.Error(w, "invalid payload", http.StatusBadRequest)
		return
	}

	var score float64
	if h.ds.Precision == dataset.Precision16 {
		q := vector.Vectorize16(p, h.ds)
		score = search.FraudScore16WithOptions(h.ds, q, h.opts)
	} else {
		q := vector.Vectorize(p, h.ds)
		score = search.FraudScoreWithOptions(h.ds, q, h.opts)
	}

	bp := bufPool.Get().(*[]byte)
	buf := (*bp)[:0]
	buf = append(buf, `{"approved":`...)
	if score < 0.5 {
		buf = append(buf, "true"...)
	} else {
		buf = append(buf, "false"...)
	}
	buf = append(buf, `,"fraud_score":`...)
	buf = strconv.AppendFloat(buf, score, 'f', -1, 64)
	buf = append(buf, '}')

	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(buf)

	*bp = buf[:0]
	bufPool.Put(bp)
}
