package vector

import (
	"time"

	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
)

type LastTransaction struct {
	Timestamp     time.Time `json:"timestamp"`
	KmFromCurrent float64   `json:"km_from_current"`
}

// Payload mirrors only the fields actually consumed by Vectorize. Anything
// else in the request body (e.g. `id`) is skipped by the JSON decoder, which
// avoids allocating strings we'd never read.
type Payload struct {
	Transaction struct {
		Amount       float64   `json:"amount"`
		Installments int       `json:"installments"`
		RequestedAt  time.Time `json:"requested_at"`
	} `json:"transaction"`
	Customer struct {
		AvgAmount      float64  `json:"avg_amount"`
		TxCount24h     int      `json:"tx_count_24h"`
		KnownMerchants []string `json:"known_merchants"`
	} `json:"customer"`
	Merchant struct {
		ID        string  `json:"id"`
		MCC       string  `json:"mcc"`
		AvgAmount float64 `json:"avg_amount"`
	} `json:"merchant"`
	Terminal struct {
		IsOnline    bool    `json:"is_online"`
		CardPresent bool    `json:"card_present"`
		KmFromHome  float64 `json:"km_from_home"`
	} `json:"terminal"`
	LastTransaction *LastTransaction `json:"last_transaction"`
}

// Reset clears the payload while keeping the KnownMerchants backing array, so
// a sync.Pool'd Payload can be re-decoded without re-allocating the slice
// header.
func (p *Payload) Reset() {
	merchants := p.Customer.KnownMerchants[:0]
	*p = Payload{}
	p.Customer.KnownMerchants = merchants
}

func clamp01(v float64) float64 { return min(1, max(0, v)) }

// Vectorize turns the payload into a 14-dimensional uint8 vector following
// docs/en/DETECTION_RULES.md.
func Vectorize(p *Payload, ds *dataset.Dataset) [dataset.VectorDim]uint8 {
	raw := vectorizeFloat(p, ds)
	var v [dataset.VectorDim]uint8
	for i, x := range raw {
		v[i] = dataset.Quantize(x)
	}
	return v
}

func Vectorize16(p *Payload, ds *dataset.Dataset) [dataset.VectorDim]uint16 {
	raw := vectorizeFloat(p, ds)
	var v [dataset.VectorDim]uint16
	for i, x := range raw {
		v[i] = dataset.Quantize16(x)
	}
	return v
}

func vectorizeFloat(p *Payload, ds *dataset.Dataset) [dataset.VectorDim]float32 {
	n := ds.Norm
	var v [dataset.VectorDim]float32

	v[0] = float32(clamp01(p.Transaction.Amount / n.MaxAmount))
	v[1] = float32(clamp01(float64(p.Transaction.Installments) / n.MaxInstallments))

	if p.Customer.AvgAmount > 0 {
		v[2] = float32(clamp01((p.Transaction.Amount / p.Customer.AvgAmount) / n.AmountVsAvgRatio))
	} else {
		v[2] = 1
	}

	t := p.Transaction.RequestedAt.UTC()
	v[3] = float32(float64(t.Hour()) / 23.0)
	// Go: Sunday=0..Saturday=6. Spec: Mon=0..Sun=6.
	v[4] = float32(float64((int(t.Weekday())+6)%7) / 6.0)

	if p.LastTransaction != nil {
		minutes := p.Transaction.RequestedAt.Sub(p.LastTransaction.Timestamp).Minutes()
		if minutes < 0 {
			minutes = 0
		}
		v[5] = float32(clamp01(minutes / n.MaxMinutes))
		v[6] = float32(clamp01(p.LastTransaction.KmFromCurrent / n.MaxKm))
	} else {
		v[5] = -1
		v[6] = -1
	}

	v[7] = float32(clamp01(p.Terminal.KmFromHome / n.MaxKm))
	v[8] = float32(clamp01(float64(p.Customer.TxCount24h) / n.MaxTxCount24h))

	if p.Terminal.IsOnline {
		v[9] = 1
	} else {
		v[9] = 0
	}
	if p.Terminal.CardPresent {
		v[10] = 1
	} else {
		v[10] = 0
	}

	known := false
	for _, m := range p.Customer.KnownMerchants {
		if m == p.Merchant.ID {
			known = true
			break
		}
	}
	if known {
		v[11] = 0
	} else {
		v[11] = 1
	}

	if r, ok := ds.MccRisk[p.Merchant.MCC]; ok {
		v[12] = r
	} else {
		v[12] = 0.5
	}

	v[13] = float32(clamp01(p.Merchant.AvgAmount / n.MaxMerchantAvgAmount))

	return v
}
