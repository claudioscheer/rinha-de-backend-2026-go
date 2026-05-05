package vector

import (
	"time"

	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
)

type LastTransaction struct {
	Timestamp     time.Time `json:"timestamp"`
	KmFromCurrent float64   `json:"km_from_current"`
}

type Payload struct {
	ID          string `json:"id"`
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

func clamp(v float64) float64 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

// Vectorize turns the payload into a 14-dimensional vector following
// docs/en/DETECTION_RULES.md.
func Vectorize(p *Payload, ds *dataset.Dataset) [dataset.VectorDim]float32 {
	n := ds.Norm
	var v [dataset.VectorDim]float32

	v[0] = float32(clamp(p.Transaction.Amount / n.MaxAmount))
	v[1] = float32(clamp(float64(p.Transaction.Installments) / n.MaxInstallments))

	if p.Customer.AvgAmount > 0 {
		v[2] = float32(clamp((p.Transaction.Amount / p.Customer.AvgAmount) / n.AmountVsAvgRatio))
	} else {
		v[2] = 1
	}

	t := p.Transaction.RequestedAt.UTC()
	v[3] = float32(float64(t.Hour()) / 23.0)
	// Go: Sunday=0..Saturday=6. Spec: Mon=0..Sun=6.
	wd := int(t.Weekday())
	monBased := (wd + 6) % 7
	v[4] = float32(float64(monBased) / 6.0)

	if p.LastTransaction != nil {
		minutes := p.Transaction.RequestedAt.Sub(p.LastTransaction.Timestamp).Minutes()
		if minutes < 0 {
			minutes = 0
		}
		v[5] = float32(clamp(minutes / n.MaxMinutes))
		v[6] = float32(clamp(p.LastTransaction.KmFromCurrent / n.MaxKm))
	} else {
		v[5] = -1
		v[6] = -1
	}

	v[7] = float32(clamp(p.Terminal.KmFromHome / n.MaxKm))
	v[8] = float32(clamp(float64(p.Customer.TxCount24h) / n.MaxTxCount24h))

	if p.Terminal.IsOnline {
		v[9] = 1
	}
	if p.Terminal.CardPresent {
		v[10] = 1
	}

	known := false
	for _, m := range p.Customer.KnownMerchants {
		if m == p.Merchant.ID {
			known = true
			break
		}
	}
	if !known {
		v[11] = 1
	}

	if r, ok := ds.MccRisk[p.Merchant.MCC]; ok {
		v[12] = r
	} else {
		v[12] = 0.5
	}

	v[13] = float32(clamp(p.Merchant.AvgAmount / n.MaxMerchantAvgAmount))

	return v
}
