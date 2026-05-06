package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"time"

	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/search"
	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/vector"
)

type testFile struct {
	Stats   map[string]any `json:"stats"`
	Entries []testEntry    `json:"entries"`
}

type testEntry struct {
	Request            vector.Payload `json:"request"`
	ExpectedApproved   bool           `json:"expected_approved"`
	ExpectedFraudScore float64        `json:"expected_fraud_score"`
}

type breakdown struct {
	TruePositive  int `json:"true_positive_detections"`
	TrueNegative  int `json:"true_negative_detections"`
	FalsePositive int `json:"false_positive_detections"`
	FalseNegative int `json:"false_negative_detections"`
	Errors        int `json:"http_errors"`
}

type report struct {
	ResourcesDir string         `json:"resources_dir"`
	TestDataPath string         `json:"test_data_path"`
	Elapsed      string         `json:"elapsed"`
	Options      search.Options `json:"options"`
	Threshold    float64        `json:"approval_threshold"`
	Expected     map[string]any `json:"expected"`
	Breakdown    breakdown      `json:"breakdown"`
	Total        int            `json:"total"`
	ScoreMatches int            `json:"fraud_score_matches"`
	ScoreDiffs   int            `json:"fraud_score_diffs"`
	FailureRate  float64        `json:"failure_rate"`
	WeightedE    int            `json:"weighted_errors_E"`
	Epsilon      float64        `json:"error_rate_epsilon"`
	Detection    float64        `json:"detection_score"`
}

type errorEntry struct {
	Index            int            `json:"index"`
	Request          vector.Payload `json:"request"`
	ExpectedApproved bool           `json:"expected_approved"`
	Predicted        bool           `json:"predicted_approved"`
	FraudScore       float64        `json:"fraud_score"`
}

type predictionEntry struct {
	Index              int            `json:"index"`
	Request            vector.Payload `json:"request"`
	ExpectedApproved   bool           `json:"expected_approved"`
	ExpectedFraudScore float64        `json:"expected_fraud_score"`
	Predicted          bool           `json:"predicted_approved"`
	FraudScore         float64        `json:"fraud_score"`
}

func main() {
	resourcesDir := flag.String("resources", "resources", "directory with normalization, mcc_risk, references")
	testDataPath := flag.String("test-data", "test/test-data.json", "labeled test-data.json path")
	limit := flag.Int("limit", 0, "optional number of entries to evaluate")
	threshold := flag.Float64("threshold", 0.6, "approval threshold applied as score < threshold")
	nprobe := flag.Int("nprobe", search.DefaultNprobe, "initial IVF probe count")
	maxNprobe := flag.Int("max-nprobe", search.DefaultMaxNprobe, "adaptive IVF probe ceiling")
	adaptive := flag.Bool("adaptive", search.DefaultOptions.Adaptive, "extend probing for borderline provisional votes")
	errorsOut := flag.String("errors-out", "", "optional JSON file with misclassified entries")
	predictionsOut := flag.String("predictions-out", "", "optional JSON file with every prediction")
	flag.Parse()

	ds, err := dataset.Load(*resourcesDir)
	if err != nil {
		log.Fatalf("load dataset: %v", err)
	}
	defer func() { _ = ds.Close() }()

	data, err := os.ReadFile(*testDataPath)
	if err != nil {
		log.Fatalf("read test data: %v", err)
	}
	var tf testFile
	if err := json.Unmarshal(data, &tf); err != nil {
		log.Fatalf("parse test data: %v", err)
	}
	if *limit > 0 && *limit < len(tf.Entries) {
		tf.Entries = tf.Entries[:*limit]
	}

	opts := search.Options{
		Nprobe:    *nprobe,
		MaxNprobe: *maxNprobe,
		Adaptive:  *adaptive,
	}
	start := time.Now()
	var b breakdown
	scoreMatches := 0
	var errors []errorEntry
	var predictions []predictionEntry
	for i := range tf.Entries {
		entry := &tf.Entries[i]
		var score float64
		if ds.Precision == dataset.Precision16 {
			q := vector.Vectorize16(&entry.Request, ds)
			score = search.FraudScore16WithOptions(ds, q, opts)
		} else {
			q := vector.Vectorize(&entry.Request, ds)
			score = search.FraudScoreWithOptions(ds, q, opts)
		}
		approved := score < *threshold

		if score == entry.ExpectedFraudScore {
			scoreMatches++
		}
		if *predictionsOut != "" {
			predictions = append(predictions, predictionEntry{
				Index:              i,
				Request:            entry.Request,
				ExpectedApproved:   entry.ExpectedApproved,
				ExpectedFraudScore: entry.ExpectedFraudScore,
				Predicted:          approved,
				FraudScore:         score,
			})
		}
		if approved == entry.ExpectedApproved {
			if approved {
				b.TrueNegative++
			} else {
				b.TruePositive++
			}
			continue
		}
		if approved {
			b.FalseNegative++
		} else {
			b.FalsePositive++
		}
		if *errorsOut != "" {
			errors = append(errors, errorEntry{
				Index:            i,
				Request:          entry.Request,
				ExpectedApproved: entry.ExpectedApproved,
				Predicted:        approved,
				FraudScore:       score,
			})
		}
	}

	total := b.TruePositive + b.TrueNegative + b.FalsePositive + b.FalseNegative + b.Errors
	weighted := b.FalsePositive + 3*b.FalseNegative + 5*b.Errors
	failureRate := 0.0
	epsilon := 0.0
	if total > 0 {
		failureRate = float64(b.FalsePositive+b.FalseNegative+b.Errors) / float64(total)
		epsilon = float64(weighted) / float64(total)
	}

	rep := report{
		ResourcesDir: *resourcesDir,
		TestDataPath: *testDataPath,
		Elapsed:      time.Since(start).String(),
		Options:      opts,
		Threshold:    *threshold,
		Expected:     tf.Stats,
		Breakdown:    b,
		Total:        total,
		ScoreMatches: scoreMatches,
		ScoreDiffs:   total - scoreMatches,
		FailureRate:  round6(failureRate),
		WeightedE:    weighted,
		Epsilon:      round6(epsilon),
		Detection:    round2(detectionScore(total, weighted, failureRate)),
	}
	out, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		log.Fatalf("marshal report: %v", err)
	}
	fmt.Println(string(out))
	if *errorsOut != "" {
		data, err := json.MarshalIndent(errors, "", "  ")
		if err != nil {
			log.Fatalf("marshal errors: %v", err)
		}
		if err := os.WriteFile(*errorsOut, data, 0o644); err != nil {
			log.Fatalf("write errors: %v", err)
		}
	}
	if *predictionsOut != "" {
		data, err := json.MarshalIndent(predictions, "", "  ")
		if err != nil {
			log.Fatalf("marshal predictions: %v", err)
		}
		if err := os.WriteFile(*predictionsOut, data, 0o644); err != nil {
			log.Fatalf("write predictions: %v", err)
		}
	}
}

func detectionScore(total, weighted int, failureRate float64) float64 {
	const (
		k          = 1000.0
		epsilonMin = 0.001
		beta       = 300.0
		cutoff     = 0.15
	)
	if total == 0 {
		return 0
	}
	if failureRate > cutoff {
		return -3000
	}
	epsilon := float64(weighted) / float64(total)
	if epsilon < epsilonMin {
		epsilon = epsilonMin
	}
	return k*math.Log10(1/epsilon) - beta*math.Log10(1+float64(weighted))
}

func round2(v float64) float64 { return math.Round(v*100) / 100 }
func round6(v float64) float64 { return math.Round(v*1_000_000) / 1_000_000 }
