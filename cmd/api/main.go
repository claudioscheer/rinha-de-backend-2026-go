package main

import (
	"log"
	"net/http"
	"os"

	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/handler"
)

func main() {
	resourcesDir := getEnv("RESOURCES_DIR", "/app/resources")
	addr := getEnv("LISTEN_ADDR", ":9999")

	log.Printf("loading dataset from %s", resourcesDir)
	ds, err := dataset.Load(resourcesDir)
	if err != nil {
		log.Fatalf("failed to load dataset: %v", err)
	}
	log.Printf("loaded %d reference vectors", ds.Size())

	mux := http.NewServeMux()
	h := handler.New(ds)
	mux.HandleFunc("GET /ready", h.Ready)
	mux.HandleFunc("POST /fraud-score", h.FraudScore)

	srv := &http.Server{
		Addr:    addr,
		Handler: mux,
	}

	log.Printf("listening on %s", addr)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("server error: %v", err)
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
