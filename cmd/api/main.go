package main

import (
	"log"
	"net/http"
	"os"
	"strconv"

	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/handler"
	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/search"
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
	opts := searchOptionsFromEnv()
	log.Printf("search options: nprobe=%d max_nprobe=%d adaptive=%v", opts.Nprobe, opts.MaxNprobe, opts.Adaptive)
	h := handler.NewWithOptions(ds, opts)
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

func searchOptionsFromEnv() search.Options {
	opts := search.DefaultOptions
	if v := os.Getenv("SEARCH_NPROBE"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			opts.Nprobe = n
		}
	}
	if v := os.Getenv("SEARCH_MAX_NPROBE"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			opts.MaxNprobe = n
		}
	}
	if v := os.Getenv("SEARCH_ADAPTIVE"); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			opts.Adaptive = b
		}
	}
	return opts
}
