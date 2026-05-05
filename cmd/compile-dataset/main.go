// compile-dataset converts the gzipped JSON reference dataset into the
// compact binary form (uint8-quantized vectors + packed fraud bitmap) the API
// process mmaps at startup. Run once at container build time.
package main

import (
	"flag"
	"log"
	"os"

	"github.com/claudioscheer/rinha-de-backend-2026-go/internal/dataset"
)

func main() {
	in := flag.String("in", "/app/resources/references.json.gz", "input references.json.gz")
	out := flag.String("out", "/app/resources/references.bin", "output references.bin")
	flag.Parse()

	if err := dataset.CompileFromJSONGz(*in, *out); err != nil {
		log.Fatalf("compile: %v", err)
	}
	st, err := os.Stat(*out)
	if err != nil {
		log.Fatalf("stat output: %v", err)
	}
	log.Printf("wrote %s (%d bytes)", *out, st.Size())
}
