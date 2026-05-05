# Agent memory

Project memory for the coding agents working on this Rinha de Backend 2026
entry. Both **Claude Code** and **Codex** read this file (Codex via the
`AGENTS.md` symlink that points here) so the working agreements stay the
same regardless of which agent is driving.

This is an **AI-only experiment**: every line of code, test, and config in
this repo is produced by Claude Code or Codex under human direction (see
the README). Keep it that way.

## What this project is

A Go HTTP service that scores transactions for fraud using K-NN over a fixed
reference dataset of 3 M labeled vectors (14 dimensions each). Search is
**approximate**: an IVF (inverted-file) index built at container-build time
partitions the references into 1 024 clusters; queries rank centroids and
scan only the nearest `Nprobe` buckets (default 8 → ~0.8 % of the dataset).
Brute-force K-NN is preserved as the fallback when no IVF index is present
(used by unit tests). Two API replicas + nginx load balancer, all under
tight CPU/memory caps set by the contest.

The challenge spec is the source of truth — read it before changing
behavior:

- [docs/en/README.md](docs/en/README.md) — overview (Portuguese:
  [docs/br/README.md](docs/br/README.md))
- [docs/en/API.md](docs/en/API.md) — `POST /fraud-score`, `GET /ready`
  contract on port 9999
- [docs/en/ARCHITECTURE.md](docs/en/ARCHITECTURE.md) — required topology,
  CPU/memory budgets, network rules
- [docs/en/DETECTION_RULES.md](docs/en/DETECTION_RULES.md) — the 14-dim
  vectorization formulas and the `-1` sentinel rule
- [docs/en/DATASET.md](docs/en/DATASET.md) — `references.json.gz`,
  `mcc_risk.json`, `normalization.json` formats
- [docs/en/VECTOR_SEARCH.md](docs/en/VECTOR_SEARCH.md) — KNN background and
  the K=5 majority-vote scoring
- [docs/en/EVALUATION.md](docs/en/EVALUATION.md) — how the load test scores
  submissions
- [docs/en/SUBMISSION.md](docs/en/SUBMISSION.md) — the submission process

## Code layout

- `cmd/api/main.go` — HTTP server entry point
- `cmd/compile-dataset/main.go` — build-time tool that converts
  `references.json.gz` into the compact `references.bin` blob the API mmaps.
  Runs k-means clustering, reorders vectors by cluster id, and writes the
  IVF index inline.
- `internal/dataset/` — quantized reference store. Owns the on-disk binary
  format (uint8 vectors + IVF centroids + cluster offsets + packed fraud
  bitmap) and the mmap loader. Has a JSON fallback path used only by tests.
  `kmeans.go` holds the build-time k-means + bucket-sort.
- `internal/vector/` — payload struct + `Vectorize` (turns a request into a
  `[14]uint8` query vector). Pool-friendly via `Payload.Reset`.
- `internal/search/` — IVF K-NN with int32 squared distances and
  bounds-check-elided inner loop. Falls back to brute force when the
  dataset has no centroids (`NumClusters == 0`).
- `internal/handler/` — HTTP handlers, request decoding, manual JSON
  response writing.
- `resources/` — fixed input data shipped to the container. The
  `references.bin` blob is **built**, not committed (see `.gitignore`).

## Constraints to respect

These come from the contest config (`config.json`, `docker-compose.yml`):

- **2 API replicas + 1 LB**, total **1.0 CPU**, total **350 MB RAM**.
- Each replica: **0.4 CPU**, **160 MB RAM** — the Go heap is bounded with
  `GOMEMLIMIT=80MiB` to leave headroom for the mmap'd reference blob.
- Endpoints: only `GET /ready` and `POST /fraud-score`. Anything else
  should 404 / 405.
- `references.json.gz`, `mcc_risk.json`, `normalization.json` do not change
  at runtime — they can be freely pre-processed at build time.
- Vector dim 14, `-1` sentinel for missing-history dims (5 and 6) must be
  preserved end-to-end (it quantizes to byte 0, distinct from in-range
  values which start at ~127).

## Working agreements

- **Run `go test ./...` before committing.** All four internal packages
  have tests; keep them green. The dataset package has a `TestLoad_RealNormalizationAndMcc` guard that catches typos in
  `resources/*.json`.
- **`go vet ./...` clean.** No exceptions.
- **Format with `gofmt -w`.** No editor imposing other styles.
- **Benchmarks live in `internal/search/bench_test.go`** and require a
  built `resources/references.bin` (skip otherwise). Run with
  `go test -bench=. -run=^$ -cpu=1 ./internal/search/` after compiling the
  blob with `go run ./cmd/compile-dataset -in resources/references.json.gz
  -out resources/references.bin`.
- **No new dependencies** unless necessary. Standard library only is the
  current state and has been enough.
- **Commits use `claudioscheer <claudioscheer@protonmail.com>`** (set in
  `.git/config`). **Never add agent attribution to commit messages** —
  no `Co-Authored-By: Claude`, no `Co-Authored-By: Codex`, no
  `https://claude.ai/...` or `https://chat.openai.com/...` links. The
  AI-only nature of the project is documented in the README; it does not
  need to bleed into the commit log.
- **PRs are the human's call.** Open as draft when asked; don't merge
  without explicit permission.
- **Don't edit `participants/*.json`** — those belong to the upstream
  contest repo.

## Performance characteristics (for context, not contracts)

Measured locally with `BenchmarkFraudScore_Real` against the real 3 M
dataset on a 2.8 GHz Xeon, GOMAXPROCS=1, `Nprobe=8`:

- ~285 µs / `FraudScore` call (~119× faster than the previous brute-force
  implementation).
- Per-replica RSS ~50 MB (mmap'd blob is the same size as before — IVF
  metadata is ~14 KB centroids + 4 KB offsets, dwarfed by the 42 MB
  reference vectors).

Compile cost: `cmd/compile-dataset` runs Lloyd's k-means with 15 iterations
across `runtime.GOMAXPROCS` workers. Roughly 5 minutes for the production
3 M / 1 024-cluster build on a 4-core host. This is build-time only; runtime
is unaffected.

Tuning knobs:

- `internal/search.Nprobe` — number of clusters scanned per query. Higher
  raises recall and latency. The current 8 was chosen as a starting point;
  retune empirically against the contest test if detection score regresses.
- `internal/dataset.chooseNumClusters` — picks 1 024 clusters for the
  production dataset, 64 for medium fixtures, and 0 (= flat layout, brute
  force) for tiny tests.
