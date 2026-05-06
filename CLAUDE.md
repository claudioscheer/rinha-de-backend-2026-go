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
partitions the references into 1 024 clusters; queries rank centroids and scan
the nearest `Nprobe` buckets (production default 12) and adaptively expand to
24 buckets on 2-vs-3 provisional votes. Wider adaptive ceilings and 16-bit
blobs exist as tuning hooks, but are not enabled by default because they cost
too much p99 in the preview load test.
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
- `cmd/evaluate/main.go` — offline evaluator for labeled payload files. Use it
  to sweep IVF probe counts/thresholds before spending full k6 runs.
- `internal/dataset/` — quantized reference store. Owns the on-disk binary
  format (uint8/uint16 vectors + IVF centroids + cluster offsets + packed fraud
  bitmap) and the mmap loader. Has a JSON fallback path used only by tests.
  `kmeans.go` holds the build-time k-means + bucket-sort.
- `internal/vector/` — payload struct + `Vectorize`/`Vectorize16` (turns a
  request into a quantized query vector). Pool-friendly via `Payload.Reset`.
- `internal/search/` — IVF K-NN with integer squared distances,
  bounds-check-elided inner loops, and adaptive probing. Falls back to brute
  force when the dataset has no centroids (`NumClusters == 0`).
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
  -out resources/references.bin -precision 8 -clusters 1024`.
- **No new dependencies** unless necessary. Standard library only is the
  current state and has been enough.
- **Commits use `claudioscheer <claudioscheer@protonmail.com>` for both
  author and committer.** When the agent's local `git config user.*`
  resolves to anything else (e.g. `Claude <noreply@anthropic.com>` or
  `Codex <noreply@openai.com>`), pass the identity inline rather than
  modifying `.git/config`:
  ```
  GIT_COMMITTER_NAME="claudioscheer" \
  GIT_COMMITTER_EMAIL="claudioscheer@protonmail.com" \
  git commit --author "claudioscheer <claudioscheer@protonmail.com>" -m ...
  ```
  Verify after committing with `git log -1 --pretty='%an <%ae> | %cn <%ce>'`
  — both fields must read `claudioscheer <claudioscheer@protonmail.com>`.
- **Never add agent attribution to commit messages** — no
  `Co-Authored-By: Claude`, no `Co-Authored-By: Codex`, no
  `https://claude.ai/...` or `https://chat.openai.com/...` links. The
  AI-only nature of the project is documented in the README; it does not
  need to bleed into the commit log.
- **PRs are the human's call.** Open as draft when asked; don't merge
  without explicit permission.
- **Don't edit `participants/*.json`** — those belong to the upstream
  contest repo.

## Performance characteristics (for context, not contracts)

Historical baseline measured locally with `BenchmarkFraudScore_Real` against
the real 3 M dataset on a 2.8 GHz Xeon, GOMAXPROCS=1, `Nprobe=8`:

- ~285 µs / `FraudScore` call (~119× faster than the previous brute-force
  implementation).
- Per-replica RSS ~50 MB (mmap'd blob is the same size as before — IVF
  metadata is ~14 KB centroids + 4 KB offsets, dwarfed by the 42 MB
  reference vectors).

Compile cost: `cmd/compile-dataset` runs Lloyd's k-means with 15 iterations
across `runtime.GOMAXPROCS` workers. The production build currently uses
3 M / 1 024 clusters / 8-bit vectors. This is build-time only; runtime is
unaffected.

Tuning knobs:

- `SEARCH_NPROBE`, `SEARCH_MAX_NPROBE`, `SEARCH_ADAPTIVE` — runtime probe
  controls consumed by `cmd/api`. Higher probe counts raise recall and
  latency. Current production default is `12/24/true`.
- `cmd/compile-dataset -clusters` — production Docker build uses 1 024
  clusters; small fixtures still use `chooseNumClusters`.
- `cmd/compile-dataset -precision` — production currently uses 8; 16 is
  available for experiments but measured worse on the preview data.
