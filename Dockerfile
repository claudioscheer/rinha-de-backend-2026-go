FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -ldflags="-s -w" -o /out/api ./cmd/api && \
    CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -ldflags="-s -w" -o /out/compile-dataset ./cmd/compile-dataset

# Pre-compile the reference dataset into the compact binary blob the API
# mmaps at startup. Doing this at build time avoids the ~300 MB peak from
# JSON-decoding 3 M records inside a 160 MB container.
COPY resources /src/resources
RUN /out/compile-dataset \
        -in /src/resources/references.json.gz \
        -out /src/resources/references.bin \
        -precision 8 \
        -clusters 1024 && \
    rm /src/resources/references.json.gz

FROM alpine:3.20
RUN adduser -D -u 10001 app
WORKDIR /app
COPY --from=build /out/api /app/api
COPY --from=build /src/resources /app/resources
USER app
EXPOSE 9999
ENV RESOURCES_DIR=/app/resources \
    LISTEN_ADDR=:9999 \
    GOGC=50 \
    GOMEMLIMIT=120MiB \
    GOMAXPROCS=1 \
    SEARCH_NPROBE=12 \
    SEARCH_MAX_NPROBE=24 \
    SEARCH_ADAPTIVE=true
ENTRYPOINT ["/app/api"]
