FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod ./
RUN go mod download
COPY cmd ./cmd
COPY internal ./internal
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -trimpath -ldflags="-s -w" -o /out/api ./cmd/api

FROM alpine:3.20
RUN adduser -D -u 10001 app
WORKDIR /app
COPY --from=build /out/api /app/api
COPY resources /app/resources
USER app
EXPOSE 9999
ENV RESOURCES_DIR=/app/resources LISTEN_ADDR=:9999
ENTRYPOINT ["/app/api"]
