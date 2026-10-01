# syntax=docker/dockerfile:1
# NVR Vision API — multi-stage build (SPEC §7).
# Bakes ONLY yolo26n; mount yolo26s via ./models_custom + MODEL_PATH.
# ORT comes from Alpine (musl-native .so, no glibc shim needed).
# Stable (3.24) only carries ORT 1.24.x, so the runtime rides edge to track
# latest; no version pin — apk resolves the newest available at build time.

FROM golang:1.27-alpine AS builder
RUN apk add --no-cache ca-certificates
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -o /out/server ./cmd/server

FROM alpine:edge AS runtime
RUN apk add --no-cache \
      onnxruntime \
      ca-certificates wget libstdc++ libgomp \
 && adduser -D -u 10001 app \
 && mkdir -p /models /app && chown app:app /models /app
COPY --from=builder /out/server /app/server
# Bake ONLY the n model (keeps the image small); mount yolo26s separately.
COPY models/yolo26n.onnx /models/yolo26n.onnx
# Sample snapshot for the startup smoke detection (see smokeDetect).
COPY tests/testdata/bus.jpg /app/smoke.jpg
RUN chown app:app /app/server /models/yolo26n.onnx /app/smoke.jpg
USER app
ENV ADDR=:8080 \
    MODEL_PATH=/models/yolo26n.onnx \
    ORT_LIB_PATH=/usr/lib/libonnxruntime.so.1 \
    IMG_SIZE=640 \
    MIN_CONF=0.25 \
    INTRA_THREADS=4 \
    POOL_SIZE=1 \
    MAX_BODY_MB=8 \
    LOG_LEVEL=info
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8080/health | grep -q '"status":"ok"'
ENTRYPOINT ["/app/server"]
