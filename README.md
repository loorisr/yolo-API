# NVR Vision API (YOLO26)

HTTP API for NVR software: send a snapshot (JPEG/PNG), get detected objects
with probabilities and boxes. CPU-only, no CGO, Docker deploy. See `API.md`
and `openapi.yaml` for the contract.

## Quickstart (local)

Requires Go ≥1.27 and an ONNX Runtime `.so`:

```bash
ADDR=:8080 MODEL_PATH=$PWD/models/yolo26n.onnx \
  ORT_LIB_PATH=/usr/lib/libonnxruntime.so.1 \
  POOL_SIZE=2 \
  go run ./cmd/server
curl localhost:8080/health
curl -F image=@tests/testdata/bus.jpg "localhost:8080/v1/detect?min_conf=0.3&classes=person,car,bicycle"
```

Raw bytes + API key:

```bash
curl -H "X-API-Key: $API_KEY" -H "Content-Type: image/jpeg" \
  --data-binary @snap.jpg "localhost:8080/v1/detect?min_conf=0.25"
```

## Docker

```bash
docker build -t yolo-api .
docker run -p 8080:8080 yolo-api
# yolo26s via mount (never mount over /models — baked n lives there):
docker run -p 8080:8080 -v $PWD/models_custom:/models_custom:ro \
  -e MODEL_PATH=/models_custom/yolo26s.onnx yolo-api
```

Or `docker compose up --build` (defaults are inlined in `docker-compose.yml`;
override via shell env, e.g. `API_KEY=secret docker compose up --build`).

## Notes

- The bundled `models/yolo26*.onnx` are **raw** exports `(1,84,8400)`; the
  server decodes them with per-class NMS (IoU 0.45) in Go. End-to-end
  `(1,300,6)` models are auto-detected — see `scripts/export-onnx.py`
  (`uv run scripts/export-onnx.py --model yolo26n.pt`).
- Binding is `shota3506/onnxruntime-purego` behind `internal/detector`
  (the primary raven engine requires e2e output).
  Swap it without touching `internal/httpapi`.
- Alpine-edge runtime with musl-native ORT `.so` from apk
  (`apk add onnxruntime`, no glibc shim); the server smoke-tests `dlopen`
  at startup (`/ready` stays 503 on failure).
- Tests: `go test ./...`; integration on `tests/testdata/bus.jpg` needs
  `ORT_LIB_PATH` + `MODEL_PATH` set.
- Tuning: `POOL_SIZE × INTRA_THREADS ≈ NumCPU` (e.g. `POOL_SIZE=2`,
  `INTRA_THREADS=4` on 8 cores). Measured on this box (loaded):
  `n` ~23 ms, `s` ~60 ms per snapshot.
