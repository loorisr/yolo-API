package httpapi

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"yolo-api/internal/detector"
	"yolo-api/internal/yolo"
)

// maxDimension caps a single image side, enforced before the full decode via
// image.DecodeConfig (decompression-bomb guard: bounded work, bounded memory).
const maxDimension = 4096

// maxPixels caps width*height (4096*4096), enforced alongside maxDimension so a
// single decode can never exceed ~64 MiB of RGBA pixels.
const maxPixels = 4096 * 4096

func writeJSON(w http.ResponseWriter, status int, v any) {
	buf := &bytes.Buffer{}
	if err := json.NewEncoder(buf).Encode(v); err != nil {
		// Never leave a 200 with an empty/partial body (e.g. non-finite floats).
		status = http.StatusInternalServerError
		buf = bytes.NewBufferString(`{"error":"response encoding failed"}`)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(buf.Bytes())
}

func writeError(w http.ResponseWriter, r *http.Request, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg, "request_id": requestID(r)})
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", "GET")
		writeError(w, r, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", "GET")
		writeError(w, r, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if !s.ready.Load() {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"ready": false})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"ready": true, "model": s.model})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", "GET")
		writeError(w, r, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	// Only the model name and size are exposed — absolute filesystem paths are
	// internal details and shouldn't leak to API clients.
	type entry struct {
		Name   string  `json:"name"`
		SizeMB float64 `json:"size_mb"`
	}
	seen := map[string]bool{}
	models := []entry{}
	add := func(path string) {
		abs, err := filepath.Abs(path)
		if err != nil {
			abs = path
		}
		if seen[abs] {
			return
		}
		seen[abs] = true
		st, err := os.Stat(abs)
		if err != nil || st.IsDir() {
			return
		}
		name := strings.TrimSuffix(filepath.Base(abs), filepath.Ext(abs))
		models = append(models, entry{Name: name, SizeMB: float64(st.Size()) / (1 << 20)})
	}
	for _, dir := range s.modelsDir {
		files, _ := filepath.Glob(filepath.Join(dir, "*.onnx"))
		for _, f := range files {
			add(f)
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"active": s.model, "models": models})
}

func (s *Server) handleDetect(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.Header().Set("Allow", "POST")
		writeError(w, r, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	if !s.ready.Load() {
		writeError(w, r, http.StatusServiceUnavailable, "model not ready")
		return
	}
	det := s.getDetector()
	if det == nil {
		writeError(w, r, http.StatusServiceUnavailable, "model not ready")
		return
	}
	minConf, err := queryMinConf(r, s.cfg.MinConf)
	if err != nil {
		writeError(w, r, http.StatusBadRequest, err.Error())
		return
	}
	classSet, err := queryClasses(r)
	if err != nil {
		writeError(w, r, http.StatusBadRequest, err.Error())
		return
	}
	raw, err := readImageBody(w, r, s.cfg.MaxBodyBytes())
	if err != nil {
		return // response already written
	}
	img, err := s.decodeImage(r.Context(), raw)
	if err != nil {
		if r.Context().Err() != nil {
			writeError(w, r, 499, "client closed request")
			return
		}
		writeError(w, r, http.StatusBadRequest, err.Error())
		return
	}
	b := img.Bounds()
	width, height := b.Dx(), b.Dy()
	dets, inferMs, err := det.Detect(r.Context(), img, float32(minConf), classSet)
	if err != nil {
		if errors.Is(err, detector.ErrPoolBusy) {
			w.Header().Set("Retry-After", "5")
			writeError(w, r, http.StatusServiceUnavailable, "server busy, try again")
			return
		}
		if r.Context().Err() != nil {
			writeError(w, r, 499, "client closed request")
			return
		}
		s.log.Error("detect failed", "request_id", requestID(r), "error", err)
		writeError(w, r, http.StatusInternalServerError, "inference failed")
		return
	}
	if dets == nil {
		dets = []detector.Detection{}
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"model": s.model, "width": width, "height": height,
		"inference_ms": inferMs, "request_id": requestID(r), "detections": dets,
	})
}

func queryMinConf(r *http.Request, def float64) (float64, error) {
	q := r.URL.Query().Get("min_conf")
	if q == "" {
		return def, nil
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(q), 64)
	if err != nil || math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || v > 1 {
		return 0, fmt.Errorf("invalid min_conf %q (must be 0..1)", q)
	}
	return v, nil
}

// queryClasses parses `classes=person,car,bicycle` into a set of class IDs.
// Empty = all classes. Unknown labels are a 400.
func queryClasses(r *http.Request) (map[int]bool, error) {
	q := r.URL.Query().Get("classes")
	if strings.TrimSpace(q) == "" {
		return nil, nil
	}
	set := map[int]bool{}
	for _, part := range strings.Split(q, ",") {
		name := strings.ToLower(strings.TrimSpace(part))
		if name == "" {
			continue
		}
		id, ok := yolo.NameToID[name]
		if !ok {
			return nil, fmt.Errorf("unknown class %q", part)
		}
		set[id] = true
	}
	if len(set) == 0 {
		return nil, fmt.Errorf("classes filter %q resolves to no classes", q)
	}
	return set, nil
}

// readImageBody reads the snapshot from multipart field `image` or raw
// image/jpeg|image/png bytes. It enforces the body cap (413) and writes the
// error response itself, returning err != nil in that case. The multipart path
// streams via MultipartReader so no temp files are ever created.
func readImageBody(w http.ResponseWriter, r *http.Request, maxBytes int64) ([]byte, error) {
	r.Body = http.MaxBytesReader(w, r.Body, maxBytes)
	ct := r.Header.Get("Content-Type")
	media := strings.ToLower(strings.TrimSpace(strings.SplitN(ct, ";", 2)[0]))
	if media == "multipart/form-data" {
		mr, err := r.MultipartReader()
		if err != nil {
			if isTooLarge(err) {
				writeError(w, r, http.StatusRequestEntityTooLarge, "body too large")
			} else {
				writeError(w, r, http.StatusBadRequest, "invalid multipart body")
			}
			return nil, err
		}
		for {
			part, err := mr.NextPart()
			if err != nil {
				if errors.Is(err, io.EOF) {
					break
				}
				if isTooLarge(err) {
					writeError(w, r, http.StatusRequestEntityTooLarge, "body too large")
				} else {
					writeError(w, r, http.StatusBadRequest, "invalid multipart body")
				}
				return nil, err
			}
			if part.FormName() != "image" {
				_, _ = io.Copy(io.Discard, io.LimitReader(part, maxBytes+1))
				continue
			}
			raw, err := io.ReadAll(io.LimitReader(part, maxBytes+1))
			if err != nil {
				if isTooLarge(err) {
					writeError(w, r, http.StatusRequestEntityTooLarge, "body too large")
				} else {
					writeError(w, r, http.StatusBadRequest, "cannot read image")
				}
				return nil, err
			}
			if int64(len(raw)) > maxBytes {
				writeError(w, r, http.StatusRequestEntityTooLarge, "body too large")
				return nil, fmt.Errorf("too large")
			}
			if len(raw) == 0 {
				writeError(w, r, http.StatusBadRequest, "empty image")
				return nil, fmt.Errorf("empty")
			}
			return raw, nil
		}
		writeError(w, r, http.StatusBadRequest, "multipart field \"image\" is required")
		return nil, fmt.Errorf("missing image field")
	}
	switch media {
	case "image/jpeg", "image/jpg", "image/png":
		// ok
	default:
		writeError(w, r, http.StatusBadRequest, "unsupported content type (use multipart field \"image\" or raw image/jpeg|image/png)")
		return nil, fmt.Errorf("bad content type %q", ct)
	}
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		if isTooLarge(err) {
			writeError(w, r, http.StatusRequestEntityTooLarge, "body too large")
		} else {
			writeError(w, r, http.StatusBadRequest, "cannot read image")
		}
		return nil, err
	}
	if len(raw) == 0 {
		writeError(w, r, http.StatusBadRequest, "empty image")
		return nil, fmt.Errorf("empty")
	}
	return raw, nil
}

func isTooLarge(err error) bool {
	var mbe *http.MaxBytesError
	return errors.As(err, &mbe)
}

// decodeImage validates dimensions with image.DecodeConfig (header-only) before
// the full decode, so decompression bombs are rejected without ever allocating
// the full pixel buffer. Concurrent full decodes are bounded to cap peak memory.
// The semaphore wait respects ctx cancellation so a disconnected client cannot
// hold the gate.
func (s *Server) decodeImage(ctx context.Context, raw []byte) (image.Image, error) {
	cfg, _, err := image.DecodeConfig(bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("bad image (jpeg/png required): %w", err)
	}
	if cfg.Width <= 0 || cfg.Height <= 0 || cfg.Width > maxDimension || cfg.Height > maxDimension {
		return nil, fmt.Errorf("image dimensions %dx%d exceed limits (max %dx%d)",
			cfg.Width, cfg.Height, maxDimension, maxDimension)
	}
	if int64(cfg.Width)*int64(cfg.Height) > maxPixels {
		return nil, fmt.Errorf("image dimensions %dx%d exceed limits (max %dx%d)",
			cfg.Width, cfg.Height, maxDimension, maxDimension)
	}
	select {
	case s.decodeSem <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	defer func() { <-s.decodeSem }()
	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return nil, fmt.Errorf("bad image (jpeg/png required): %w", err)
	}
	return img, nil
}
