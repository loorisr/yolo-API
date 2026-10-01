package httpapi

import (
	"bytes"
	"context"
	"encoding/binary"
	"encoding/json"
	"hash/crc32"
	"image"
	"image/png"
	"log/slog"
	"math"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"yolo-api/internal/config"
	"yolo-api/internal/detector"
)

type mockDetector struct {
	dets []detector.Detection
	err  error
}

func (m *mockDetector) Detect(context.Context, image.Image, float32, map[int]bool) ([]detector.Detection, float64, error) {
	return m.dets, 3.5, m.err
}
func (m *mockDetector) Close() error { return nil }

// testImage returns a minimal 8x8 PNG.
func testImage(t *testing.T) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 8, 8))
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	return buf.Bytes()
}

func testServer(apiKey string, det detector.Detector) (*Server, *config.Config) {
	cfg := &config.Config{Addr: ":0", ModelPath: "/models/yolo26n.onnx", ImgSize: 640,
		MinConf: 0.25, PoolSize: 1, APIKey: apiKey, MaxBodyMB: 8}
	log := slog.New(slog.DiscardHandler)
	s := New(cfg, "yolo26n", log, []string{filepath.Dir(cfg.ModelPath)})
	if det != nil {
		s.SetDetector(det)
		s.MarkReady()
	}
	return s, cfg
}

func TestHealthOpen(t *testing.T) {
	s, _ := testServer("secret", nil)
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status=%d", rec.Code)
	}
	var body map[string]string
	_ = json.Unmarshal(rec.Body.Bytes(), &body)
	if body["status"] != "ok" {
		t.Fatalf("body=%v", body)
	}
}

func TestReady503Then200(t *testing.T) {
	s, _ := testServer("", nil)
	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 503 {
		t.Fatalf("want 503 before ready, got %d", rec.Code)
	}
	s.SetDetector(&mockDetector{})
	s.MarkReady()
	rec = httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("want 200 after ready, got %d", rec.Code)
	}
}

func TestDetect503WhenNotReady(t *testing.T) {
	s, _ := testServer("", nil)
	req := httptest.NewRequest(http.MethodPost, "/v1/detect", strings.NewReader("x"))
	req.Header.Set("Content-Type", "image/jpeg")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 503 {
		t.Fatalf("want 503, got %d", rec.Code)
	}
}

func multipartBody(t *testing.T, field, name string, content []byte) (*bytes.Buffer, string) {
	t.Helper()
	var buf bytes.Buffer
	w := multipart.NewWriter(&buf)
	fw, err := w.CreateFormFile(field, name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fw.Write(content); err != nil {
		t.Fatal(err)
	}
	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	return &buf, w.FormDataContentType()
}

func TestDetectMultipartOK(t *testing.T) {
	mock := &mockDetector{dets: []detector.Detection{
		{ClassID: 0, Label: "person", Confidence: 0.9,
			Box: detector.Box{XMin: 0, YMin: 0, XMax: 4, YMax: 8}, BoxNorm: detector.Box{XMin: 0, YMin: 0, XMax: 0.5, YMax: 1}},
	}}
	s, _ := testServer("", mock)
	body, ct := multipartBody(t, "image", "snap.png", testImage(t))
	req := httptest.NewRequest(http.MethodPost, "/v1/detect?min_conf=0.25", body)
	req.Header.Set("Content-Type", ct)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var resp map[string]any
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatal(err)
	}
	if resp["model"] != "yolo26n" || resp["request_id"] == "" {
		t.Fatalf("bad response: %v", resp)
	}
	dets := resp["detections"].([]any)
	if len(dets) != 1 {
		t.Fatalf("want 1 det, got %v", resp)
	}
}

func TestDetectRawBytesOK(t *testing.T) {
	s, _ := testServer("", &mockDetector{})
	req := httptest.NewRequest(http.MethodPost, "/v1/detect", bytes.NewReader(testImage(t)))
	req.Header.Set("Content-Type", "image/png")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
}

func TestAuthEnforced(t *testing.T) {
	s, _ := testServer("secret", &mockDetector{})
	do := func(key string) int {
		body, ct := multipartBody(t, "image", "a.png", testImage(t))
		req := httptest.NewRequest(http.MethodPost, "/v1/detect", body)
		req.Header.Set("Content-Type", ct)
		if key != "" {
			req.Header.Set("X-API-Key", key)
		}
		rec := httptest.NewRecorder()
		s.Handler().ServeHTTP(rec, req)
		return rec.Code
	}
	if do("") != 401 || do("wrong") != 401 || do("secret") != 200 {
		t.Fatal("auth matrix failed")
	}
}

func TestDetectBadInputs(t *testing.T) {
	s, _ := testServer("", &mockDetector{})
	cases := []struct {
		name string
		url  string
		ct   string
		body []byte
		want int
	}{
		{"unknown class", "/v1/detect?classes=nope", "image/png", testImage(t), 400},
		{"bad min_conf", "/v1/detect?min_conf=2", "image/png", testImage(t), 400},
		{"nan min_conf", "/v1/detect?min_conf=NaN", "image/png", testImage(t), 400},
		{"inf min_conf", "/v1/detect?min_conf=Inf", "image/png", testImage(t), 400},
		{"empty classes filter", "/v1/detect?classes=,", "image/png", testImage(t), 400},
		{"bad bytes", "/v1/detect", "image/jpeg", []byte("not an image"), 400},
		{"bad content type", "/v1/detect", "application/json", []byte("{}"), 400},
		{"missing field", "/v1/detect", "multipart", nil, 400},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			var body *bytes.Buffer
			ct := c.ct
			if c.ct == "multipart" {
				var buf bytes.Buffer
				w := multipart.NewWriter(&buf)
				_ = w.WriteField("other", "x")
				_ = w.Close()
				body = &buf
				ct = w.FormDataContentType()
			} else {
				body = bytes.NewBuffer(c.body)
			}
			req := httptest.NewRequest(http.MethodPost, c.url, body)
			req.Header.Set("Content-Type", ct)
			rec := httptest.NewRecorder()
			s.Handler().ServeHTTP(rec, req)
			if rec.Code != c.want {
				t.Fatalf("want %d, got %d (%s)", c.want, rec.Code, rec.Body.String())
			}
			var errBody map[string]string
			_ = json.Unmarshal(rec.Body.Bytes(), &errBody)
			if errBody["error"] == "" || errBody["request_id"] == "" {
				t.Fatalf("error shape broken: %s", rec.Body.String())
			}
		})
	}
}

func TestModelsListsDir(t *testing.T) {
	dir := t.TempDir()
	for _, n := range []string{"a.onnx", "b.onnx"} {
		if err := os.WriteFile(filepath.Join(dir, n), []byte("x"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	cfg := &config.Config{ModelPath: filepath.Join(dir, "a.onnx"), MaxBodyMB: 8}
	s := New(cfg, "a", slog.New(slog.DiscardHandler), []string{dir})
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != 200 {
		t.Fatalf("status=%d", rec.Code)
	}
	var resp struct {
		Active string `json:"active"`
		Models []struct {
			Name string `json:"name"`
		} `json:"models"`
	}
	_ = json.Unmarshal(rec.Body.Bytes(), &resp)
	if resp.Active != "a" || len(resp.Models) != 2 {
		t.Fatalf("bad models response: %+v", resp)
	}
	if strings.Contains(rec.Body.String(), dir) {
		t.Fatalf("models response leaked an absolute path: %s", rec.Body.String())
	}
}

// pngWithDims builds a minimal PNG whose IHDR claims the given dimensions.
// Only the header is present; image.DecodeConfig reads it without decoding
// pixels, which is exactly the path the decompression-bomb guard exercises.
func pngWithDims(width, height uint32) []byte {
	var buf bytes.Buffer
	buf.Write([]byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'})
	ihdr := make([]byte, 13)
	binary.BigEndian.PutUint32(ihdr[0:4], width)
	binary.BigEndian.PutUint32(ihdr[4:8], height)
	ihdr[8] = 8 // bit depth
	ihdr[9] = 6 // RGBA
	writeChunk(&buf, "IHDR", ihdr)
	return buf.Bytes()
}

func writeChunk(buf *bytes.Buffer, typ string, data []byte) {
	_ = binary.Write(buf, binary.BigEndian, uint32(len(data)))
	buf.WriteString(typ)
	buf.Write(data)
	h := crc32.NewIEEE()
	_, _ = h.Write([]byte(typ))
	_, _ = h.Write(data)
	_ = binary.Write(buf, binary.BigEndian, h.Sum32())
}

// TestDetectDecompressionBombRejected verifies oversized images are refused on
// the header alone (image.DecodeConfig) rather than after a full decode.
func TestDetectDecompressionBombRejected(t *testing.T) {
	s, _ := testServer("", &mockDetector{})
	req := httptest.NewRequest(http.MethodPost, "/v1/detect", bytes.NewReader(pngWithDims(20000, 20000)))
	req.Header.Set("Content-Type", "image/png")
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("want 400 for oversized image, got %d body=%s", rec.Code, rec.Body.String())
	}
}

// TestDetectNaNConfidenceKeepsValidJSON ensures a non-finite value can't produce
// a 200 with an empty/invalid JSON body.
func TestDetectNaNConfidenceKeepsValidJSON(t *testing.T) {
	mock := &mockDetector{dets: []detector.Detection{
		{ClassID: 0, Label: "person", Confidence: math.NaN()},
	}}
	s, _ := testServer("", mock)
	body, ct := multipartBody(t, "image", "a.png", testImage(t))
	req := httptest.NewRequest(http.MethodPost, "/v1/detect", body)
	req.Header.Set("Content-Type", ct)
	rec := httptest.NewRecorder()
	s.Handler().ServeHTTP(rec, req)
	if rec.Body.Len() == 0 {
		t.Fatal("empty response body")
	}
	var v any
	if err := json.Unmarshal(rec.Body.Bytes(), &v); err != nil {
		t.Fatalf("response is not valid JSON: %v body=%q", err, rec.Body.String())
	}
}
