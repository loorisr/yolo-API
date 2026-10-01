// Integration test against the real model. Requires:
//
//	ORT_LIB_PATH=/path/to/libonnxruntime.so
//	MODEL_PATH=models/yolo26n.onnx
//
// Skipped automatically when the files are absent, so `go test ./...`
// stays green on machines without the native lib.
package detector

import (
	"context"
	"image/jpeg"
	"log/slog"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestORTDetectBus(t *testing.T) {
	lib := os.Getenv("ORT_LIB_PATH")
	model := os.Getenv("MODEL_PATH")
	if lib == "" || model == "" {
		t.Skip("ORT_LIB_PATH and MODEL_PATH must be set")
	}
	// Tests run with the package dir as cwd; resolve a repo-relative
	// MODEL_PATH (e.g. models/yolo26n.onnx) against the repo root.
	if !filepath.IsAbs(model) {
		if _, err := os.Stat(model); err != nil {
			if _, err2 := os.Stat(filepath.Join("..", "..", model)); err2 == nil {
				model = filepath.Join("..", "..", model)
			}
		}
	}
	if _, err := os.Stat(lib); err != nil {
		t.Skipf("missing ORT lib: %v", err)
	}
	imgPath := filepath.Join("..", "..", "tests", "testdata", "bus.jpg")
	if _, err := os.Stat(imgPath); err != nil {
		alt := filepath.Join("tests", "testdata", "bus.jpg")
		if _, err2 := os.Stat(alt); err2 == nil {
			imgPath = alt
		} else {
			t.Skipf("missing test image: %v", err)
		}
	}
	f, err := os.Open(imgPath)
	if err != nil {
		t.Fatal(err)
	}
	img, err := jpeg.Decode(f)
	_ = f.Close()
	if err != nil {
		t.Fatal(err)
	}
	log := slog.New(slog.DiscardHandler)
	det, err := New(model, lib, 640, 1, 0, log)
	if err != nil {
		t.Fatal(err)
	}
	defer det.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	dets, inferMs, err := det.Detect(ctx, img, 0.25, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("inference_ms=%.1f detections=%d", inferMs, len(dets))
	found := map[string]int{}
	for _, d := range dets {
		found[d.Label]++
		if d.Confidence < 0.25 {
			t.Errorf("detection below min_conf: %+v", d)
		}
		b := img.Bounds()
		if d.Box.XMax <= d.Box.XMin || d.Box.YMax <= d.Box.YMin {
			t.Errorf("degenerate box: %+v", d)
		}
		if d.Box.XMin < 0 || d.Box.YMin < 0 || d.Box.XMax > float64(b.Dx()) || d.Box.YMax > float64(b.Dy()) {
			t.Errorf("box outside image: %+v", d)
		}
	}
	if found["bus"] < 1 {
		t.Errorf("want ≥1 bus, got %v", found)
	}
	if found["person"] < 3 {
		t.Errorf("want ≥3 persons, got %v", found)
	}
}
