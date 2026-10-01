// Command server wires config → detector → HTTP and serves the NVR Vision API.
package main

import (
	"context"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"log/slog"
	"math"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"yolo-api/internal/config"
	"yolo-api/internal/detector"
	"yolo-api/internal/httpapi"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "fatal:", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := config.Load()
	if err != nil {
		return err
	}
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: cfg.LogLevel}))

	modelsDir := dedupe([]string{filepath.Dir(cfg.ModelPath), "/models_custom", "./models", "./models_custom"})
	srv := httpapi.New(cfg, shortModelName(cfg.ModelPath), log, modelsDir)

	// activeDet is closed after graceful shutdown once all handlers (and thus
	// in-flight Detect calls) have drained.
	var activeMu sync.Mutex
	var activeDet detector.Detector

	// Load the model in the background so /health answers immediately and
	// /ready reports 503 until the ORT pool is up.
	go func() {
		if err := ensureModel(cfg, log); err != nil {
			log.Error("model unavailable", "error", err)
			return
		}
		det, err := detector.New(cfg.ModelPath, cfg.ORTLibPath, cfg.ImgSize, cfg.PoolSize, cfg.IntraThreads, log)
		if err != nil {
			log.Error("detector failed", "error", err)
			return
		}
		log.Info("ort loaded", "version", det.Version())
		// Startup smoke test: one real inference so the logs prove end-to-end
		// detection works (and show what the model sees), not just library
		// load. /ready stays 503 until this succeeds.
		if err := smokeDetect(det, cfg, log); err != nil {
			log.Error("startup smoke detection failed — staying unready", "error", err)
			_ = det.Close()
			return
		}
		srv.SetDetector(det)
		activeMu.Lock()
		activeDet = det
		activeMu.Unlock()
		srv.MarkReady()
	}()

	httpSrv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      60 * time.Second,
		IdleTimeout:       120 * time.Second,
	}
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		log.Info("listening", "addr", cfg.Addr, "model", cfg.ModelPath)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			errCh <- err
		}
		close(errCh)
	}()

	select {
	case <-ctx.Done():
		log.Info("shutting down")
	case err := <-errCh:
		if err != nil {
			return fmt.Errorf("listen: %w", err)
		}
		return nil
	}
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	err = httpSrv.Shutdown(shutdownCtx)

	// In-flight handlers have drained; release the ORT sessions/runtime.
	activeMu.Lock()
	d := activeDet
	activeMu.Unlock()
	if d != nil {
		_ = d.Close()
	}
	return err
}

// smokeDetect runs a single inference at startup and logs every detected
// object (label, confidence, pixel box). It proves the full pipeline —
// image decode → ORT session → postprocess — works before /ready flips.
func smokeDetect(det detector.Detector, cfg *config.Config, log *slog.Logger) error {
	img, src := smokeImage(log)
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	dets, inferMs, err := det.Detect(ctx, img, float32(cfg.MinConf), nil)
	if err != nil {
		return err
	}
	b := img.Bounds()
	objs := make([]any, 0, len(dets))
	for _, d := range dets {
		objs = append(objs, map[string]any{
			"label":      d.Label,
			"confidence": math.Round(d.Confidence*10000) / 10000,
			"box":        d.Box,
		})
	}
	log.Info("startup smoke detection",
		"image", src,
		"size", fmt.Sprintf("%dx%d", b.Dx(), b.Dy()),
		"objects", len(dets),
		"inference_ms", math.Round(inferMs*10)/10,
		"detections", objs,
	)
	return nil
}

// smokeImage loads the bundled sample snapshot (/app/smoke.jpg in Docker,
// tests/testdata/bus.jpg for local runs), falling back to a synthetic
// gradient when neither exists (expect 0 objects in that case).
func smokeImage(log *slog.Logger) (image.Image, string) {
	for _, p := range []string{"/app/smoke.jpg", "tests/testdata/bus.jpg"} {
		if img, err := loadImageFile(p); err == nil {
			return img, p
		}
	}
	log.Warn("no sample snapshot found, smoke test uses a synthetic image")
	return syntheticImage(), "synthetic 640x640 gradient"
}

func loadImageFile(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, err
	}
	return img, nil
}

// syntheticImage builds a deterministic gradient frame for the smoke test
// when no sample snapshot is available.
func syntheticImage() image.Image {
	const w, h = 640, 640
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{uint8(x * 255 / w), uint8(y * 255 / h), 114, 255})
		}
	}
	return img
}

// ensureModel lazily downloads MODEL_URL to MODEL_PATH when the file is
// missing. Without MODEL_URL a missing file is a fatal startup error for the
// loader goroutine (server stays up, /ready stays 503).
func ensureModel(cfg *config.Config, log *slog.Logger) error {
	if _, err := os.Stat(cfg.ModelPath); err == nil {
		return nil
	}
	if cfg.ModelURL == "" {
		return fmt.Errorf("model file %q missing and MODEL_URL is empty", cfg.ModelPath)
	}
	log.Info("downloading model", "url", cfg.ModelURL, "dest", cfg.ModelPath)
	if err := os.MkdirAll(filepath.Dir(cfg.ModelPath), 0o755); err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Minute)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, cfg.ModelURL, nil)
	if err != nil {
		return err
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("model download: %s", resp.Status)
	}
	tmp := cfg.ModelPath + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	n, err := io.Copy(f, resp.Body)
	if err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	// Reject trivially small payloads (e.g. HTML error pages with status 200).
	if n < 1<<20 {
		_ = f.Close()
		_ = os.Remove(tmp)
		return fmt.Errorf("model download: file too small (%d bytes)", n)
	}
	if err := f.Sync(); err != nil {
		_ = f.Close()
		_ = os.Remove(tmp)
		return err
	}
	_ = f.Close()
	if err := os.Rename(tmp, cfg.ModelPath); err != nil {
		return err
	}
	log.Info("model downloaded", "bytes", n)
	return nil
}

func shortModelName(p string) string {
	base := filepath.Base(p)
	name := strings.TrimSuffix(base, filepath.Ext(base))
	return strings.TrimSuffix(name, "-e2e")
}

func dedupe(in []string) []string {
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}
