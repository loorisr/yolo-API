package httpapi

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"log/slog"
	"net/http"
	"runtime/debug"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"yolo-api/internal/config"
	"yolo-api/internal/detector"
)

type ctxKey int

const requestIDKey ctxKey = iota

// Server wires config, detector readiness and stdlib routes.
type Server struct {
	cfg       *config.Config
	detMu     sync.RWMutex
	det       detector.Detector
	model     string
	ready     atomic.Bool
	log       *slog.Logger
	mux       *http.ServeMux
	modelsDir []string
	decodeSem chan struct{}
}

// decodeConcurrency bounds how many full image decodes run at once, to cap
// peak memory across concurrent requests (decoding happens outside the ORT pool).
func decodeConcurrency(poolSize int) int {
	n := poolSize
	if n < 1 {
		n = 1
	}
	if n > 8 {
		n = 8
	}
	return 2 * n
}

// New builds the route table. SetDetector + MarkReady flip /ready to 200
// once the ORT session pool is loaded (model loads in the background so
// /health answers immediately).
func New(cfg *config.Config, model string, log *slog.Logger, modelsDir []string) *Server {
	s := &Server{cfg: cfg, model: model, log: log, mux: http.NewServeMux(), modelsDir: modelsDir,
		decodeSem: make(chan struct{}, decodeConcurrency(cfg.PoolSize))}
	s.mux.HandleFunc("/health", s.handleHealth)
	s.mux.HandleFunc("/ready", s.handleReady)
	s.mux.HandleFunc("/v1/models", s.requireAuth(s.handleModels))
	s.mux.HandleFunc("/v1/detect", s.requireAuth(s.handleDetect))
	return s
}

// SetDetector installs the loaded detector.
func (s *Server) SetDetector(d detector.Detector) {
	s.detMu.Lock()
	defer s.detMu.Unlock()
	s.det = d
}

func (s *Server) getDetector() detector.Detector {
	s.detMu.RLock()
	defer s.detMu.RUnlock()
	return s.det
}

// MarkReady signals that the model is loaded.
func (s *Server) MarkReady() { s.ready.Store(true) }

// Handler returns the full middleware chain.
func (s *Server) Handler() http.Handler {
	return s.withRequestID(s.recover(s.mux))
}

func requestID(r *http.Request) string {
	if v, ok := r.Context().Value(requestIDKey).(string); ok {
		return v
	}
	return ""
}

func newRequestID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return time.Now().UTC().Format("20060102150405.000000000")
	}
	return hex.EncodeToString(b[:])
}

func (s *Server) withRequestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := newRequestID()
		w.Header().Set("X-Request-ID", id)
		ctx := context.WithValue(r.Context(), requestIDKey, id)
		t0 := time.Now()
		rec := &statusRecorder{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rec, r.WithContext(ctx))
		lvl := slog.LevelInfo
		if r.URL.Path == "/health" || r.URL.Path == "/ready" {
			lvl = slog.LevelDebug // probes would otherwise flood the logs
		}
		s.log.Log(context.Background(), lvl, "request",
			"request_id", id, "method", r.Method, "path", r.URL.Path,
			"status", rec.status, "duration_ms", float64(time.Since(t0).Microseconds())/1000.0)
	})
}

// statusRecorder captures the status code for access logs and guards against a
// superfluous second WriteHeader (e.g. the panic handler after a partial write).
type statusRecorder struct {
	http.ResponseWriter
	status      int
	wroteHeader bool
}

func (rec *statusRecorder) WriteHeader(code int) {
	if rec.wroteHeader {
		return
	}
	rec.wroteHeader = true
	rec.status = code
	rec.ResponseWriter.WriteHeader(code)
}

func (rec *statusRecorder) Write(b []byte) (int, error) {
	if !rec.wroteHeader {
		rec.WriteHeader(http.StatusOK)
	}
	return rec.ResponseWriter.Write(b)
}

func (s *Server) recover(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				s.log.Error("panic", "request_id", requestID(r), "panic", rec, "stack", string(debug.Stack()))
				writeError(w, r, http.StatusInternalServerError, "internal error")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// requireAuth enforces API_KEY on /v1/* when set. Empty key = open (LAN).
// Accepts X-API-Key or Authorization: Bearer, compared in constant time.
// /health and /ready never require auth (registered without this wrapper).
func (s *Server) requireAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.cfg.APIKey == "" {
			next(w, r)
			return
		}
		got := r.Header.Get("X-API-Key")
		if got == "" {
			if auth := r.Header.Get("Authorization"); strings.HasPrefix(strings.ToLower(auth), "bearer ") {
				got = strings.TrimSpace(auth[7:])
			}
		}
		if got == "" || subtle.ConstantTimeCompare([]byte(got), []byte(s.cfg.APIKey)) != 1 {
			writeError(w, r, http.StatusUnauthorized, "missing or invalid API key")
			return
		}
		next(w, r)
	}
}
