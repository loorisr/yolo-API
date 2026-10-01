package config

import (
	"fmt"
	"log/slog"
	"math"
	"os"
	"runtime"
	"strconv"
	"strings"
)

// Config holds all server configuration, parsed from the environment.
// See SPEC.md §6 and docker-compose.yml.
type Config struct {
	Addr         string
	ModelPath    string
	ModelURL     string
	ORTLibPath   string
	ImgSize      int
	MinConf      float64
	IntraThreads int
	PoolSize     int
	APIKey       string
	MaxBodyMB    int
	LogLevel     slog.Level
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func getint(key string, def int) (int, error) {
	v := getenv(key, "")
	if v == "" {
		return def, nil
	}
	n, err := strconv.Atoi(strings.TrimSpace(v))
	if err != nil {
		return 0, fmt.Errorf("invalid %s=%q: %w", key, v, err)
	}
	return n, nil
}

func getfloat(key string, def float64) (float64, error) {
	v := getenv(key, "")
	if v == "" {
		return def, nil
	}
	f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
	if err != nil {
		return 0, fmt.Errorf("invalid %s=%q: %w", key, v, err)
	}
	return f, nil
}

// Load parses the environment into a Config, applying SPEC §6 defaults.
func Load() (*Config, error) {
	c := &Config{
		Addr:       getenv("ADDR", ":8080"),
		ModelPath:  getenv("MODEL_PATH", "/models/yolo26n.onnx"),
		ModelURL:   getenv("MODEL_URL", ""),
		ORTLibPath: getenv("ORT_LIB_PATH", "/usr/lib/libonnxruntime.so.1"),
		APIKey:     getenv("API_KEY", ""),
	}
	var err error
	if c.ImgSize, err = getint("IMG_SIZE", 640); err != nil {
		return nil, err
	}
	if c.MinConf, err = getfloat("MIN_CONF", 0.25); err != nil {
		return nil, err
	}
	if c.IntraThreads, err = getint("INTRA_THREADS", 0); err != nil {
		return nil, err
	}
	if c.PoolSize, err = getint("POOL_SIZE", 0); err != nil {
		return nil, err
	}
	if c.MaxBodyMB, err = getint("MAX_BODY_MB", 8); err != nil {
		return nil, err
	}
	if c.PoolSize <= 0 {
		c.PoolSize = runtime.NumCPU()
		if c.PoolSize > 64 {
			c.PoolSize = 64
		}
		if c.PoolSize < 1 {
			c.PoolSize = 1
		}
	}
	switch lvl := strings.ToLower(strings.TrimSpace(getenv("LOG_LEVEL", "info"))); lvl {
	case "", "info":
		c.LogLevel = slog.LevelInfo
	case "debug":
		c.LogLevel = slog.LevelDebug
	case "warn", "warning":
		c.LogLevel = slog.LevelWarn
	case "error":
		c.LogLevel = slog.LevelError
	default:
		return nil, fmt.Errorf("invalid LOG_LEVEL=%q (must be debug/info/warn/error)", getenv("LOG_LEVEL", ""))
	}

	if c.ImgSize <= 0 || c.ImgSize > 2048 {
		return nil, fmt.Errorf("invalid IMG_SIZE=%d (must be 1..2048)", c.ImgSize)
	}
	if c.MinConf < 0 || c.MinConf > 1 || math.IsNaN(c.MinConf) || math.IsInf(c.MinConf, 0) {
		return nil, fmt.Errorf("invalid MIN_CONF=%v (must be 0..1)", c.MinConf)
	}
	if c.MaxBodyMB <= 0 || c.MaxBodyMB > 64 {
		return nil, fmt.Errorf("invalid MAX_BODY_MB=%d (must be 1..64)", c.MaxBodyMB)
	}
	if c.IntraThreads < 0 {
		return nil, fmt.Errorf("invalid INTRA_THREADS=%d (must be >= 0)", c.IntraThreads)
	}
	if c.PoolSize <= 0 || c.PoolSize > 64 {
		return nil, fmt.Errorf("invalid POOL_SIZE=%d (must be 1..64)", c.PoolSize)
	}
	return c, nil
}

// MaxBodyBytes returns the max request body size in bytes.
func (c *Config) MaxBodyBytes() int64 { return int64(c.MaxBodyMB) << 20 }
