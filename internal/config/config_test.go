package config

import (
	"runtime"
	"testing"
)

func TestDefaults(t *testing.T) {
	c, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if c.Addr != ":8080" || c.ImgSize != 640 || c.MinConf != 0.25 || c.MaxBodyMB != 8 {
		t.Fatalf("bad defaults: %+v", c)
	}
	if c.PoolSize != runtime.NumCPU() {
		t.Fatalf("POOL_SIZE=0 should mean NumCPU=%d, got %d", runtime.NumCPU(), c.PoolSize)
	}
	if c.MaxBodyBytes() != 8<<20 {
		t.Fatalf("bad MaxBodyBytes: %d", c.MaxBodyBytes())
	}
}

func TestInvalidMinConf(t *testing.T) {
	t.Setenv("MIN_CONF", "1.5")
	if _, err := Load(); err == nil {
		t.Fatal("want error for MIN_CONF=1.5")
	}
	t.Setenv("MIN_CONF", "NaN")
	if _, err := Load(); err == nil {
		t.Fatal("want error for MIN_CONF=NaN")
	}
}

func TestInvalidImgSize(t *testing.T) {
	t.Setenv("IMG_SIZE", "0")
	if _, err := Load(); err == nil {
		t.Fatal("want error for IMG_SIZE=0")
	}
}

func TestExplicitPoolSize(t *testing.T) {
	t.Setenv("POOL_SIZE", "3")
	c, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if c.PoolSize != 3 {
		t.Fatalf("want POOL_SIZE=3, got %d", c.PoolSize)
	}
}
