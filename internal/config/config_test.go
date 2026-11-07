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
	if want := max(1, runtime.NumCPU()/c.PoolSize); c.IntraThreads != want {
		t.Fatalf("INTRA_THREADS=0 should mean NumCPU/POOL_SIZE=%d, got %d", want, c.IntraThreads)
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

func TestIntraThreadsSplitsCores(t *testing.T) {
	t.Setenv("POOL_SIZE", "2")
	c, err := Load()
	if err != nil {
		t.Fatal(err)
	}
	if want := max(1, runtime.NumCPU()/2); c.IntraThreads != want {
		t.Fatalf("want INTRA_THREADS=%d with POOL_SIZE=2, got %d", want, c.IntraThreads)
	}
	t.Setenv("INTRA_THREADS", "3")
	if c, err = Load(); err != nil || c.IntraThreads != 3 {
		t.Fatalf("explicit INTRA_THREADS=3 not honored: %+v, %v", c, err)
	}
}
