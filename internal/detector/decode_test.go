package detector

import (
	"image"
	"image/color"
	"testing"
)

func solidImage(w, h int) image.Image {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{uint8(x % 256), uint8(y % 256), 128, 255})
		}
	}
	return img
}

func TestLetterboxGeometry(t *testing.T) {
	// 810x1080 portrait into 640: scale = 640/1080, padX = (640-480)/2.
	data, scale, padX, padY := letterbox(solidImage(810, 1080), 640)
	if len(data) != 3*640*640 {
		t.Fatalf("bad tensor len %d", len(data))
	}
	wantScale := 640.0 / 1080.0
	if diff := scale - wantScale; diff > 1e-9 || diff < -1e-9 {
		t.Fatalf("scale=%v want %v", scale, wantScale)
	}
	if padX != 80 || padY != 0 {
		t.Fatalf("pad=%v,%v want 80,0", padX, padY)
	}
	// Round-trip: box covering the full letterboxed area maps back to full image.
	x1 := (0 - padX) / scale
	tl := clampF((0-padX)/scale, 0, 810)
	br := clampF((640-padX)/scale, 0, 810)
	if tl != 0 || br != 810 {
		t.Fatalf("round-trip x: %v..%v (x1=%v)", tl, br, x1)
	}
}

func TestLetterboxTensorRange(t *testing.T) {
	data, _, _, _ := letterbox(solidImage(100, 100), 64)
	for i, v := range data {
		if v < 0 || v > 1 {
			t.Fatalf("tensor[%d]=%v out of range", i, v)
		}
	}
}

func TestDecodeE2EFilter(t *testing.T) {
	flat := []float32{
		10, 10, 50, 50, 0.9, 0, // keep
		10, 10, 50, 50, 0.1, 1, // below floor
		0, 0, 5, 5, 0.8, 79, // keep
	}
	out := decodeE2E(flat, 3, 0.25)
	if len(out) != 2 {
		t.Fatalf("want 2, got %d", len(out))
	}
	if out[0].classID != 0 || out[1].classID != 79 {
		t.Fatalf("bad class ids: %+v", out)
	}
}

func TestIoU(t *testing.T) {
	a := rawDet{x1: 0, y1: 0, x2: 10, y2: 10}
	if got := iou(a, a); got != 1 {
		t.Fatalf("self-iou=%v want 1", got)
	}
	b := rawDet{x1: 20, y1: 20, x2: 30, y2: 30}
	if got := iou(a, b); got != 0 {
		t.Fatalf("disjoint iou=%v want 0", got)
	}
	c := rawDet{x1: 5, y1: 0, x2: 15, y2: 10} // half overlap
	if got := iou(a, c); got < 0.32 || got > 0.34 {
		t.Fatalf("half iou=%v want ~1/3", got)
	}
}

// TestDecodeRawNMS builds a tiny raw tensor: 2 overlapping anchors of class 0
// (NMS must keep 1) + 1 anchor of class 1 (kept).
func TestDecodeRawNMS(t *testing.T) {
	nc, anchors := 2, 3
	flat := make([]float32, (4+nc)*anchors)
	set := func(row int, vals ...float32) {
		for a, v := range vals {
			flat[row*anchors+a] = v
		}
	}
	// rows: cx, cy, w, h, cls0, cls1
	set(0, 100, 102, 300) // cx
	set(1, 100, 102, 300) // cy
	set(2, 50, 52, 40)    // w
	set(3, 50, 52, 40)    // h
	set(4, 0.9, 0.8, 0.1) // class 0 scores
	set(5, 0.05, 0.05, 0.85)
	out := decodeRawNMS(flat, nc, anchors, 0.25, nil)
	if len(out) != 2 {
		t.Fatalf("want 2 after NMS, got %d: %+v", len(out), out)
	}
	classes := map[int]bool{}
	for _, d := range out {
		classes[d.classID] = true
	}
	if !classes[0] || !classes[1] {
		t.Fatalf("want one det per class, got %+v", out)
	}
}

// TestLetterboxTransparent checks that a fully transparent (non-opaque) image
// is composited over the gray pad color rather than black.
func TestLetterboxTransparent(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 50, 30)) // all zero = transparent
	data, _, _, _ := letterbox(img, 64)
	want := float32(letterboxPad) / 255
	for i, v := range data {
		if d := v - want; d > 1e-6 || d < -1e-6 {
			t.Fatalf("tensor[%d]=%v want %v (gray pad)", i, v, want)
		}
	}
}

// TestLetterboxContent checks that an opaque image lands inside the padded
// area with its colors preserved, and the padding is gray.
func TestLetterboxContent(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 64, 32))
	for i := 0; i < len(img.Pix); i += 4 {
		img.Pix[i], img.Pix[i+1], img.Pix[i+2], img.Pix[i+3] = 255, 0, 0, 255
	}
	data, _, padX, padY := letterbox(img, 64)
	if padX != 0 || padY != 16 {
		t.Fatalf("pad=%v,%v want 0,16", padX, padY)
	}
	plane := 64 * 64
	gray := float32(letterboxPad) / 255
	if data[0] != gray || data[plane-1] != gray {
		t.Fatalf("padding not gray: %v %v", data[0], data[plane-1])
	}
	mid := 32*64 + 32
	if data[mid] != 1 || data[plane+mid] != 0 || data[2*plane+mid] != 0 {
		t.Fatalf("content pixel = %v,%v,%v want 1,0,0", data[mid], data[plane+mid], data[2*plane+mid])
	}
}

func BenchmarkLetterbox1080p(b *testing.B) {
	img := solidImage(1920, 1080)
	b.ReportAllocs()
	for b.Loop() {
		letterbox(img, 640)
	}
}
