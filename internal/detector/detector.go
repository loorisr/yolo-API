// Package detector isolates the ONNX Runtime binding behind a swappable
// interface. Primary binding: github.com/shota3506/onnxruntime-purego
// (purego, no CGO). SPEC fallback order is preserved: this implementation can
// be replaced by a raven-onnxruntime or onnxer backend without touching the
// HTTP layer.
//
// Both model layouts are supported (auto-detected at startup with a zero
// dummy inference):
//   - e2e NMS-free export: (1, N, 6) = [x1,y1,x2,y2,score,class_id]
//   - raw export: (1, 4+nc, anchors), e.g. (1, 84, 8400) for COCO,
//     decoded here with per-class NMS.
package detector

import (
	"context"
	"errors"
	"fmt"
	"image"
	"image/color"
	"log/slog"
	"math"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	ort "github.com/shota3506/onnxruntime-purego/onnxruntime"
	"golang.org/x/image/draw"

	"yolo-api/internal/yolo"
)

// ortAPIVersion is the ONNX Runtime C API version used by the binding.
const ortAPIVersion = 23

// iouThreshold is the NMS IoU threshold for raw-output models.
const iouThreshold = 0.45

// maxDetections caps the number of detections returned per image.
const maxDetections = 300

// decodeFloor is the minimum score floor used during raw-output decoding.
// The request min_conf is applied on top; min_conf values below the floor
// are clamped to it to bound NMS cost.
const decodeFloor = 0.01

// letterboxPad is the gray padding value used by YOLO letterboxing.
const letterboxPad = 114

// poolWaitTimeout bounds how long Detect waits for a free session before
// returning ErrPoolBusy (mapped to HTTP 503 by the transport).
const poolWaitTimeout = 30 * time.Second

// ErrPoolBusy is returned when no inference session frees up in time.
var ErrPoolBusy = errors.New("inference pool busy")

// finite reports whether v is neither NaN nor ±Inf.
func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// Box is a bounding box in original-image pixel coordinates.
type Box struct {
	XMin float64 `json:"x_min"`
	YMin float64 `json:"y_min"`
	XMax float64 `json:"x_max"`
	YMax float64 `json:"y_max"`
}

// Detection is a single filtered detection.
type Detection struct {
	ClassID    int     `json:"class_id"`
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence"`
	Box        Box     `json:"box"`
	BoxNorm    Box     `json:"box_norm"`
}

// Detector runs object detection on decoded images.
type Detector interface {
	Detect(ctx context.Context, img image.Image, minConf float32, classes map[int]bool) (dets []Detection, inferMs float64, err error)
	Close() error
}

// layout describes the model's output tensor layout.
type layout int

const (
	layoutE2E layout = iota
	layoutRaw
	layoutRawTransposed
)

// ORTDetector is a pool of ONNX Runtime sessions. Sessions are NOT shared
// between goroutines: each Detect call exclusively borrows one session.
type ORTDetector struct {
	rt         *ort.Runtime
	env        *ort.Env
	pool       chan *ort.Session
	inputName  string
	outputName string
	layout     layout
	numClasses int
	numAnchors int
	maxDet     int
	imgSize    int
	modelName  string
	intra      int
	log        *slog.Logger
	done       chan struct{}
	closeOnce  sync.Once
	closeErr   error
}

// ModelName returns the short model name (e.g. "yolo26n").
func (d *ORTDetector) ModelName() string { return d.modelName }

// Version returns the loaded ONNX Runtime version string.
func (d *ORTDetector) Version() string {
	if d.rt == nil {
		return ""
	}
	return d.rt.GetVersionString()
}

// New creates an ORTDetector with poolSize sessions.
func New(modelPath, libPath string, imgSize, poolSize, intraThreads int, log *slog.Logger) (*ORTDetector, error) {
	rt, err := ort.NewRuntime(libPath, ortAPIVersion)
	if err != nil {
		return nil, fmt.Errorf("load ORT lib %q: %w", libPath, err)
	}
	env, err := rt.NewEnv("yolo-api", ort.LoggingLevelWarning)
	if err != nil {
		_ = rt.Close()
		return nil, fmt.Errorf("create ORT env: %w", err)
	}
	d := &ORTDetector{
		rt:        rt,
		env:       env,
		pool:      make(chan *ort.Session, poolSize),
		imgSize:   imgSize,
		modelName: modelNameFromPath(modelPath),
		intra:     intraThreads,
		log:       log,
		done:      make(chan struct{}),
	}
	opts := &ort.SessionOptions{IntraOpNumThreads: intraThreads}
	probe, err := rt.NewSession(env, modelPath, opts)
	if err != nil {
		env.Close()
		_ = rt.Close()
		return nil, fmt.Errorf("create ORT session for %q: %w", modelPath, err)
	}
	if err := d.learnLayout(probe); err != nil {
		probe.Close()
		env.Close()
		_ = rt.Close()
		return nil, err
	}
	probe.Close()
	for i := 0; i < poolSize; i++ {
		s, err := rt.NewSession(env, modelPath, opts)
		if err != nil {
			_ = d.Close()
			return nil, fmt.Errorf("create ORT session %d/%d: %w", i+1, poolSize, err)
		}
		d.pool <- s
	}
	log.Info("detector ready",
		"model", d.modelName, "layout", d.layout.String(),
		"pool", poolSize, "img_size", imgSize, "ort_version", rt.GetVersionString())
	return d, nil
}

func (l layout) String() string {
	if l == layoutE2E {
		return "e2e"
	}
	if l == layoutRawTransposed {
		return "raw-transposed+nms"
	}
	return "raw+nms"
}

func modelNameFromPath(p string) string {
	base := filepath.Base(p)
	name := strings.TrimSuffix(base, filepath.Ext(base))
	name = strings.TrimSuffix(name, "-e2e")
	return name
}

// learnLayout runs a zero dummy inference to discover input/output names and
// the output tensor layout.
func (d *ORTDetector) learnLayout(s *ort.Session) error {
	in := s.InputNames()
	out := s.OutputNames()
	if len(in) == 0 || len(out) == 0 {
		return fmt.Errorf("model has no inputs/outputs")
	}
	d.inputName, d.outputName = in[0], out[0]
	n := d.imgSize * d.imgSize * 3
	zeros := make([]float32, n)
	tensor, err := ort.NewTensorValue(d.rt, zeros, []int64{1, 3, int64(d.imgSize), int64(d.imgSize)})
	if err != nil {
		return fmt.Errorf("dummy input tensor: %w", err)
	}
	defer tensor.Close()
	results, err := s.Run(context.Background(), map[string]*ort.Value{d.inputName: tensor})
	runtime.KeepAlive(zeros) // zeros backs the input tensor via an unsafe pointer
	if err != nil {
		return fmt.Errorf("dummy inference (is IMG_SIZE=%d matching the export?): %w", d.imgSize, err)
	}
	defer func() {
		for _, v := range results {
			v.Close()
		}
	}()
	v := results[d.outputName]
	if v == nil {
		return fmt.Errorf("missing output %q", d.outputName)
	}
	shape, err := v.GetTensorShape()
	if err != nil {
		return fmt.Errorf("output shape: %w", err)
	}
	if len(shape) != 3 || shape[0] != 1 {
		return fmt.Errorf("unexpected output shape %v (want [1,N,6] or [1,4+nc,A])", shape)
	}
	d1, d2 := int(shape[1]), int(shape[2])
	// e2e NMS-free: [1,N,6] with N bounded (typically <= 300). The check on
	// d1 guards against misreading a raw [1,6,A] model (nc=2) as e2e.
	if d2 == 6 && d1 != 6 {
		if d1 <= 0 || d1 > 4096 {
			return fmt.Errorf("unexpected e2e output shape %v", shape)
		}
		d.layout = layoutE2E
		d.maxDet = d1
		d.numClasses = len(yolo.Labels)
		return nil
	}
	// raw: standard [1,4+nc,A] or transposed [1,A,4+nc]. Disambiguate by
	// which axis is small (4+nc <= 256) and which is large (anchors).
	if d1 >= 5 && d1 <= 260 && d2 >= 100 && d2 <= 200000 {
		d.layout = layoutRaw
		d.numClasses = d1 - 4
		d.numAnchors = d2
		d.maxDet = maxDetections
	} else if d2 >= 5 && d2 <= 260 && d1 >= 100 && d1 <= 200000 {
		d.layout = layoutRawTransposed
		d.numClasses = d2 - 4
		d.numAnchors = d1
		d.maxDet = maxDetections
	} else {
		return fmt.Errorf("unexpected output shape %v (want [1,N,6] or [1,4+nc,A])", shape)
	}
	if d.numClasses < 1 || d.numClasses > 256 {
		return fmt.Errorf("unexpected class count %d from shape %v", d.numClasses, shape)
	}
	if (d.layout == layoutRaw || d.layout == layoutRawTransposed) && d.numClasses != len(yolo.Labels) {
		d.log.Warn("model class count differs from the COCO-80 label map; labels may be wrong",
			"model", d.modelName, "num_classes", d.numClasses)
	}
	return nil
}

// Detect runs inference on img, filters by minConf and classes, and returns
// detections in original-image pixel coordinates plus normalized boxes.
func (d *ORTDetector) Detect(ctx context.Context, img image.Image, minConf float32, classes map[int]bool) ([]Detection, float64, error) {
	var s *ort.Session
	timer := time.NewTimer(poolWaitTimeout)
	defer timer.Stop()
	select {
	case s = <-d.pool:
	case <-ctx.Done():
		return nil, 0, ctx.Err()
	case <-timer.C:
		return nil, 0, ErrPoolBusy
	}
	defer func() {
		// Prefer Close when shutting down: checking done first avoids the
		// random select when both branches are ready, which would leak a
		// session past rt.Close().
		select {
		case <-d.done:
			s.Close()
		default:
			select {
			case d.pool <- s:
			case <-d.done:
				s.Close()
			}
		}
	}()

	origW, origH := img.Bounds().Dx(), img.Bounds().Dy()
	if origW <= 0 || origH <= 0 {
		return nil, 0, fmt.Errorf("empty image")
	}
	data, scale, padX, padY := letterbox(img, d.imgSize)
	tensor, err := ort.NewTensorValue(d.rt, data, []int64{1, 3, int64(d.imgSize), int64(d.imgSize)})
	if err != nil {
		return nil, 0, fmt.Errorf("input tensor: %w", err)
	}
	defer tensor.Close()

	t0 := time.Now()
	results, err := s.Run(ctx, map[string]*ort.Value{d.inputName: tensor})
	inferMs := float64(time.Since(t0).Microseconds()) / 1000.0
	runtime.KeepAlive(data) // data backs the input tensor via an unsafe pointer
	if err != nil {
		return nil, 0, fmt.Errorf("inference: %w", err)
	}
	defer func() {
		for _, v := range results {
			v.Close()
		}
	}()
	v := results[d.outputName]
	if v == nil {
		return nil, 0, fmt.Errorf("missing output %q", d.outputName)
	}
	flat, _, err := ort.GetTensorData[float32](v)
	if err != nil {
		return nil, 0, fmt.Errorf("output data: %w", err)
	}

	floor := minConf
	if (d.layout == layoutRaw || d.layout == layoutRawTransposed) && floor < decodeFloor {
		floor = decodeFloor
	}
	var raw []rawDet
	if d.layout == layoutE2E {
		raw = decodeE2E(flat, d.maxDet, floor)
	} else if d.layout == layoutRawTransposed {
		if len(flat) < d.numAnchors*(4+d.numClasses) {
			return nil, 0, fmt.Errorf("output size mismatch: got %d, want %d", len(flat), d.numAnchors*(4+d.numClasses))
		}
		raw = decodeRawNMSTransposed(flat, d.numClasses, d.numAnchors, floor, classes)
	} else {
		if len(flat) < (4+d.numClasses)*d.numAnchors {
			return nil, 0, fmt.Errorf("output size mismatch: got %d, want %d", len(flat), (4+d.numClasses)*d.numAnchors)
		}
		raw = decodeRawNMS(flat, d.numClasses, d.numAnchors, floor, classes)
	}

	dets := make([]Detection, 0, len(raw))
	wf, hf := float64(origW), float64(origH)
	for _, r := range raw {
		if !finite(float64(r.score)) || !finite(float64(r.x1)) || !finite(float64(r.y1)) ||
			!finite(float64(r.x2)) || !finite(float64(r.y2)) {
			continue
		}
		if r.score < minConf {
			continue
		}
		if len(classes) > 0 && !classes[r.classID] {
			continue
		}
		if r.classID < 0 || r.classID >= len(yolo.Labels) {
			continue
		}
		conf := float64(r.score)
		if conf > 1 {
			conf = 1
		}
		x1 := clampF((float64(r.x1)-padX)/scale, 0, wf)
		y1 := clampF((float64(r.y1)-padY)/scale, 0, hf)
		x2 := clampF((float64(r.x2)-padX)/scale, 0, wf)
		y2 := clampF((float64(r.y2)-padY)/scale, 0, hf)
		if x2 <= x1 || y2 <= y1 {
			continue
		}
		dets = append(dets, Detection{
			ClassID:    r.classID,
			Label:      yolo.Labels[r.classID],
			Confidence: conf,
			Box:        Box{x1, y1, x2, y2},
			BoxNorm:    Box{x1 / wf, y1 / hf, x2 / wf, y2 / hf},
		})
		if len(dets) >= maxDetections {
			break
		}
	}
	sort.Slice(dets, func(i, j int) bool { return dets[i].Confidence > dets[j].Confidence })
	return dets, inferMs, nil
}

// Close releases all sessions, the env and the runtime. It is idempotent and
// safe to call while requests are still returning sessions (the HTTP layer
// drains in-flight handlers before closing).
func (d *ORTDetector) Close() error {
	d.closeOnce.Do(func() {
		close(d.done)
	drain:
		for {
			select {
			case s := <-d.pool:
				s.Close()
			default:
				break drain
			}
		}
		if d.env != nil {
			d.env.Close()
		}
		if d.rt != nil {
			d.closeErr = d.rt.Close()
		}
	})
	return d.closeErr
}

// rawDet is a decoded box in letterboxed imgSize coordinates.
type rawDet struct {
	x1, y1, x2, y2 float32
	score          float32
	classID        int
}

// letterbox resizes img keeping aspect ratio into imgSize x imgSize with gray
// padding, and returns the NCHW float32 tensor (0..1) plus the scale and pad
// offsets needed to map boxes back to the original image.
func letterbox(img image.Image, size int) (data []float32, scale, padX, padY float64) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	scale = min(float64(size)/float64(w), float64(size)/float64(h))
	nw, nh := int(float64(w)*scale+0.5), int(float64(h)*scale+0.5)
	if nw < 1 {
		nw = 1
	}
	if nh < 1 {
		nh = 1
	}
	if nw > size {
		nw = size
	}
	if nh > size {
		nh = size
	}
	dst := image.NewRGBA(image.Rect(0, 0, nw, nh))
	// Composite over the gray pad color so images with an alpha channel don't
	// get blended over transparent black (which would darken them).
	padCol := color.RGBA{letterboxPad, letterboxPad, letterboxPad, 255}
	draw.Draw(dst, dst.Bounds(), image.NewUniform(padCol), image.Point{}, draw.Src)
	draw.BiLinear.Scale(dst, dst.Bounds(), img, b, draw.Over, nil)
	padW, padH := size-nw, size-nh
	// Integer grid: odd pads split floor/ceil (extra pixel goes right/bottom).
	// padX/padY match the integer offsets below so un-letterbox inverts exactly.
	padX, padY = float64(padW/2), float64(padH/2)

	data = make([]float32, 3*size*size)
	plane := size * size
	offY, offX := padH/2, padW/2
	const inv = 1.0 / 255.0
	for y := 0; y < size; y++ {
		sy := y - offY
		for x := 0; x < size; x++ {
			sx := x - offX
			idx := y*size + x
			if sx < 0 || sy < 0 || sx >= nw || sy >= nh {
				data[idx] = letterboxPad * inv
				data[plane+idx] = letterboxPad * inv
				data[2*plane+idx] = letterboxPad * inv
				continue
			}
			r, g, bl, _ := dst.RGBAAt(sx, sy).RGBA()
			data[idx] = float32(r>>8) * inv
			data[plane+idx] = float32(g>>8) * inv
			data[2*plane+idx] = float32(bl>>8) * inv
		}
	}
	return data, scale, padX, padY
}

// decodeE2E decodes (N,6) rows [x1,y1,x2,y2,score,class] with a score filter.
// Non-finite scores, boxes and class ids are dropped so they can never reach JSON.
func decodeE2E(flat []float32, n int, floor float32) []rawDet {
	out := make([]rawDet, 0, min(n, 64))
	for i := 0; i < n; i++ {
		o := i * 6
		if o+5 >= len(flat) {
			break
		}
		score, cls := flat[o+4], flat[o+5]
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) {
			continue
		}
		if score < floor {
			continue
		}
		if math.IsNaN(float64(cls)) || math.IsInf(float64(cls), 0) {
			continue
		}
		if math.IsNaN(float64(flat[o])) || math.IsInf(float64(flat[o]), 0) ||
			math.IsNaN(float64(flat[o+1])) || math.IsInf(float64(flat[o+1]), 0) ||
			math.IsNaN(float64(flat[o+2])) || math.IsInf(float64(flat[o+2]), 0) ||
			math.IsNaN(float64(flat[o+3])) || math.IsInf(float64(flat[o+3]), 0) {
			continue
		}
		out = append(out, rawDet{
			x1: flat[o], y1: flat[o+1], x2: flat[o+2], y2: flat[o+3],
			score: score, classID: int(cls),
		})
	}
	return out
}

// decodeRawNMS decodes a raw (4+nc, anchors) row-major tensor with per-class
// greedy NMS. Boxes are cx,cy,w,h in letterboxed coordinates. When classes is
// non-empty, anchors whose best class isn't requested are skipped before NMS.
func decodeRawNMS(flat []float32, nc, anchors int, floor float32, classes map[int]bool) []rawDet {
	if nc <= 0 || nc > 256 || anchors <= 0 || anchors > 200000 {
		return nil
	}
	if len(flat) < (4+nc)*anchors {
		return nil
	}
	perClass := make([][]rawDet, nc)
	for a := 0; a < anchors; a++ {
		cx := flat[a]
		cy := flat[anchors+a]
		w := flat[2*anchors+a]
		h := flat[3*anchors+a]
		best, bestC := float32(0), -1
		for c := 0; c < nc; c++ {
			sc := flat[(4+c)*anchors+a]
			if sc > best {
				best, bestC = sc, c
			}
		}
		if bestC < 0 || best < floor {
			continue
		}
		if len(classes) > 0 && !classes[bestC] {
			continue
		}
		hw, hh := w/2, h/2
		perClass[bestC] = append(perClass[bestC], rawDet{
			x1: cx - hw, y1: cy - hh, x2: cx + hw, y2: cy + hh,
			score: best, classID: bestC,
		})
	}
	out := nmsPerClass(perClass)
	return out
}

// nmsPerClass runs greedy per-class NMS and merges the survivors sorted by score.
func nmsPerClass(perClass [][]rawDet) []rawDet {
	out := make([]rawDet, 0, 64)
	for _, cands := range perClass {
		if len(cands) == 0 {
			continue
		}
		sort.Slice(cands, func(i, j int) bool { return cands[i].score > cands[j].score })
		kept := cands[:0]
		for _, c := range cands {
			drop := false
			for _, k := range kept {
				if iou(c, k) > iouThreshold {
					drop = true
					break
				}
			}
			if !drop {
				kept = append(kept, c)
			}
		}
		out = append(out, kept...)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].score > out[j].score })
	if len(out) > maxDetections {
		out = out[:maxDetections]
	}
	return out
}

// decodeRawNMSTransposed decodes a transposed raw [anchors, 4+nc] row-major
// tensor (e.g. [1,8400,84]) with the same per-class greedy NMS as decodeRawNMS.
func decodeRawNMSTransposed(flat []float32, nc, anchors int, floor float32, classes map[int]bool) []rawDet {
	if nc <= 0 || nc > 256 || anchors <= 0 || anchors > 200000 {
		return nil
	}
	if len(flat) < anchors*(4+nc) {
		return nil
	}
	stride := 4 + nc
	perClass := make([][]rawDet, nc)
	for a := 0; a < anchors; a++ {
		base := a * stride
		cx := flat[base]
		cy := flat[base+1]
		w := flat[base+2]
		h := flat[base+3]
		best, bestC := float32(0), -1
		for c := 0; c < nc; c++ {
			sc := flat[base+4+c]
			if sc > best {
				best, bestC = sc, c
			}
		}
		if bestC < 0 || best < floor {
			continue
		}
		if len(classes) > 0 && !classes[bestC] {
			continue
		}
		hw, hh := w/2, h/2
		perClass[bestC] = append(perClass[bestC], rawDet{
			x1: cx - hw, y1: cy - hh, x2: cx + hw, y2: cy + hh,
			score: best, classID: bestC,
		})
	}
	return nmsPerClass(perClass)
}

func iou(a, b rawDet) float32 {
	ix1 := max(a.x1, b.x1)
	iy1 := max(a.y1, b.y1)
	ix2 := min(a.x2, b.x2)
	iy2 := min(a.y2, b.y2)
	iw, ih := ix2-ix1, iy2-iy1
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	ua := (a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter
	if ua <= 0 {
		return 0
	}
	return inter / ua
}

func clampF(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
