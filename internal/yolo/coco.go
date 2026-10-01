// Package yolo holds the COCO-80 label map used by YOLO26 detection models.
package yolo

// Labels maps class IDs 0..79 to COCO label names.
var Labels = []string{
	"person", "bicycle", "car", "motorcycle", "airplane",
	"bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird",
	"cat", "dog", "horse", "sheep", "cow",
	"elephant", "bear", "zebra", "giraffe", "backpack",
	"umbrella", "handbag", "tie", "suitcase", "frisbee",
	"skis", "snowboard", "sports ball", "kite", "baseball bat",
	"baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon",
	"bowl", "banana", "apple", "sandwich", "orange",
	"broccoli", "carrot", "hot dog", "pizza", "donut",
	"cake", "chair", "couch", "potted plant", "bed",
	"dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven",
	"toaster", "sink", "refrigerator", "book", "clock",
	"vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}

// NameToID maps lowercase COCO label names to class IDs for the
// `classes` query filter. Built once at init.
var NameToID = func() map[string]int {
	m := make(map[string]int, len(Labels))
	for i, name := range Labels {
		m[name] = i
	}
	return m
}()
