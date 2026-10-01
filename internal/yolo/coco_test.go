package yolo

import "testing"

func TestLabelsCount(t *testing.T) {
	if len(Labels) != 80 {
		t.Fatalf("want 80 COCO labels, got %d", len(Labels))
	}
	if Labels[0] != "person" || Labels[2] != "car" || Labels[1] != "bicycle" {
		t.Fatalf("unexpected label map: %q %q %q", Labels[0], Labels[1], Labels[2])
	}
}

func TestNameToIDRoundTrip(t *testing.T) {
	for id, name := range Labels {
		if got, ok := NameToID[name]; !ok || got != id {
			t.Fatalf("NameToID[%q] = %d,%v want %d", name, got, ok, id)
		}
	}
}
