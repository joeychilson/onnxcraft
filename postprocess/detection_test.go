package postprocess

import (
	"image"
	"math"
	"slices"
	"testing"
)

func TestDetectDETRFiltersSuppressesAndLimits(t *testing.T) {
	t.Parallel()
	logits := []float32{
		8, 1, 0,
		7, 1, 0,
		0, 8, 1,
		0, 1, 8,
	}
	boxes := []float32{
		0.5, 0.5, 0.4, 0.4,
		0.5, 0.5, 0.4, 0.4,
		0.1, 0.1, 0.4, 0.4,
		0.9, 0.9, 0.1, 0.1,
	}
	got, err := DetectDETR(logits, boxes, image.Pt(100, 50), DetectionOptions{
		Labels:        map[int]string{0: "person", 1: "dog"},
		MaxDetections: 2,
		MinScore:      0.5,
		IoUThreshold:  0.45,
		ApplyNMS:      true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Label != "person" || got[1].Label != "dog" {
		t.Fatalf("DetectDETR() = %+v", got)
	}
	wantBox := Box{X1: 30, Y1: 15, X2: 70, Y2: 35}
	if !boxesClose(got[0].Box, wantBox) {
		t.Fatalf("first box = %+v", got[0].Box)
	}
	if got[1].Box.X1 != 0 || got[1].Box.Y1 != 0 {
		t.Fatalf("clipped box = %+v", got[1].Box)
	}
}

func TestDetectDETRExcludesNoObjectFromSelection(t *testing.T) {
	t.Parallel()
	got, err := DetectDETR(
		[]float32{2, 1, 3},
		[]float32{0.5, 0.5, 0.25, 0.25},
		image.Pt(100, 100),
		DetectionOptions{Labels: map[int]string{0: "object"}, MaxDetections: 1, MinScore: 0.2},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Class != 0 {
		t.Fatalf("DetectDETR() = %+v", got)
	}
}

func TestDetectDETRDoesNotApplyNMSByDefault(t *testing.T) {
	t.Parallel()
	logits := []float32{4, 0, 4, 0}
	boxes := []float32{0.5, 0.5, 0.4, 0.4, 0.5, 0.5, 0.4, 0.4}
	options := DetectionOptions{Labels: map[int]string{0: "object"}, MaxDetections: 2}
	got, err := DetectDETR(logits, boxes, image.Pt(100, 100), options)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("DetectDETR() returned %d detections, want 2", len(got))
	}
	options.ApplyNMS = true
	options.IoUThreshold = 0.5
	got, err = DetectDETR(logits, boxes, image.Pt(100, 100), options)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("DetectDETR() with NMS returned %d detections, want 1", len(got))
	}
}

func boxesClose(got, want Box) bool {
	return math.Abs(float64(got.X1-want.X1)) < 1e-4 &&
		math.Abs(float64(got.Y1-want.Y1)) < 1e-4 &&
		math.Abs(float64(got.X2-want.X2)) < 1e-4 &&
		math.Abs(float64(got.Y2-want.Y2)) < 1e-4
}

func TestNonMaxSuppressionDoesNotMutateInput(t *testing.T) {
	t.Parallel()
	detections := []Detection{
		{Class: 0, Score: 0.2, Box: Box{X2: 1, Y2: 1}},
		{Class: 1, Score: 0.9, Box: Box{X2: 1, Y2: 1}},
	}
	original := slices.Clone(detections)
	got := NonMaxSuppression(detections, 0.5)
	if got[0].Score != 0.9 {
		t.Fatalf("NonMaxSuppression() = %+v", got)
	}
	if !slices.Equal(detections, original) {
		t.Fatalf("NonMaxSuppression() mutated input: %+v", detections)
	}
}

func TestDetectDETRRejectsMalformedOutputs(t *testing.T) {
	t.Parallel()
	validOptions := DetectionOptions{Labels: map[int]string{0: "object"}, MaxDetections: 1, IoUThreshold: 0.5}
	tests := []struct {
		name   string
		logits []float32
		boxes  []float32
		size   image.Point
		option DetectionOptions
	}{
		{name: "boxes", logits: []float32{1, 0}, boxes: []float32{1}, size: image.Pt(1, 1), option: validOptions},
		{name: "logits", logits: []float32{1, 2, 3}, boxes: make([]float32, 8), size: image.Pt(1, 1), option: validOptions},
		{name: "size", logits: []float32{1, 0}, boxes: []float32{0, 0, 1, 1}, option: validOptions},
		{name: "maximum", logits: []float32{1, 0}, boxes: []float32{0, 0, 1, 1}, size: image.Pt(1, 1), option: DetectionOptions{}},
		{name: "missing no object", logits: []float32{1}, boxes: []float32{0, 0, 1, 1}, size: image.Pt(1, 1), option: validOptions},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if _, err := DetectDETR(test.logits, test.boxes, test.size, test.option); err == nil {
				t.Fatal("DetectDETR() error = nil")
			}
		})
	}
}
