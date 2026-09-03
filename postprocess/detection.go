package postprocess

import (
	"cmp"
	"errors"
	"fmt"
	"image"
	"math"
	"slices"

	"github.com/joeychilson/infergo/internal/math32"
)

// Box is an axis-aligned box in pixel coordinates.
type Box struct {
	X1 float32
	Y1 float32
	X2 float32
	Y2 float32
}

// Detection is one object prediction.
type Detection struct {
	Class int
	Label string
	Score float32
	Box   Box
}

// DetectionOptions configures DetectDETR.
type DetectionOptions struct {
	Labels        map[int]string
	MaxDetections int
	MinScore      float32
	IoUThreshold  float32
}

// DetectDETR converts flattened DETR-style logits and normalized center boxes
// into class-aware, non-max-suppressed pixel detections. Classes absent from
// Labels, including the usual no-object class, are ignored.
func DetectDETR(logits, boxes []float32, imageSize image.Point, options DetectionOptions) ([]Detection, error) {
	if len(boxes) == 0 || len(boxes)%4 != 0 {
		return nil, errors.New("postprocess: boxes must contain a positive multiple of four values")
	}
	boxCount := len(boxes) / 4
	if len(logits) == 0 || len(logits)%boxCount != 0 {
		return nil, fmt.Errorf("postprocess: %d logits cannot be divided across %d boxes", len(logits), boxCount)
	}
	if imageSize.X < 1 || imageSize.Y < 1 {
		return nil, errors.New("postprocess: image dimensions must be positive")
	}
	if options.MaxDetections < 1 {
		return nil, errors.New("postprocess: MaxDetections must be positive")
	}
	if options.MinScore < 0 || options.MinScore > 1 || math.IsNaN(float64(options.MinScore)) {
		return nil, errors.New("postprocess: MinScore must be between 0 and 1")
	}
	if options.IoUThreshold < 0 || options.IoUThreshold > 1 || math.IsNaN(float64(options.IoUThreshold)) {
		return nil, errors.New("postprocess: IoUThreshold must be between 0 and 1")
	}

	classCount := len(logits) / boxCount
	detections := make([]Detection, 0, boxCount)
	for boxIndex := range boxCount {
		boxValues := boxes[boxIndex*4 : (boxIndex+1)*4]
		for _, value := range boxValues {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("postprocess: box %d contains a non-finite coordinate", boxIndex)
			}
		}
		if boxValues[2] < 0 || boxValues[3] < 0 {
			return nil, fmt.Errorf("postprocess: box %d has a negative dimension", boxIndex)
		}
		scores, err := math32.Softmax(logits[boxIndex*classCount : (boxIndex+1)*classCount])
		if err != nil {
			return nil, fmt.Errorf("postprocess: box %d: %w", boxIndex, err)
		}
		class := math32.TopK(scores, 1)[0]
		label, ok := options.Labels[class]
		if !ok || scores[class] < options.MinScore {
			continue
		}
		box := normalizedBox(boxValues, imageSize)
		detections = append(detections, Detection{Class: class, Label: label, Score: scores[class], Box: box})
	}

	detections = NonMaxSuppression(detections, options.IoUThreshold)
	return slices.Clone(detections[:min(len(detections), options.MaxDetections)]), nil
}

// NonMaxSuppression removes lower-scoring overlapping boxes of the same class.
// It does not mutate detections.
func NonMaxSuppression(detections []Detection, threshold float32) []Detection {
	sorted := slices.Clone(detections)
	slices.SortStableFunc(sorted, func(left, right Detection) int {
		return cmp.Compare(right.Score, left.Score)
	})
	result := make([]Detection, 0, len(sorted))
	for _, candidate := range sorted {
		suppressed := false
		for _, kept := range result {
			if candidate.Class != kept.Class {
				continue
			}
			left := [4]float32{candidate.Box.X1, candidate.Box.Y1, candidate.Box.X2, candidate.Box.Y2}
			right := [4]float32{kept.Box.X1, kept.Box.Y1, kept.Box.X2, kept.Box.Y2}
			if math32.IntersectionOverUnion(left, right) > threshold {
				suppressed = true
				break
			}
		}
		if !suppressed {
			result = append(result, candidate)
		}
	}
	return result
}

func normalizedBox(values []float32, size image.Point) Box {
	centerX, centerY, width, height := values[0], values[1], values[2], values[3]
	return Box{
		X1: max(0, min(float32(size.X), (centerX-width/2)*float32(size.X))),
		Y1: max(0, min(float32(size.Y), (centerY-height/2)*float32(size.Y))),
		X2: max(0, min(float32(size.X), (centerX+width/2)*float32(size.X))),
		Y2: max(0, min(float32(size.Y), (centerY+height/2)*float32(size.Y))),
	}
}
