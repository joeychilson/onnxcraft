// Package math32 provides numerically stable operations for model outputs.
package math32

import (
	"cmp"
	"errors"
	"math"
	"slices"
)

// Softmax returns normalized probabilities without mutating logits.
func Softmax(logits []float32) ([]float32, error) {
	if len(logits) == 0 {
		return nil, errors.New("math32: logits cannot be empty")
	}
	maximum := float32(math.Inf(-1))
	positiveInfinities := 0
	for _, logit := range logits {
		if math.IsNaN(float64(logit)) {
			return nil, errors.New("math32: logits contain NaN")
		}
		if math.IsInf(float64(logit), 1) {
			positiveInfinities++
		}
		if logit > maximum {
			maximum = logit
		}
	}
	if positiveInfinities > 0 {
		probabilities := make([]float32, len(logits))
		probability := 1 / float32(positiveInfinities)
		for index, logit := range logits {
			if math.IsInf(float64(logit), 1) {
				probabilities[index] = probability
			}
		}
		return probabilities, nil
	}
	if math.IsInf(float64(maximum), -1) {
		return nil, errors.New("math32: all logits are negative infinity")
	}

	probabilities := make([]float32, len(logits))
	var sum float64
	for index, logit := range logits {
		value := math.Exp(float64(logit - maximum))
		probabilities[index] = float32(value)
		sum += value
	}
	for index := range probabilities {
		probabilities[index] = float32(float64(probabilities[index]) / sum)
	}
	return probabilities, nil
}

// TopK returns the indices of the largest values in descending, stable order.
func TopK(values []float32, count int) []int {
	count = min(max(count, 0), len(values))
	indices := make([]int, len(values))
	for index := range indices {
		indices[index] = index
	}
	slices.SortStableFunc(indices, func(left, right int) int {
		return cmp.Compare(values[right], values[left])
	})
	return slices.Clone(indices[:count])
}

// IntersectionOverUnion computes IoU for two x1, y1, x2, y2 boxes.
func IntersectionOverUnion(left, right [4]float32) float32 {
	intersectionWidth := max(0, min(left[2], right[2])-max(left[0], right[0]))
	intersectionHeight := max(0, min(left[3], right[3])-max(left[1], right[1]))
	intersection := intersectionWidth * intersectionHeight
	leftArea := max(0, left[2]-left[0]) * max(0, left[3]-left[1])
	rightArea := max(0, right[2]-right[0]) * max(0, right[3]-right[1])
	union := leftArea + rightArea - intersection
	if union <= 0 {
		return 0
	}
	return intersection / union
}
