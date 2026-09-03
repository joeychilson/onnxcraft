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
	return SoftmaxInto(make([]float32, len(logits)), logits)
}

// SoftmaxInto writes normalized probabilities to destination. It returns an
// error when the slices differ in length or logits cannot be normalized.
func SoftmaxInto(destination, logits []float32) ([]float32, error) {
	if len(logits) == 0 {
		return nil, errors.New("math32: logits cannot be empty")
	}
	if len(destination) != len(logits) {
		return nil, errors.New("math32: softmax destination length does not match logits")
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
		probability := 1 / float32(positiveInfinities)
		for index, logit := range logits {
			if math.IsInf(float64(logit), 1) {
				destination[index] = probability
			} else {
				destination[index] = 0
			}
		}
		return destination, nil
	}
	if math.IsInf(float64(maximum), -1) {
		return nil, errors.New("math32: all logits are negative infinity")
	}

	var sum float64
	for index, logit := range logits {
		value := math.Exp(float64(logit - maximum))
		destination[index] = float32(value)
		sum += value
	}
	for index := range destination {
		destination[index] = float32(float64(destination[index]) / sum)
	}
	return destination, nil
}

// SoftmaxArgMax returns the highest-logit index among the first candidateCount
// values and its probability after softmax normalization over every logit.
func SoftmaxArgMax(logits []float32, candidateCount int) (int, float32, error) {
	if len(logits) == 0 {
		return 0, 0, errors.New("math32: logits cannot be empty")
	}
	if candidateCount < 1 || candidateCount > len(logits) {
		return 0, 0, errors.New("math32: candidate count is out of range")
	}
	maximum := float32(math.Inf(-1))
	positiveInfinities := 0
	best := 0
	for index, logit := range logits {
		if math.IsNaN(float64(logit)) {
			return 0, 0, errors.New("math32: logits contain NaN")
		}
		if math.IsInf(float64(logit), 1) {
			positiveInfinities++
		}
		if logit > maximum {
			maximum = logit
		}
		if index < candidateCount && logit > logits[best] {
			best = index
		}
	}
	if positiveInfinities > 0 {
		if math.IsInf(float64(logits[best]), 1) {
			return best, 1 / float32(positiveInfinities), nil
		}
		return best, 0, nil
	}
	if math.IsInf(float64(maximum), -1) {
		return 0, 0, errors.New("math32: all logits are negative infinity")
	}
	var sum float64
	for _, logit := range logits {
		sum += math.Exp(float64(logit - maximum))
	}
	probability := math.Exp(float64(logits[best]-maximum)) / sum
	return best, float32(probability), nil
}

// TopK returns the indices of the largest values in descending, stable order.
func TopK(values []float32, count int) []int {
	count = min(max(count, 0), len(values))
	if count == 0 {
		return []int{}
	}
	indices := make([]int, 0, count)
	for index := range values {
		if len(indices) < count {
			indices = append(indices, index)
			siftTopKWorstUp(indices, values, len(indices)-1)
			continue
		}
		if topKBetter(values, index, indices[0]) {
			indices[0] = index
			siftTopKWorstDown(indices, values, 0)
		}
	}
	slices.SortFunc(indices, func(left, right int) int {
		if order := cmp.Compare(values[right], values[left]); order != 0 {
			return order
		}
		return cmp.Compare(left, right)
	})
	return indices
}

func topKBetter(values []float32, left, right int) bool {
	return values[left] > values[right] || values[left] == values[right] && left < right
}

func siftTopKWorstUp(heap []int, values []float32, index int) {
	for index > 0 {
		parent := (index - 1) / 2
		if !topKBetter(values, heap[parent], heap[index]) {
			return
		}
		heap[parent], heap[index] = heap[index], heap[parent]
		index = parent
	}
}

func siftTopKWorstDown(heap []int, values []float32, index int) {
	for {
		left := index*2 + 1
		if left >= len(heap) {
			return
		}
		worst := left
		right := left + 1
		if right < len(heap) && topKBetter(values, heap[left], heap[right]) {
			worst = right
		}
		if !topKBetter(values, heap[index], heap[worst]) {
			return
		}
		heap[index], heap[worst] = heap[worst], heap[index]
		index = worst
	}
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
