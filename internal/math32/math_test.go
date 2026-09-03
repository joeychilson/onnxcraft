package math32

import (
	"math"
	"slices"
	"testing"
)

func TestSoftmaxIsStable(t *testing.T) {
	t.Parallel()
	probabilities, err := Softmax([]float32{1000, 1001, 1002})
	if err != nil {
		t.Fatal(err)
	}
	var sum float32
	for _, probability := range probabilities {
		sum += probability
	}
	if math.Abs(float64(sum-1)) > 1e-6 {
		t.Fatalf("sum = %v", sum)
	}
	if !(probabilities[2] > probabilities[1] && probabilities[1] > probabilities[0]) {
		t.Fatalf("probabilities = %v", probabilities)
	}
}

func TestSoftmaxHandlesInfinities(t *testing.T) {
	t.Parallel()
	probabilities, err := Softmax([]float32{float32(math.Inf(1)), 0, float32(math.Inf(1))})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{0.5, 0, 0.5}
	if !slices.Equal(probabilities, want) {
		t.Fatalf("Softmax() = %v, want %v", probabilities, want)
	}
	if _, err := Softmax([]float32{float32(math.NaN())}); err == nil {
		t.Fatal("Softmax(NaN) error = nil")
	}
	if _, err := Softmax([]float32{float32(math.Inf(-1))}); err == nil {
		t.Fatal("Softmax(-Inf) error = nil")
	}
}

func TestTopKIsStableAndBounded(t *testing.T) {
	t.Parallel()
	if got := TopK([]float32{3, 3, 1}, 8); !slices.Equal(got, []int{0, 1, 2}) {
		t.Fatalf("TopK() = %v", got)
	}
	if got := TopK([]float32{1}, -1); len(got) != 0 {
		t.Fatalf("TopK(-1) = %v", got)
	}
}

func TestIntersectionOverUnion(t *testing.T) {
	t.Parallel()
	left := [4]float32{0, 0, 2, 2}
	right := [4]float32{1, 1, 3, 3}
	if got := IntersectionOverUnion(left, right); math.Abs(float64(got-1.0/7.0)) > 1e-6 {
		t.Fatalf("IntersectionOverUnion() = %v", got)
	}
	if got := IntersectionOverUnion([4]float32{}, [4]float32{}); got != 0 {
		t.Fatalf("IntersectionOverUnion(empty) = %v", got)
	}
}
