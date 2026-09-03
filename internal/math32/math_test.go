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

func TestSoftmaxIntoReusesDestination(t *testing.T) {
	t.Parallel()
	destination := make([]float32, 2)
	got, err := SoftmaxInto(destination, []float32{0, 0})
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []float32{0.5, 0.5}) || &got[0] != &destination[0] {
		t.Fatalf("SoftmaxInto() = %v", got)
	}
	if _, err := SoftmaxInto(nil, []float32{0}); err == nil {
		t.Fatal("SoftmaxInto() length error = nil")
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
	if got := TopK([]float32{1, 4, 2, 4, 3}, 2); !slices.Equal(got, []int{1, 3}) {
		t.Fatalf("TopK() = %v", got)
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

func FuzzSoftmax(f *testing.F) {
	f.Add([]byte{0, 1, 2})
	f.Add([]byte{255})
	f.Fuzz(func(t *testing.T, values []byte) {
		if len(values) == 0 || len(values) > 1024 {
			return
		}
		logits := make([]float32, len(values))
		for index, value := range values {
			logits[index] = float32(int(value) - 128)
		}
		probabilities, err := Softmax(logits)
		if err != nil {
			t.Fatal(err)
		}
		var sum float64
		for _, probability := range probabilities {
			if probability < 0 || probability > 1 || math.IsNaN(float64(probability)) {
				t.Fatalf("invalid probability %v", probability)
			}
			sum += float64(probability)
		}
		if math.Abs(sum-1) > 1e-5 {
			t.Fatalf("probability sum = %v", sum)
		}
	})
}

func BenchmarkTopK(b *testing.B) {
	values := make([]float32, 30_522)
	for index := range values {
		values[index] = float32(index%997) / 997
	}
	for b.Loop() {
		_ = TopK(values, 5)
	}
}

func BenchmarkSoftmax(b *testing.B) {
	values := make([]float32, 30_522)
	destination := make([]float32, len(values))
	for b.Loop() {
		if _, err := SoftmaxInto(destination, values); err != nil {
			b.Fatal(err)
		}
	}
}
