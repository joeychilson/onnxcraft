package postprocess

import (
	"math"
	"slices"
	"testing"
)

func TestClassifyRanksAndLabels(t *testing.T) {
	t.Parallel()
	logits := []float32{1, 3, 2}
	original := slices.Clone(logits)
	got, err := Classify(logits, ClassificationOptions{
		Labels:  map[int]string{1: "winner"},
		TopK:    2,
		Softmax: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0].Class != 1 || got[0].Label != "winner" || got[1].Label != "2" {
		t.Fatalf("Classify() = %+v", got)
	}
	if got[0].Score <= got[1].Score {
		t.Fatalf("scores are not descending: %+v", got)
	}
	if !slices.Equal(logits, original) {
		t.Fatalf("Classify() mutated logits: %v", logits)
	}
}

func TestClassifyAppliesMinimumScore(t *testing.T) {
	t.Parallel()
	got, err := Classify([]float32{0.8, 0.2}, ClassificationOptions{TopK: 2, MinScore: 0.5})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Class != 0 {
		t.Fatalf("Classify() = %+v", got)
	}
}

func TestClassifyRejectsInvalidInput(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		logits  []float32
		options ClassificationOptions
	}{
		{name: "empty", options: ClassificationOptions{TopK: 1}},
		{name: "top k", logits: []float32{1}},
		{name: "threshold", logits: []float32{1}, options: ClassificationOptions{TopK: 1, MinScore: 2, Softmax: true}},
		{name: "nan", logits: []float32{float32(math.NaN())}, options: ClassificationOptions{TopK: 1}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if _, err := Classify(test.logits, test.options); err == nil {
				t.Fatal("Classify() error = nil")
			}
		})
	}
}

func TestClassifySigmoid(t *testing.T) {
	t.Parallel()
	results, err := Classify([]float32{0, 2}, ClassificationOptions{TopK: 2, Sigmoid: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 || results[0].Class != 1 || results[0].Score < 0.88 || results[1].Score != 0.5 {
		t.Fatalf("results = %+v", results)
	}
	if _, err := Classify([]float32{0}, ClassificationOptions{TopK: 1, Softmax: true, Sigmoid: true}); err == nil {
		t.Fatal("Classify() activation error = nil")
	}
}
