package bert

import (
	"math"
	"testing"
)

func TestValidateClassifyOptions(t *testing.T) {
	t.Parallel()
	valid := ClassifyOptions{TopK: 1, MaxLength: 8, Activation: ClassificationSoftmax}
	if err := validateClassifyOptions(valid); err != nil {
		t.Fatal(err)
	}
	invalid := []ClassifyOptions{
		{TopK: 0, MaxLength: 8, Activation: ClassificationSoftmax},
		{TopK: 1, MaxLength: 1, Activation: ClassificationSoftmax},
		{TopK: 1, MaxLength: 8, Activation: ClassificationActivation("other")},
		{TopK: 1, MaxLength: 8, MinScore: 2, Activation: ClassificationSigmoid},
		{TopK: 1, MaxLength: 8, MinScore: float32(math.NaN()), Activation: ClassificationRaw},
	}
	for _, options := range invalid {
		if err := validateClassifyOptions(options); err == nil {
			t.Errorf("validateClassifyOptions(%+v) error = nil", options)
		}
	}
}

func TestClassifierClassCount(t *testing.T) {
	t.Parallel()
	if count, err := classifierClassCount([]int64{2, 3}, 2, 6); err != nil || count != 3 {
		t.Fatalf("classifierClassCount() = %d, %v", count, err)
	}
	if count, err := classifierClassCount([]int64{3}, 1, 3); err != nil || count != 3 {
		t.Fatalf("classifierClassCount() = %d, %v", count, err)
	}
	if _, err := classifierClassCount([]int64{2, 3, 4}, 2, 24); err == nil {
		t.Fatal("classifierClassCount() error = nil")
	}
}

func TestNilClassifierClose(t *testing.T) {
	t.Parallel()
	var classifier *Classifier
	if err := classifier.Close(); err != nil {
		t.Fatal(err)
	}
}
