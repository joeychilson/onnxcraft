package resnet

import (
	"math"
	"testing"

	"github.com/joeychilson/infergo"
)

func TestModelOptions(t *testing.T) {
	t.Parallel()
	config := modelConfig{}
	if err := WithImageSize(320, 240)(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithNormalization([3]float32{1, 2, 3}, [3]float32{4, 5, 6})(&config); err != nil {
		t.Fatal(err)
	}
	inputLabels := map[int]string{0: "zero"}
	if err := WithLabels(inputLabels)(&config); err != nil {
		t.Fatal(err)
	}
	inputLabels[0] = "changed"
	if err := WithTensorNames("input", "output")(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithSessionOptions(infergo.WithOptimization(infergo.OptimizationBasic))(&config); err != nil {
		t.Fatal(err)
	}
	if config.width != 320 || config.height != 240 || config.labels[0] != "zero" || config.inputName != "input" || config.outputName != "output" {
		t.Fatalf("config = %+v", config)
	}
}

func TestModelOptionsRejectInvalidValues(t *testing.T) {
	t.Parallel()
	config := modelConfig{}
	if err := WithImageSize(0, 1)(&config); err == nil {
		t.Fatal("WithImageSize() error = nil")
	}
	if err := WithNormalization([3]float32{float32(math.NaN())}, [3]float32{1, 1, 1})(&config); err == nil {
		t.Fatal("WithNormalization(NaN) error = nil")
	}
	if err := WithNormalization([3]float32{}, [3]float32{})(&config); err == nil {
		t.Fatal("WithNormalization(zero) error = nil")
	}
	if err := WithLabels(nil)(&config); err == nil {
		t.Fatal("WithLabels() error = nil")
	}
	if err := WithTensorNames("", "output")(&config); err == nil {
		t.Fatal("WithTensorNames() error = nil")
	}
}

func TestNilModelClose(t *testing.T) {
	t.Parallel()
	var model *Model
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestClassificationCount(t *testing.T) {
	t.Parallel()
	if count, err := classificationCount([]int64{3, 1_000}, 3, 3_000); err != nil || count != 1_000 {
		t.Fatalf("classificationCount() = %d, %v", count, err)
	}
	if count, err := classificationCount([]int64{1_000}, 1, 1_000); err != nil || count != 1_000 {
		t.Fatalf("classificationCount() = %d, %v", count, err)
	}
	if _, err := classificationCount([]int64{3, 1_000}, 2, 3_000); err == nil {
		t.Fatal("classificationCount() error = nil")
	}
}
