package yolos

import (
	"math"
	"testing"

	"github.com/joeychilson/infergo"
)

func TestModelOptions(t *testing.T) {
	t.Parallel()
	config := modelConfig{}
	if err := WithImageEdges(480, 800)(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithNormalization([3]float32{1, 2, 3}, [3]float32{4, 5, 6})(&config); err != nil {
		t.Fatal(err)
	}
	inputLabels := map[int]string{1: "object"}
	if err := WithLabels(inputLabels)(&config); err != nil {
		t.Fatal(err)
	}
	inputLabels[1] = "changed"
	if err := WithTensorNames("input", "scores", "boxes")(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithSessionOptions(infergo.WithOptimization(infergo.OptimizationBasic))(&config); err != nil {
		t.Fatal(err)
	}
	if config.shortEdge != 480 || config.longEdge != 800 || config.labels[1] != "object" || config.inputName != "input" || config.logitsName != "scores" || config.boxesName != "boxes" {
		t.Fatalf("config = %+v", config)
	}
}

func TestModelOptionsRejectInvalidValues(t *testing.T) {
	t.Parallel()
	config := modelConfig{}
	if err := WithImageEdges(800, 480)(&config); err == nil {
		t.Fatal("WithImageEdges() error = nil")
	}
	if err := WithNormalization([3]float32{float32(math.Inf(1))}, [3]float32{1, 1, 1})(&config); err == nil {
		t.Fatal("WithNormalization(Inf) error = nil")
	}
	if err := WithNormalization([3]float32{}, [3]float32{})(&config); err == nil {
		t.Fatal("WithNormalization(zero) error = nil")
	}
	if err := WithLabels(nil)(&config); err == nil {
		t.Fatal("WithLabels() error = nil")
	}
	if err := WithTensorNames("input", "", "boxes")(&config); err == nil {
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
