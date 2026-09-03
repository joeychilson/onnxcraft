package bert

import (
	"testing"

	"github.com/joeychilson/infergo"
)

func TestModelOptions(t *testing.T) {
	t.Parallel()
	config := modelConfig{}
	if err := WithTensorNames("ids", "mask", "scores")(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithTokenTypeIDs("types")(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithSessionOptions(infergo.WithOptimization(infergo.OptimizationBasic))(&config); err != nil {
		t.Fatal(err)
	}
	if config.inputIDsName != "ids" || config.attentionMaskName != "mask" || config.tokenTypeIDsName != "types" || config.outputName != "scores" {
		t.Fatalf("config = %+v", config)
	}
	if len(config.sessionOptions) != 1 {
		t.Fatalf("session options = %d", len(config.sessionOptions))
	}
}

func TestModelOptionsRejectEmptyNames(t *testing.T) {
	t.Parallel()
	config := modelConfig{}
	if err := WithTensorNames("", "mask", "scores")(&config); err == nil {
		t.Fatal("WithTensorNames() error = nil")
	}
	if err := WithTokenTypeIDs("")(&config); err == nil {
		t.Fatal("WithTokenTypeIDs() error = nil")
	}
}

func TestNilModelClose(t *testing.T) {
	t.Parallel()
	var model *Model
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
}
