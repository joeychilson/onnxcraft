package embedding

import (
	"math"
	"slices"
	"testing"

	"github.com/joeychilson/infergo/bert"
)

func TestMeanPoolingIgnoresPadding(t *testing.T) {
	t.Parallel()
	encoding := bert.BatchEncoding{
		AttentionMask:  []int64{1, 1, 0, 1, 0, 0},
		BatchSize:      2,
		SequenceLength: 3,
	}
	values := []float32{
		1, 2, 3, 4, 100, 200,
		5, 6, 100, 200, 300, 400,
	}
	result, err := pool(values, []int64{2, 3, 2}, encoding, PoolingMean)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float32{{2, 3}, {5, 6}}
	for index := range want {
		if !slices.Equal(result[index], want[index]) {
			t.Fatalf("result[%d] = %v, want %v", index, result[index], want[index])
		}
	}
}

func TestCLSPooling(t *testing.T) {
	t.Parallel()
	encoding := bert.BatchEncoding{AttentionMask: []int64{1, 1}, BatchSize: 1, SequenceLength: 2}
	result, err := pool([]float32{1, 2, 3, 4}, []int64{1, 2, 2}, encoding, PoolingCLS)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(result[0], []float32{1, 2}) {
		t.Fatalf("result = %v", result)
	}
}

func TestSentenceEmbeddingOutputSkipsPooling(t *testing.T) {
	t.Parallel()
	encoding := bert.BatchEncoding{BatchSize: 2, SequenceLength: 3}
	result, err := pool([]float32{1, 2, 3, 4}, []int64{2, 2}, encoding, PoolingMean)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(result[0], []float32{1, 2}) || !slices.Equal(result[1], []float32{3, 4}) {
		t.Fatalf("result = %v", result)
	}
}

func TestCosineSimilarity(t *testing.T) {
	t.Parallel()
	similarity, err := CosineSimilarity([]float32{1, 0}, []float32{1, 1})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(similarity)-1/math.Sqrt2) > 1e-6 {
		t.Fatalf("similarity = %f", similarity)
	}
	if _, err := CosineSimilarity([]float32{0}, []float32{0}); err == nil {
		t.Fatal("CosineSimilarity() error = nil")
	}
}

func TestNormalize(t *testing.T) {
	t.Parallel()
	vector := []float32{3, 4}
	if err := normalize(vector); err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(vector[0]-0.6)) > 1e-6 || math.Abs(float64(vector[1]-0.8)) > 1e-6 {
		t.Fatalf("vector = %v", vector)
	}
}

func TestOptions(t *testing.T) {
	t.Parallel()
	config := modelConfig{}
	if err := WithTensorNames("ids", "mask", "output")(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithTokenTypeIDs("types")(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithPooling(PoolingCLS)(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithNormalization(false)(&config); err != nil {
		t.Fatal(err)
	}
	if config.inputIDsName != "ids" || config.attentionMaskName != "mask" || config.outputName != "output" || config.tokenTypeIDsName != "types" || config.pooling != PoolingCLS || config.normalize {
		t.Fatalf("config = %+v", config)
	}
	if err := WithPooling(Pooling("invalid"))(&config); err == nil {
		t.Fatal("WithPooling() error = nil")
	}
}
