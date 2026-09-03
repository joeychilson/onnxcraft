package postprocess

import (
	"errors"
	"fmt"
	"math"
	"strconv"

	"github.com/joeychilson/infergo/internal/math32"
)

// Classification is one model class and its score.
type Classification struct {
	Class int
	Label string
	Score float32
}

// ClassificationOptions configures Classify.
type ClassificationOptions struct {
	Labels   map[int]string
	TopK     int
	MinScore float32
	Softmax  bool
	Sigmoid  bool
}

// Classify ranks raw class scores. Missing labels fall back to the numeric
// class index.
func Classify(logits []float32, options ClassificationOptions) ([]Classification, error) {
	if len(logits) == 0 {
		return nil, errors.New("postprocess: classification logits cannot be empty")
	}
	if options.TopK < 1 {
		return nil, errors.New("postprocess: TopK must be positive")
	}
	if options.Softmax && options.Sigmoid {
		return nil, errors.New("postprocess: Softmax and Sigmoid cannot both be enabled")
	}
	if math.IsNaN(float64(options.MinScore)) {
		return nil, errors.New("postprocess: MinScore cannot be NaN")
	}
	if (options.Softmax || options.Sigmoid) && (options.MinScore < 0 || options.MinScore > 1) {
		return nil, errors.New("postprocess: MinScore must be between 0 and 1 when using probabilities")
	}
	for _, logit := range logits {
		if math.IsNaN(float64(logit)) {
			return nil, errors.New("postprocess: classification logits contain NaN")
		}
	}

	scores := make([]float32, len(logits))
	if options.Softmax {
		var err error
		scores, err = math32.SoftmaxInto(scores, logits)
		if err != nil {
			return nil, fmt.Errorf("postprocess: calculate softmax: %w", err)
		}
	}
	if options.Sigmoid {
		copy(scores, logits)
		for index, score := range scores {
			if score >= 0 {
				scores[index] = 1 / (1 + float32(math.Exp(float64(-score))))
				continue
			}
			exponential := float32(math.Exp(float64(score)))
			scores[index] = exponential / (1 + exponential)
		}
	}
	if !options.Softmax && !options.Sigmoid {
		copy(scores, logits)
	}
	indices := math32.TopK(scores, options.TopK)
	classifications := make([]Classification, 0, len(indices))
	for _, class := range indices {
		if scores[class] < options.MinScore {
			continue
		}
		label, ok := options.Labels[class]
		if !ok {
			label = strconv.Itoa(class)
		}
		classifications = append(classifications, Classification{
			Class: class,
			Label: label,
			Score: scores[class],
		})
	}
	return classifications, nil
}
