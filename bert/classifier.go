package bert

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"math"

	"github.com/joeychilson/infergo"
	"github.com/joeychilson/infergo/postprocess"
)

// ClassificationActivation converts classifier logits to scores.
type ClassificationActivation string

// Supported text-classification activations.
const (
	ClassificationSoftmax ClassificationActivation = "softmax"
	ClassificationSigmoid ClassificationActivation = "sigmoid"
	ClassificationRaw     ClassificationActivation = "raw"
)

// ClassifyOptions configures text classification.
type ClassifyOptions struct {
	TopK       int
	MaxLength  int
	MinScore   float32
	Activation ClassificationActivation
}

// Classifier runs sequence-classification BERT-family ONNX models.
type Classifier struct {
	session          *infergo.Session
	tokenizer        *Tokenizer
	labels           map[int]string
	usesTokenTypeIDs bool
}

// NewClassifier loads a sequence-classification model. The standard BERT
// options configure tensor names, token types, tokenization, and execution.
func NewClassifier(
	runtime *infergo.Runtime,
	modelPath string,
	labels map[int]string,
	options ...Option,
) (*Classifier, error) {
	config, err := resolveModelConfig(options)
	if err != nil {
		return nil, err
	}
	if detectionErr := detectTokenTypeIDs(runtime, modelPath, &config); detectionErr != nil {
		return nil, detectionErr
	}
	tokenizer, err := tokenizerFromConfig(config)
	if err != nil {
		return nil, err
	}
	inputNames := []string{config.inputIDsName, config.attentionMaskName}
	if config.tokenTypeIDsName != "" {
		inputNames = append(inputNames, config.tokenTypeIDsName)
	}
	session, err := runtime.NewSession(modelPath, inputNames, []string{config.outputName}, config.sessionOptions...)
	if err != nil {
		return nil, fmt.Errorf("bert: load classifier: %w", err)
	}
	return &Classifier{
		session:          session,
		tokenizer:        tokenizer,
		labels:           maps.Clone(labels),
		usesTokenTypeIDs: config.tokenTypeIDsName != "",
	}, nil
}

// Classify predicts ranked labels for text.
func (m *Classifier) Classify(
	ctx context.Context,
	text string,
	options ClassifyOptions,
) ([]postprocess.Classification, error) {
	results, err := m.ClassifyBatch(ctx, []string{text}, options)
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

// ClassifyBatch predicts ranked labels for a batch in one or more runs.
func (m *Classifier) ClassifyBatch(
	ctx context.Context,
	texts []string,
	options ClassifyOptions,
) ([][]postprocess.Classification, error) {
	if m == nil || m.session == nil {
		return nil, errors.New("bert: classifier is closed")
	}
	if ctx == nil {
		return nil, errors.New("bert: context cannot be nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(texts) == 0 {
		return nil, errors.New("bert: text batch cannot be empty")
	}
	if options.TopK == 0 {
		options.TopK = 5
	}
	if options.MaxLength == 0 {
		options.MaxLength = 512
	}
	if options.Activation == "" {
		options.Activation = ClassificationSoftmax
	}
	if err := validateClassifyOptions(options); err != nil {
		return nil, err
	}

	encoding, err := m.tokenizer.EncodeBatch(texts, options.MaxLength)
	if err != nil {
		return nil, err
	}
	shape := []int64{int64(encoding.BatchSize), int64(encoding.SequenceLength)}
	inputIDs, err := infergo.NewTensor(shape, encoding.IDs)
	if err != nil {
		return nil, err
	}
	attentionMask, err := infergo.NewTensor(shape, encoding.AttentionMask)
	if err != nil {
		return nil, err
	}
	inputs := []infergo.Tensor{inputIDs, attentionMask}
	if m.usesTokenTypeIDs {
		tokenTypes := make([]int64, encoding.BatchSize*encoding.SequenceLength)
		tokenTypeIDs, tensorErr := infergo.NewTensor(shape, tokenTypes)
		if tensorErr != nil {
			return nil, tensorErr
		}
		inputs = append(inputs, tokenTypeIDs)
	}
	outputs, err := m.session.Run(ctx, inputs...)
	if err != nil {
		return nil, fmt.Errorf("bert: run classifier: %w", err)
	}
	logits, err := outputs[0].Data[float32]()
	if err != nil {
		return nil, fmt.Errorf("bert: read classifier logits: %w", err)
	}
	classCount, err := classifierClassCount(outputs[0].Shape(), encoding.BatchSize, len(logits))
	if err != nil {
		return nil, err
	}
	results := make([][]postprocess.Classification, encoding.BatchSize)
	for batchIndex := range results {
		start := batchIndex * classCount
		results[batchIndex], err = postprocess.Classify(logits[start:start+classCount], postprocess.ClassificationOptions{
			Labels:   m.labels,
			TopK:     options.TopK,
			MinScore: options.MinScore,
			Softmax:  options.Activation == ClassificationSoftmax,
			Sigmoid:  options.Activation == ClassificationSigmoid,
		})
		if err != nil {
			return nil, fmt.Errorf("bert: classify batch item %d: %w", batchIndex, err)
		}
	}
	return results, nil
}

// Tokenizer returns the classifier's tokenizer.
func (m *Classifier) Tokenizer() *Tokenizer {
	if m == nil {
		return nil
	}
	return m.tokenizer
}

// Close releases the classifier session.
func (m *Classifier) Close() error {
	if m == nil || m.session == nil {
		return nil
	}
	return m.session.Close()
}

func validateClassifyOptions(options ClassifyOptions) error {
	if options.TopK < 1 {
		return errors.New("bert: TopK must be positive")
	}
	if options.MaxLength < 2 {
		return errors.New("bert: MaxLength must be at least two")
	}
	if math.IsNaN(float64(options.MinScore)) {
		return errors.New("bert: MinScore cannot be NaN")
	}
	if options.Activation != ClassificationSoftmax &&
		options.Activation != ClassificationSigmoid &&
		options.Activation != ClassificationRaw {
		return fmt.Errorf("bert: unsupported classification activation %q", options.Activation)
	}
	if options.Activation != ClassificationRaw && (options.MinScore < 0 || options.MinScore > 1) {
		return errors.New("bert: MinScore must be between 0 and 1 for probability scores")
	}
	return nil
}

func classifierClassCount(shape []int64, batchSize, dataLength int) (int, error) {
	if len(shape) == 1 && batchSize == 1 && shape[0] > 0 && int(shape[0]) == dataLength {
		return dataLength, nil
	}
	if len(shape) != 2 || shape[0] != int64(batchSize) || shape[1] < 1 {
		return 0, fmt.Errorf("bert: classifier returned shape %v, want [%d, classes]", shape, batchSize)
	}
	classCount := int(shape[1])
	if dataLength != batchSize*classCount {
		return 0, fmt.Errorf("bert: classifier returned %d logits, want %d", dataLength, batchSize*classCount)
	}
	return classCount, nil
}
