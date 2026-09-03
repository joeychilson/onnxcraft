package bert

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/joeychilson/infergo"
	"github.com/joeychilson/infergo/postprocess"
)

// Option configures a BERT-family model.
type Option func(*modelConfig) error

type modelConfig struct {
	inputIDsName      string
	attentionMaskName string
	tokenTypeIDsName  string
	outputName        string
	tokenizer         *Tokenizer
	sessionOptions    []infergo.SessionOption
}

// FillMaskOptions configures masked-token prediction.
type FillMaskOptions struct {
	TopK           int
	MaxLength      int
	MinScore       float32
	PadToMaxLength bool
}

// MaskPrediction contains ranked candidates for one mask position.
type MaskPrediction struct {
	Position    int
	Predictions []postprocess.Classification
}

// Model is an uncased BERT-family masked-language model.
type Model struct {
	session          *infergo.Session
	tokenizer        *Tokenizer
	labels           map[int]string
	usesTokenTypeIDs bool
}

// WithTensorNames overrides the ONNX input and output names.
func WithTensorNames(inputIDs, attentionMask, output string) Option {
	return func(config *modelConfig) error {
		if inputIDs == "" || attentionMask == "" || output == "" {
			return errors.New("bert: tensor names cannot be empty")
		}
		config.inputIDsName = inputIDs
		config.attentionMaskName = attentionMask
		config.outputName = output
		return nil
	}
}

// WithTokenTypeIDs enables a zero-valued token-type input for models that
// require one, using name as its ONNX input name.
func WithTokenTypeIDs(name string) Option {
	return func(config *modelConfig) error {
		if name == "" {
			return errors.New("bert: token-type input name cannot be empty")
		}
		config.tokenTypeIDsName = name
		return nil
	}
}

// WithSessionOptions applies low-level ONNX session options.
func WithSessionOptions(options ...infergo.SessionOption) Option {
	return func(config *modelConfig) error {
		config.sessionOptions = append(config.sessionOptions, options...)
		return nil
	}
}

// WithTokenizer uses tokenizer instead of the embedded uncased tokenizer.
func WithTokenizer(tokenizer *Tokenizer) Option {
	return func(config *modelConfig) error {
		if tokenizer == nil {
			return errors.New("bert: tokenizer cannot be nil")
		}
		config.tokenizer = tokenizer
		return nil
	}
}

// New loads a masked-language model using the embedded uncased vocabulary.
func New(runtime *infergo.Runtime, modelPath string, options ...Option) (*Model, error) {
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
	session, err := runtime.NewSession(
		modelPath,
		inputNames,
		[]string{config.outputName},
		config.sessionOptions...,
	)
	if err != nil {
		return nil, fmt.Errorf("bert: load model: %w", err)
	}
	return &Model{
		session:          session,
		tokenizer:        tokenizer,
		labels:           tokenizer.labels(),
		usesTokenTypeIDs: config.tokenTypeIDsName != "",
	}, nil
}

func resolveModelConfig(options []Option) (modelConfig, error) {
	config := modelConfig{
		inputIDsName:      "input_ids",
		attentionMaskName: "attention_mask",
		outputName:        "logits",
	}
	for _, option := range options {
		if option == nil {
			return modelConfig{}, errors.New("bert: option cannot be nil")
		}
		if err := option(&config); err != nil {
			return modelConfig{}, err
		}
	}
	return config, nil
}

func tokenizerFromConfig(config modelConfig) (*Tokenizer, error) {
	if config.tokenizer != nil {
		return config.tokenizer, nil
	}
	return NewTokenizer()
}

func detectTokenTypeIDs(runtime *infergo.Runtime, modelPath string, config *modelConfig) error {
	if config.tokenTypeIDsName != "" {
		return nil
	}
	info, err := runtime.Inspect(modelPath, config.sessionOptions...)
	if err != nil {
		return fmt.Errorf("bert: inspect model inputs: %w", err)
	}
	for _, input := range info.Inputs {
		if input.Name == "token_type_ids" {
			config.tokenTypeIDsName = input.Name
			break
		}
	}
	return nil
}

// FillMask predicts replacements for every [MASK] token in text.
func (m *Model) FillMask(ctx context.Context, text string, options FillMaskOptions) ([]MaskPrediction, error) {
	if m == nil || m.session == nil {
		return nil, errors.New("bert: model is closed")
	}
	if ctx == nil {
		return nil, errors.New("bert: context cannot be nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if options.TopK == 0 {
		options.TopK = 5
	}
	if options.MaxLength == 0 {
		options.MaxLength = 512
	}
	if options.TopK < 1 {
		return nil, errors.New("bert: TopK must be positive")
	}
	if options.MinScore < 0 || options.MinScore > 1 || math.IsNaN(float64(options.MinScore)) {
		return nil, errors.New("bert: MinScore must be between 0 and 1")
	}
	var encoding Encoding
	var err error
	if options.PadToMaxLength {
		encoding, err = m.tokenizer.EncodePadded(text, options.MaxLength)
	} else {
		encoding, err = m.tokenizer.Encode(text, options.MaxLength)
	}
	if err != nil {
		return nil, err
	}
	positions := m.tokenizer.MaskPositions(encoding)
	if len(positions) == 0 {
		return nil, errors.New("bert: input does not contain a [MASK] token")
	}

	shape := []int64{1, int64(len(encoding.IDs))}
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
		tokenTypeIDs, tensorErr := infergo.NewTensor(shape, make([]int64, len(encoding.IDs)))
		if tensorErr != nil {
			return nil, tensorErr
		}
		inputs = append(inputs, tokenTypeIDs)
	}
	outputs, err := m.session.Run(ctx, inputs...)
	if err != nil {
		return nil, fmt.Errorf("bert: run model: %w", err)
	}
	logits, err := infergo.Data[float32](outputs[0])
	if err != nil {
		return nil, fmt.Errorf("bert: read logits: %w", err)
	}
	vocabularySize := m.tokenizer.VocabularySize()
	wantLogits := len(encoding.Tokens) * vocabularySize
	if len(logits) != wantLogits {
		return nil, fmt.Errorf("bert: model returned %d logits, want %d", len(logits), wantLogits)
	}

	results := make([]MaskPrediction, 0, len(positions))
	for _, position := range positions {
		start := position * vocabularySize
		predictions, err := postprocess.Classify(logits[start:start+vocabularySize], postprocess.ClassificationOptions{
			Labels:   m.labels,
			TopK:     options.TopK,
			MinScore: options.MinScore,
			Softmax:  true,
		})
		if err != nil {
			return nil, fmt.Errorf("bert: classify mask at position %d: %w", position, err)
		}
		results = append(results, MaskPrediction{Position: position, Predictions: predictions})
	}
	return results, nil
}

// Tokenizer returns the model's tokenizer.
func (m *Model) Tokenizer() *Tokenizer {
	if m == nil {
		return nil
	}
	return m.tokenizer
}

// Close releases the model session.
func (m *Model) Close() error {
	if m == nil || m.session == nil {
		return nil
	}
	return m.session.Close()
}
