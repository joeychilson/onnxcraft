package embedding

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/joeychilson/infergo"
	"github.com/joeychilson/infergo/bert"
)

// Pooling selects how token embeddings become one sentence embedding.
type Pooling string

// Supported token pooling strategies.
const (
	PoolingMean Pooling = "mean"
	PoolingCLS  Pooling = "cls"
)

// Option configures a sentence-embedding Model.
type Option func(*modelConfig) error

type modelConfig struct {
	inputIDsName      string
	attentionMaskName string
	tokenTypeIDsName  string
	outputName        string
	tokenizer         *bert.Tokenizer
	pooling           Pooling
	normalize         bool
	sessionOptions    []infergo.SessionOption
}

// EmbedOptions configures embedding inference.
type EmbedOptions struct {
	MaxLength int
	BatchSize int
}

// Model creates fixed-size sentence vectors from text.
type Model struct {
	session          *infergo.Session
	tokenizer        *bert.Tokenizer
	pooling          Pooling
	normalize        bool
	usesTokenTypeIDs bool
}

// WithTensorNames overrides the input-ID, attention-mask, and output names.
func WithTensorNames(inputIDs, attentionMask, output string) Option {
	return func(config *modelConfig) error {
		if inputIDs == "" || attentionMask == "" || output == "" {
			return errors.New("embedding: tensor names cannot be empty")
		}
		config.inputIDsName = inputIDs
		config.attentionMaskName = attentionMask
		config.outputName = output
		return nil
	}
}

// WithTokenTypeIDs enables a zero-valued token-type input using name.
func WithTokenTypeIDs(name string) Option {
	return func(config *modelConfig) error {
		if name == "" {
			return errors.New("embedding: token-type input name cannot be empty")
		}
		config.tokenTypeIDsName = name
		return nil
	}
}

// WithTokenizer uses tokenizer instead of the embedded uncased tokenizer.
func WithTokenizer(tokenizer *bert.Tokenizer) Option {
	return func(config *modelConfig) error {
		if tokenizer == nil {
			return errors.New("embedding: tokenizer cannot be nil")
		}
		config.tokenizer = tokenizer
		return nil
	}
}

// WithPooling selects token pooling. The default is attention-aware mean
// pooling. Models that directly output [batch, hidden] vectors skip pooling.
func WithPooling(pooling Pooling) Option {
	return func(config *modelConfig) error {
		if pooling != PoolingMean && pooling != PoolingCLS {
			return fmt.Errorf("embedding: unsupported pooling %q", pooling)
		}
		config.pooling = pooling
		return nil
	}
}

// WithNormalization controls L2 normalization. It is enabled by default.
func WithNormalization(enabled bool) Option {
	return func(config *modelConfig) error {
		config.normalize = enabled
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

// New loads a BERT-family feature-extraction model. By default it reads
// last_hidden_state and performs attention-aware mean pooling.
func New(runtime *infergo.Runtime, modelPath string, options ...Option) (*Model, error) {
	config := modelConfig{
		inputIDsName:      "input_ids",
		attentionMaskName: "attention_mask",
		outputName:        "last_hidden_state",
		pooling:           PoolingMean,
		normalize:         true,
	}
	for _, option := range options {
		if option == nil {
			return nil, errors.New("embedding: option cannot be nil")
		}
		if err := option(&config); err != nil {
			return nil, err
		}
	}
	if config.tokenTypeIDsName == "" {
		info, err := runtime.Inspect(modelPath, config.sessionOptions...)
		if err != nil {
			return nil, fmt.Errorf("embedding: inspect model inputs: %w", err)
		}
		for _, input := range info.Inputs {
			if input.Name == "token_type_ids" {
				config.tokenTypeIDsName = input.Name
				break
			}
		}
	}
	tokenizer := config.tokenizer
	if tokenizer == nil {
		var err error
		tokenizer, err = bert.NewTokenizer()
		if err != nil {
			return nil, err
		}
	}
	inputNames := []string{config.inputIDsName, config.attentionMaskName}
	if config.tokenTypeIDsName != "" {
		inputNames = append(inputNames, config.tokenTypeIDsName)
	}
	session, err := runtime.NewSession(modelPath, inputNames, []string{config.outputName}, config.sessionOptions...)
	if err != nil {
		return nil, fmt.Errorf("embedding: load model: %w", err)
	}
	return &Model{
		session:          session,
		tokenizer:        tokenizer,
		pooling:          config.pooling,
		normalize:        config.normalize,
		usesTokenTypeIDs: config.tokenTypeIDsName != "",
	}, nil
}

// Embed creates one vector per input text while preserving input order.
func (m *Model) Embed(ctx context.Context, texts []string, options EmbedOptions) ([][]float32, error) {
	if m == nil || m.session == nil {
		return nil, errors.New("embedding: model is closed")
	}
	if ctx == nil {
		return nil, errors.New("embedding: context cannot be nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	if options.MaxLength == 0 {
		options.MaxLength = 256
	}
	if options.BatchSize == 0 {
		options.BatchSize = 32
	}
	if options.MaxLength < 2 {
		return nil, errors.New("embedding: MaxLength must be at least two")
	}
	if options.BatchSize < 1 {
		return nil, errors.New("embedding: BatchSize must be positive")
	}

	result := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += options.BatchSize {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := min(start+options.BatchSize, len(texts))
		batch, err := m.embedBatch(ctx, texts[start:end], options.MaxLength)
		if err != nil {
			return nil, fmt.Errorf("embedding: batch starting at %d: %w", start, err)
		}
		result = append(result, batch...)
	}
	return result, nil
}

// Tokenizer returns the model's tokenizer.
func (m *Model) Tokenizer() *bert.Tokenizer {
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

// CosineSimilarity returns the cosine similarity of two vectors.
func CosineSimilarity(left, right []float32) (float32, error) {
	if len(left) == 0 || len(left) != len(right) {
		return 0, errors.New("embedding: vectors must be non-empty and have equal length")
	}
	var dot, leftSquared, rightSquared float64
	for index, leftValue := range left {
		rightValue := right[index]
		if math.IsNaN(float64(leftValue)) || math.IsInf(float64(leftValue), 0) ||
			math.IsNaN(float64(rightValue)) || math.IsInf(float64(rightValue), 0) {
			return 0, errors.New("embedding: vector contains a non-finite value")
		}
		dot += float64(leftValue) * float64(rightValue)
		leftSquared += float64(leftValue) * float64(leftValue)
		rightSquared += float64(rightValue) * float64(rightValue)
	}
	if leftSquared == 0 || rightSquared == 0 {
		return 0, errors.New("embedding: cosine similarity is undefined for a zero vector")
	}
	result := dot / math.Sqrt(leftSquared*rightSquared)
	if math.IsNaN(result) || math.IsInf(result, 0) {
		return 0, errors.New("embedding: cosine similarity is not finite")
	}
	return float32(result), nil
}

func (m *Model) embedBatch(ctx context.Context, texts []string, maxLength int) ([][]float32, error) {
	encoding, err := m.tokenizer.EncodeBatchContext(ctx, texts, maxLength)
	if err != nil {
		return nil, err
	}
	shape := []int64{int64(encoding.BatchSize), int64(encoding.SequenceLength)}
	inputIDs, err := infergo.TakeTensor(shape, encoding.IDs)
	if err != nil {
		return nil, err
	}
	attentionMask, err := infergo.TakeTensor(shape, encoding.AttentionMask)
	if err != nil {
		return nil, err
	}
	inputs := []infergo.Tensor{inputIDs, attentionMask}
	if m.usesTokenTypeIDs {
		tokenTypes := make([]int64, encoding.BatchSize*encoding.SequenceLength)
		tokenTypeIDs, tensorErr := infergo.TakeTensor(shape, tokenTypes)
		if tensorErr != nil {
			return nil, tensorErr
		}
		inputs = append(inputs, tokenTypeIDs)
	}
	outputs, err := m.session.Run(ctx, inputs...)
	if err != nil {
		return nil, fmt.Errorf("run model: %w", err)
	}
	values, err := outputs[0].Data[float32]()
	if err != nil {
		return nil, fmt.Errorf("read embeddings: %w", err)
	}
	vectors, err := pool(ctx, values, outputs[0].Shape(), encoding, m.pooling)
	if err != nil {
		return nil, err
	}
	if m.normalize {
		for index := range vectors {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			if err := normalize(vectors[index]); err != nil {
				return nil, fmt.Errorf("normalize embedding %d: %w", index, err)
			}
		}
	}
	return vectors, nil
}

func pool(
	ctx context.Context,
	values []float32,
	shape []int64,
	encoding bert.BatchEncoding,
	pooling Pooling,
) ([][]float32, error) {
	if len(shape) == 2 {
		if shape[0] != int64(encoding.BatchSize) || shape[1] < 1 || len(values) != int(shape[0]*shape[1]) {
			return nil, fmt.Errorf("embedding: model returned invalid sentence embedding shape %v", shape)
		}
		hiddenSize := int(shape[1])
		result := make([][]float32, encoding.BatchSize)
		for batchIndex := range result {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			start := batchIndex * hiddenSize
			result[batchIndex] = slices.Clone(values[start : start+hiddenSize])
		}
		return result, nil
	}
	if len(shape) != 3 || shape[0] != int64(encoding.BatchSize) ||
		shape[1] != int64(encoding.SequenceLength) || shape[2] < 1 ||
		len(values) != int(shape[0]*shape[1]*shape[2]) {
		return nil, fmt.Errorf("embedding: model returned shape %v, want [batch, sequence, hidden]", shape)
	}
	hiddenSize := int(shape[2])
	result := make([][]float32, encoding.BatchSize)
	for batchIndex := range result {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		vector := make([]float32, hiddenSize)
		if pooling == PoolingCLS {
			start := batchIndex * encoding.SequenceLength * hiddenSize
			copy(vector, values[start:start+hiddenSize])
			result[batchIndex] = vector
			continue
		}
		activeTokens := 0
		for tokenIndex := range encoding.SequenceLength {
			maskIndex := batchIndex*encoding.SequenceLength + tokenIndex
			if encoding.AttentionMask[maskIndex] == 0 {
				continue
			}
			activeTokens++
			start := maskIndex * hiddenSize
			for hiddenIndex := range hiddenSize {
				vector[hiddenIndex] += values[start+hiddenIndex]
			}
		}
		if activeTokens == 0 {
			return nil, fmt.Errorf("embedding: batch item %d has no active tokens", batchIndex)
		}
		for hiddenIndex := range vector {
			vector[hiddenIndex] /= float32(activeTokens)
		}
		result[batchIndex] = vector
	}
	return result, nil
}

func normalize(vector []float32) error {
	var squaredMagnitude float64
	for _, value := range vector {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return errors.New("vector contains a non-finite value")
		}
		squaredMagnitude += float64(value) * float64(value)
	}
	if squaredMagnitude == 0 {
		return errors.New("cannot normalize a zero vector")
	}
	inverseMagnitude := float32(1 / math.Sqrt(squaredMagnitude))
	for index := range vector {
		vector[index] *= inverseMagnitude
	}
	return nil
}
