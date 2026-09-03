package resnet

import (
	"context"
	"errors"
	"fmt"
	"image"
	"maps"
	"math"

	"github.com/joeychilson/infergo"
	"github.com/joeychilson/infergo/labels"
	"github.com/joeychilson/infergo/postprocess"
	"github.com/joeychilson/infergo/vision"
)

// Option configures a Model.
type Option func(*modelConfig) error

type modelConfig struct {
	width          int
	height         int
	resizeEdge     int
	interpolation  vision.Interpolation
	mean           [3]float32
	deviation      [3]float32
	labels         map[int]string
	inputName      string
	outputName     string
	sessionOptions []infergo.SessionOption
}

// ClassifyOptions configures one classification.
type ClassifyOptions struct {
	TopK     int
	MinScore float32
}

// Model is a ResNet-family image classifier.
type Model struct {
	session *infergo.Session
	config  modelConfig
}

// WithImageSize overrides the square or rectangular model input size.
func WithImageSize(width, height int) Option {
	return func(config *modelConfig) error {
		if width < 1 || height < 1 {
			return errors.New("resnet: image dimensions must be positive")
		}
		config.width, config.height = width, height
		return nil
	}
}

// WithResize sets the shortest edge used before the center crop and the
// interpolation filter. The default is 256 pixels with bicubic filtering.
func WithResize(shortestEdge int, interpolation vision.Interpolation) Option {
	return func(config *modelConfig) error {
		if shortestEdge < 1 {
			return errors.New("resnet: resize edge must be positive")
		}
		if interpolation < vision.InterpolationBilinear || interpolation > vision.InterpolationNearest {
			return fmt.Errorf("resnet: unsupported interpolation %d", interpolation)
		}
		config.resizeEdge = shortestEdge
		config.interpolation = interpolation
		return nil
	}
}

// WithNormalization overrides per-channel RGB mean and standard deviation.
func WithNormalization(mean, standardDeviation [3]float32) Option {
	return func(config *modelConfig) error {
		for channel, value := range mean {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("resnet: mean for channel %d must be finite", channel)
			}
		}
		for channel, deviation := range standardDeviation {
			if deviation == 0 || math.IsNaN(float64(deviation)) || math.IsInf(float64(deviation), 0) {
				return fmt.Errorf("resnet: standard deviation for channel %d must be finite and non-zero", channel)
			}
		}
		config.mean, config.deviation = mean, standardDeviation
		return nil
	}
}

// WithLabels overrides the default ImageNet-1K labels.
func WithLabels(classLabels map[int]string) Option {
	return func(config *modelConfig) error {
		if len(classLabels) == 0 {
			return errors.New("resnet: labels cannot be empty")
		}
		config.labels = maps.Clone(classLabels)
		return nil
	}
}

// WithTensorNames overrides the ONNX input and output names.
func WithTensorNames(input, output string) Option {
	return func(config *modelConfig) error {
		if input == "" || output == "" {
			return errors.New("resnet: tensor names cannot be empty")
		}
		config.inputName, config.outputName = input, output
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

// New loads a ResNet-family image classifier.
func New(runtime *infergo.Runtime, modelPath string, options ...Option) (*Model, error) {
	config := modelConfig{
		width:         224,
		height:        224,
		resizeEdge:    256,
		interpolation: vision.InterpolationBicubic,
		mean:          [3]float32{0.485, 0.456, 0.406},
		deviation:     [3]float32{0.229, 0.224, 0.225},
		labels:        labels.ImageNetMap(),
		inputName:     "pixel_values",
		outputName:    "logits",
	}
	for _, option := range options {
		if option == nil {
			return nil, errors.New("resnet: option cannot be nil")
		}
		if err := option(&config); err != nil {
			return nil, err
		}
	}
	session, err := runtime.NewSession(modelPath, []string{config.inputName}, []string{config.outputName}, config.sessionOptions...)
	if err != nil {
		return nil, fmt.Errorf("resnet: load model: %w", err)
	}
	return &Model{session: session, config: config}, nil
}

// Classify preprocesses source, runs the model, and ranks its predictions.
func (m *Model) Classify(ctx context.Context, source image.Image, options ClassifyOptions) ([]postprocess.Classification, error) {
	results, err := m.ClassifyBatch(ctx, []image.Image{source}, options)
	if err != nil {
		return nil, err
	}
	return results[0], nil
}

// ClassifyBatch preprocesses and classifies images in one model run while
// preserving input order.
func (m *Model) ClassifyBatch(
	ctx context.Context,
	sources []image.Image,
	options ClassifyOptions,
) ([][]postprocess.Classification, error) {
	if m == nil || m.session == nil {
		return nil, errors.New("resnet: model is closed")
	}
	if ctx == nil {
		return nil, errors.New("resnet: context cannot be nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(sources) == 0 {
		return nil, errors.New("resnet: image batch cannot be empty")
	}
	if options.TopK == 0 {
		options.TopK = 5
	}
	if options.TopK < 1 {
		return nil, errors.New("resnet: TopK must be positive")
	}
	if options.MinScore < 0 || options.MinScore > 1 || math.IsNaN(float64(options.MinScore)) {
		return nil, errors.New("resnet: MinScore must be between 0 and 1")
	}
	pixelsPerImage := 3 * m.config.width * m.config.height
	if len(sources) > math.MaxInt/pixelsPerImage {
		return nil, errors.New("resnet: image batch is too large")
	}
	pixels := make([]float32, len(sources)*pixelsPerImage)
	for index, source := range sources {
		start := index * pixelsPerImage
		_, err := vision.ProcessIntoContext(ctx, source, vision.Options{
			Width:         m.config.width,
			Height:        m.config.height,
			ShortEdge:     m.config.resizeEdge,
			LongEdge:      math.MaxInt,
			Mode:          vision.ResizeShortestEdge,
			Interpolation: m.config.interpolation,
			Mean:          m.config.mean,
			StdDev:        m.config.deviation,
			CenterCrop:    true,
		}, pixels[start:start+pixelsPerImage])
		if err != nil {
			return nil, fmt.Errorf("resnet: preprocess image %d: %w", index, err)
		}
	}
	input, err := infergo.TakeTensor(
		[]int64{int64(len(sources)), 3, int64(m.config.height), int64(m.config.width)},
		pixels,
	)
	if err != nil {
		return nil, err
	}
	outputs, err := m.session.Run(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("resnet: run model: %w", err)
	}
	logits, err := infergo.BorrowData[float32](outputs[0])
	if err != nil {
		return nil, fmt.Errorf("resnet: read logits: %w", err)
	}
	classCount, err := classificationCount(outputs[0].Shape(), len(sources), len(logits))
	if err != nil {
		return nil, err
	}
	results := make([][]postprocess.Classification, len(sources))
	for index := range results {
		start := index * classCount
		results[index], err = postprocess.Classify(logits[start:start+classCount], postprocess.ClassificationOptions{
			Labels:   m.config.labels,
			TopK:     options.TopK,
			MinScore: options.MinScore,
			Softmax:  true,
		})
		if err != nil {
			return nil, fmt.Errorf("resnet: classify image %d: %w", index, err)
		}
	}
	return results, nil
}

// Close releases the model session.
func (m *Model) Close() error {
	if m == nil || m.session == nil {
		return nil
	}
	return m.session.Close()
}

func classificationCount(shape []int64, batchSize, dataLength int) (int, error) {
	if len(shape) == 1 && batchSize == 1 && shape[0] > 0 && int(shape[0]) == dataLength {
		return dataLength, nil
	}
	if len(shape) != 2 || shape[0] != int64(batchSize) || shape[1] < 1 {
		return 0, fmt.Errorf("resnet: model returned shape %v, want [%d, classes]", shape, batchSize)
	}
	classCount := int(shape[1])
	if dataLength != batchSize*classCount {
		return 0, fmt.Errorf("resnet: model returned %d logits, want %d", dataLength, batchSize*classCount)
	}
	return classCount, nil
}
