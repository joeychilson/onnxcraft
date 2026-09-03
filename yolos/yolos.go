package yolos

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
	shortEdge      int
	longEdge       int
	mean           [3]float32
	deviation      [3]float32
	labels         map[int]string
	inputName      string
	logitsName     string
	boxesName      string
	sessionOptions []infergo.SessionOption
}

// DetectOptions configures one detection run.
type DetectOptions struct {
	MinScore      float32
	IoUThreshold  float32
	MaxDetections int
}

// Model is a YOLOS transformer object detector.
type Model struct {
	session *infergo.Session
	config  modelConfig
}

// WithImageEdges sets the shortest target edge and longest allowed edge.
func WithImageEdges(shortest, longest int) Option {
	return func(config *modelConfig) error {
		if shortest < 1 || longest < shortest {
			return errors.New("yolos: longest edge must be at least the positive shortest edge")
		}
		config.shortEdge, config.longEdge = shortest, longest
		return nil
	}
}

// WithNormalization overrides per-channel RGB mean and standard deviation.
func WithNormalization(mean, standardDeviation [3]float32) Option {
	return func(config *modelConfig) error {
		for channel, value := range mean {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return fmt.Errorf("yolos: mean for channel %d must be finite", channel)
			}
		}
		for channel, deviation := range standardDeviation {
			if deviation == 0 || math.IsNaN(float64(deviation)) || math.IsInf(float64(deviation), 0) {
				return fmt.Errorf("yolos: standard deviation for channel %d must be finite and non-zero", channel)
			}
		}
		config.mean, config.deviation = mean, standardDeviation
		return nil
	}
}

// WithLabels overrides the default COCO labels.
func WithLabels(classLabels map[int]string) Option {
	return func(config *modelConfig) error {
		if len(classLabels) == 0 {
			return errors.New("yolos: labels cannot be empty")
		}
		config.labels = maps.Clone(classLabels)
		return nil
	}
}

// WithTensorNames overrides the ONNX input, logits, and boxes names.
func WithTensorNames(input, logits, boxes string) Option {
	return func(config *modelConfig) error {
		if input == "" || logits == "" || boxes == "" {
			return errors.New("yolos: tensor names cannot be empty")
		}
		config.inputName, config.logitsName, config.boxesName = input, logits, boxes
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

// New loads a YOLOS object detector.
func New(runtime *infergo.Runtime, modelPath string, options ...Option) (*Model, error) {
	config := modelConfig{
		shortEdge:  800,
		longEdge:   1333,
		mean:       [3]float32{0.485, 0.456, 0.406},
		deviation:  [3]float32{0.229, 0.224, 0.225},
		labels:     labels.COCOMap(),
		inputName:  "pixel_values",
		logitsName: "logits",
		boxesName:  "pred_boxes",
	}
	for _, option := range options {
		if option == nil {
			return nil, errors.New("yolos: option cannot be nil")
		}
		if err := option(&config); err != nil {
			return nil, err
		}
	}
	session, err := runtime.NewSession(
		modelPath,
		[]string{config.inputName},
		[]string{config.logitsName, config.boxesName},
		config.sessionOptions...,
	)
	if err != nil {
		return nil, fmt.Errorf("yolos: load model: %w", err)
	}
	return &Model{session: session, config: config}, nil
}

// Detect preprocesses source, runs the model, and returns pixel detections.
func (m *Model) Detect(ctx context.Context, source image.Image, options DetectOptions) ([]postprocess.Detection, error) {
	if m == nil || m.session == nil {
		return nil, errors.New("yolos: model is closed")
	}
	if ctx == nil {
		return nil, errors.New("yolos: context cannot be nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if options.IoUThreshold == 0 {
		options.IoUThreshold = 0.45
	}
	if options.MaxDetections == 0 {
		options.MaxDetections = 100
	}
	processed, err := vision.Process(source, vision.Options{
		ShortEdge: m.config.shortEdge,
		LongEdge:  m.config.longEdge,
		Mode:      vision.ResizeShortestEdge,
		Mean:      m.config.mean,
		StdDev:    m.config.deviation,
	})
	if err != nil {
		return nil, fmt.Errorf("yolos: preprocess image: %w", err)
	}
	input, err := infergo.NewTensor(
		[]int64{1, 3, int64(processed.Height), int64(processed.Width)},
		processed.Pixels,
	)
	if err != nil {
		return nil, err
	}
	outputs, err := m.session.Run(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("yolos: run model: %w", err)
	}
	logits, err := infergo.Data[float32](outputs[0])
	if err != nil {
		return nil, fmt.Errorf("yolos: read logits: %w", err)
	}
	boxes, err := infergo.Data[float32](outputs[1])
	if err != nil {
		return nil, fmt.Errorf("yolos: read boxes: %w", err)
	}
	return postprocess.DetectDETR(logits, boxes, processed.OriginalSize, postprocess.DetectionOptions{
		Labels:        m.config.labels,
		MaxDetections: options.MaxDetections,
		MinScore:      options.MinScore,
		IoUThreshold:  options.IoUThreshold,
	})
}

// Close releases the model session.
func (m *Model) Close() error {
	if m == nil || m.session == nil {
		return nil
	}
	return m.session.Close()
}
