package infergo

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"os"
	"slices"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// OptimizationLevel controls ONNX graph optimization for a Session.
type OptimizationLevel int

// Supported graph optimization levels.
const (
	OptimizationDisabled OptimizationLevel = iota
	OptimizationBasic
	OptimizationExtended
	OptimizationAll
)

// SessionOption configures a Session.
type SessionOption func(*sessionConfig) error

type sessionConfig struct {
	intraOpThreads int
	interOpThreads int
	optimization   OptimizationLevel
	providers      []func(*ort.SessionOptions) error
}

// Session runs one ONNX model. Run may be called concurrently. Close waits
// for active runs and is safe to call more than once.
type Session struct {
	mu          sync.RWMutex
	active      sync.WaitGroup
	raw         *ort.DynamicAdvancedSession
	release     func() error
	inputNames  []string
	outputNames []string
}

// WithIntraOpThreads sets the number of threads used within an operator.
func WithIntraOpThreads(count int) SessionOption {
	return func(config *sessionConfig) error {
		if count < 1 {
			return errors.New("infergo: intra-op thread count must be positive")
		}
		config.intraOpThreads = count
		return nil
	}
}

// WithInterOpThreads sets the number of threads used across operators.
func WithInterOpThreads(count int) SessionOption {
	return func(config *sessionConfig) error {
		if count < 1 {
			return errors.New("infergo: inter-op thread count must be positive")
		}
		config.interOpThreads = count
		return nil
	}
}

// WithOptimization sets the graph optimization level. The default is
// OptimizationAll.
func WithOptimization(level OptimizationLevel) SessionOption {
	return func(config *sessionConfig) error {
		if level < OptimizationDisabled || level > OptimizationAll {
			return fmt.Errorf("infergo: invalid optimization level %d", level)
		}
		config.optimization = level
		return nil
	}
}

// WithCoreML enables Apple's Core ML execution provider. Provider settings
// use the keys documented by ONNX Runtime.
func WithCoreML(settings map[string]string) SessionOption {
	settings = maps.Clone(settings)
	return func(config *sessionConfig) error {
		config.providers = append(config.providers, func(options *ort.SessionOptions) error {
			if err := options.AppendExecutionProviderCoreMLV2(settings); err != nil {
				return fmt.Errorf("infergo: enable Core ML execution provider: %w", err)
			}
			return nil
		})
		return nil
	}
}

// WithCUDA enables NVIDIA's CUDA execution provider. Provider settings use
// the keys documented by ONNX Runtime. A CUDA-enabled native runtime and its
// dependencies must be supplied with WithLibraryPath.
func WithCUDA(settings map[string]string) SessionOption {
	settings = maps.Clone(settings)
	return func(config *sessionConfig) error {
		config.providers = append(config.providers, func(options *ort.SessionOptions) error {
			providerOptions, err := ort.NewCUDAProviderOptions()
			if err != nil {
				return fmt.Errorf("infergo: create CUDA provider options: %w", err)
			}
			if err := providerOptions.Update(settings); err != nil {
				return errors.Join(fmt.Errorf("infergo: configure CUDA execution provider: %w", err), providerOptions.Destroy())
			}
			appendErr := options.AppendExecutionProviderCUDA(providerOptions)
			destroyErr := providerOptions.Destroy()
			if appendErr != nil {
				return errors.Join(fmt.Errorf("infergo: enable CUDA execution provider: %w", appendErr), destroyErr)
			}
			return destroyErr
		})
		return nil
	}
}

// WithTensorRT enables NVIDIA's TensorRT execution provider. A TensorRT-enabled
// native runtime and its CUDA/TensorRT dependencies must be supplied with
// WithLibraryPath.
func WithTensorRT(settings map[string]string) SessionOption {
	settings = maps.Clone(settings)
	return func(config *sessionConfig) error {
		config.providers = append(config.providers, func(options *ort.SessionOptions) error {
			providerOptions, err := ort.NewTensorRTProviderOptions()
			if err != nil {
				return fmt.Errorf("infergo: create TensorRT provider options: %w", err)
			}
			if err := providerOptions.Update(settings); err != nil {
				return errors.Join(fmt.Errorf("infergo: configure TensorRT execution provider: %w", err), providerOptions.Destroy())
			}
			appendErr := options.AppendExecutionProviderTensorRT(providerOptions)
			destroyErr := providerOptions.Destroy()
			if appendErr != nil {
				return errors.Join(fmt.Errorf("infergo: enable TensorRT execution provider: %w", appendErr), destroyErr)
			}
			return destroyErr
		})
		return nil
	}
}

// WithOpenVINO enables Intel's OpenVINO execution provider. An OpenVINO-enabled
// native runtime and its dependencies must be supplied with WithLibraryPath.
func WithOpenVINO(settings map[string]string) SessionOption {
	settings = maps.Clone(settings)
	return func(config *sessionConfig) error {
		config.providers = append(config.providers, func(options *ort.SessionOptions) error {
			if err := options.AppendExecutionProviderOpenVINO(settings); err != nil {
				return fmt.Errorf("infergo: enable OpenVINO execution provider: %w", err)
			}
			return nil
		})
		return nil
	}
}

// WithExecutionProvider enables a provider through ONNX Runtime's generic
// provider API. This supports providers such as QNN or XNNPACK when they are
// included in the supplied native runtime.
func WithExecutionProvider(name string, settings map[string]string) SessionOption {
	settings = maps.Clone(settings)
	return func(config *sessionConfig) error {
		if name == "" {
			return errors.New("infergo: execution provider name cannot be empty")
		}
		config.providers = append(config.providers, func(options *ort.SessionOptions) error {
			if err := options.AppendExecutionProvider(name, settings); err != nil {
				return fmt.Errorf("infergo: enable %s execution provider: %w", name, err)
			}
			return nil
		})
		return nil
	}
}

// WithDirectML enables Microsoft's DirectML execution provider on deviceID.
// A DirectML-enabled native runtime must be supplied with WithLibraryPath.
func WithDirectML(deviceID int) SessionOption {
	return func(config *sessionConfig) error {
		if deviceID < 0 {
			return errors.New("infergo: DirectML device ID cannot be negative")
		}
		config.providers = append(config.providers, func(options *ort.SessionOptions) error {
			if err := options.AppendExecutionProviderDirectML(deviceID); err != nil {
				return fmt.Errorf("infergo: enable DirectML execution provider: %w", err)
			}
			return nil
		})
		return nil
	}
}

// NewSession loads modelPath with positional input and output names.
func (r *Runtime) NewSession(
	modelPath string,
	inputNames []string,
	outputNames []string,
	options ...SessionOption,
) (*Session, error) {
	if err := validateModel(modelPath); err != nil {
		return nil, err
	}
	if err := validateNames("input", inputNames); err != nil {
		return nil, err
	}
	if err := validateNames("output", outputNames); err != nil {
		return nil, err
	}

	config, err := resolveSessionConfig(options)
	if err != nil {
		return nil, err
	}

	release, err := r.retain()
	if err != nil {
		return nil, err
	}
	session, err := newSession(modelPath, inputNames, outputNames, config)
	if err != nil {
		return nil, errors.Join(err, release())
	}
	session.release = release
	return session, nil
}

// InputNames returns the model inputs in the positional order expected by Run.
func (s *Session) InputNames() []string {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return slices.Clone(s.inputNames)
}

// OutputNames returns the model outputs in the positional order returned by Run.
func (s *Session) OutputNames() []string {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return slices.Clone(s.outputNames)
}

// Run executes the model with positional input tensors. Returned tensors own
// independent Go memory and remain valid after the next run or Close.
func (s *Session) Run(ctx context.Context, inputs ...Tensor) (result []Tensor, resultErr error) {
	if ctx == nil {
		return nil, errors.New("infergo: context cannot be nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if s == nil {
		return nil, errors.New("infergo: nil session")
	}

	s.mu.RLock()
	if s.raw == nil {
		s.mu.RUnlock()
		return nil, errors.New("infergo: session is closed")
	}
	s.active.Add(1)
	raw := s.raw
	inputNames := s.inputNames
	outputNames := s.outputNames
	s.mu.RUnlock()
	defer s.active.Done()

	if len(inputs) != len(inputNames) {
		return nil, fmt.Errorf("infergo: got %d input tensors, want %d", len(inputs), len(inputNames))
	}

	inputValues := make([]ort.Value, len(inputs))
	for index, input := range inputs {
		value, err := toORTValue(input)
		if err != nil {
			return nil, errors.Join(fmt.Errorf("infergo: create input %q: %w", inputNames[index], err), destroyValues(inputValues))
		}
		inputValues[index] = value
	}
	defer func() { resultErr = errors.Join(resultErr, destroyValues(inputValues)) }()

	runOptions, err := ort.NewRunOptions()
	if err != nil {
		return nil, fmt.Errorf("infergo: create run options: %w", err)
	}
	defer func() { resultErr = errors.Join(resultErr, runOptions.Destroy()) }()

	outputValues := make([]ort.Value, len(outputNames))
	err = runWithContext(ctx, raw, inputValues, outputValues, runOptions)
	if err != nil {
		return nil, errors.Join(err, destroyValues(outputValues))
	}
	defer func() { resultErr = errors.Join(resultErr, destroyValues(outputValues)) }()

	outputs := make([]Tensor, len(outputValues))
	for index, value := range outputValues {
		output, err := fromORTValue(value)
		if err != nil {
			return nil, fmt.Errorf("infergo: read output %q: %w", outputNames[index], err)
		}
		outputs[index] = output
	}
	return outputs, nil
}

// RunNamed executes the model using input names and returns outputs keyed by
// name. Every declared input must be present and unknown inputs are rejected.
func (s *Session) RunNamed(ctx context.Context, inputs map[string]Tensor) (map[string]Tensor, error) {
	if s == nil {
		return nil, errors.New("infergo: nil session")
	}
	s.mu.RLock()
	inputNames := slices.Clone(s.inputNames)
	outputNames := slices.Clone(s.outputNames)
	s.mu.RUnlock()
	ordered := make([]Tensor, len(inputNames))
	for index, name := range inputNames {
		input, ok := inputs[name]
		if !ok {
			return nil, fmt.Errorf("infergo: missing input %q", name)
		}
		ordered[index] = input
	}
	known := make(map[string]struct{}, len(inputNames))
	for _, name := range inputNames {
		known[name] = struct{}{}
	}
	unknown := make([]string, 0)
	for name := range inputs {
		if _, ok := known[name]; !ok {
			unknown = append(unknown, name)
		}
	}
	if len(unknown) > 0 {
		slices.Sort(unknown)
		return nil, fmt.Errorf("infergo: unknown input %q", unknown[0])
	}

	outputs, err := s.Run(ctx, ordered...)
	if err != nil {
		return nil, err
	}
	result := make(map[string]Tensor, len(outputs))
	for index, output := range outputs {
		result[outputNames[index]] = output
	}
	return result, nil
}

// Metadata returns descriptive fields embedded in the loaded ONNX model.
func (s *Session) Metadata() (result ModelMetadata, resultErr error) {
	if s == nil {
		return ModelMetadata{}, errors.New("infergo: nil session")
	}
	s.mu.RLock()
	if s.raw == nil {
		s.mu.RUnlock()
		return ModelMetadata{}, errors.New("infergo: session is closed")
	}
	s.active.Add(1)
	raw := s.raw
	s.mu.RUnlock()
	defer s.active.Done()
	metadata, err := raw.GetModelMetadata()
	if err != nil {
		return ModelMetadata{}, fmt.Errorf("infergo: read model metadata: %w", err)
	}
	defer func() { resultErr = errors.Join(resultErr, metadata.Destroy()) }()
	if result.Producer, err = metadata.GetProducerName(); err != nil {
		return ModelMetadata{}, fmt.Errorf("infergo: read model producer: %w", err)
	}
	if result.Graph, err = metadata.GetGraphName(); err != nil {
		return ModelMetadata{}, fmt.Errorf("infergo: read model graph name: %w", err)
	}
	if result.Domain, err = metadata.GetDomain(); err != nil {
		return ModelMetadata{}, fmt.Errorf("infergo: read model domain: %w", err)
	}
	if result.Description, err = metadata.GetDescription(); err != nil {
		return ModelMetadata{}, fmt.Errorf("infergo: read model description: %w", err)
	}
	if result.Version, err = metadata.GetVersion(); err != nil {
		return ModelMetadata{}, fmt.Errorf("infergo: read model version: %w", err)
	}
	keys, err := metadata.GetCustomMetadataMapKeys()
	if err != nil {
		return ModelMetadata{}, fmt.Errorf("infergo: read custom metadata keys: %w", err)
	}
	result.Custom = make(map[string]string, len(keys))
	for _, key := range keys {
		value, found, lookupErr := metadata.LookupCustomMetadataMap(key)
		if lookupErr != nil {
			return ModelMetadata{}, fmt.Errorf("infergo: read custom metadata %q: %w", key, lookupErr)
		}
		if found {
			result.Custom[key] = value
		}
	}
	return result, nil
}

// Close releases the model session.
func (s *Session) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	if s.raw == nil {
		s.mu.Unlock()
		return nil
	}
	raw := s.raw
	s.raw = nil
	release := s.release
	s.release = nil
	s.mu.Unlock()
	s.active.Wait()
	destroyErr := raw.Destroy()
	if release == nil {
		return destroyErr
	}
	return errors.Join(destroyErr, release())
}

func newSession(modelPath string, inputNames, outputNames []string, config sessionConfig) (*Session, error) {
	options, err := newORTSessionOptions(config)
	if err != nil {
		return nil, err
	}

	raw, loadErr := ort.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, options)
	destroyErr := options.Destroy()
	if loadErr != nil {
		return nil, errors.Join(fmt.Errorf("infergo: load model: %w", loadErr), destroyErr)
	}
	if destroyErr != nil {
		return nil, errors.Join(fmt.Errorf("infergo: close session options: %w", destroyErr), raw.Destroy())
	}
	return &Session{
		raw:         raw,
		inputNames:  slices.Clone(inputNames),
		outputNames: slices.Clone(outputNames),
	}, nil
}

func resolveSessionConfig(options []SessionOption) (sessionConfig, error) {
	config := sessionConfig{optimization: OptimizationAll}
	for _, option := range options {
		if option == nil {
			return sessionConfig{}, errors.New("infergo: session option cannot be nil")
		}
		if err := option(&config); err != nil {
			return sessionConfig{}, err
		}
	}
	return config, nil
}

func newORTSessionOptions(config sessionConfig) (*ort.SessionOptions, error) {
	options, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("infergo: create session options: %w", err)
	}
	optimization := [...]ort.GraphOptimizationLevel{
		ort.GraphOptimizationLevelDisableAll,
		ort.GraphOptimizationLevelEnableBasic,
		ort.GraphOptimizationLevelEnableExtended,
		ort.GraphOptimizationLevelEnableAll,
	}[config.optimization]
	if optionErr := options.SetGraphOptimizationLevel(optimization); optionErr != nil {
		return nil, errors.Join(fmt.Errorf("infergo: set graph optimization: %w", optionErr), options.Destroy())
	}
	if config.intraOpThreads > 0 {
		if optionErr := options.SetIntraOpNumThreads(config.intraOpThreads); optionErr != nil {
			return nil, errors.Join(fmt.Errorf("infergo: set intra-op threads: %w", optionErr), options.Destroy())
		}
	}
	if config.interOpThreads > 0 {
		if optionErr := options.SetInterOpNumThreads(config.interOpThreads); optionErr != nil {
			return nil, errors.Join(fmt.Errorf("infergo: set inter-op threads: %w", optionErr), options.Destroy())
		}
	}
	for _, configureProvider := range config.providers {
		if providerErr := configureProvider(options); providerErr != nil {
			return nil, errors.Join(providerErr, options.Destroy())
		}
	}
	return options, nil
}

func validateModel(path string) error {
	if path == "" {
		return errors.New("infergo: model path cannot be empty")
	}
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("infergo: inspect model: %w", err)
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("infergo: model %q is not a regular file", path)
	}
	return nil
}

func validateNames(kind string, names []string) error {
	if len(names) == 0 {
		return fmt.Errorf("infergo: at least one %s name is required", kind)
	}
	seen := make(map[string]struct{}, len(names))
	for index, name := range names {
		if name == "" {
			return fmt.Errorf("infergo: %s name %d is empty", kind, index)
		}
		if _, ok := seen[name]; ok {
			return fmt.Errorf("infergo: duplicate %s name %q", kind, name)
		}
		seen[name] = struct{}{}
	}
	return nil
}

func runWithContext(
	ctx context.Context,
	session *ort.DynamicAdvancedSession,
	inputs []ort.Value,
	outputs []ort.Value,
	options *ort.RunOptions,
) error {
	if ctx.Done() == nil {
		if err := session.RunWithOptions(inputs, outputs, options); err != nil {
			return fmt.Errorf("infergo: run model: %w", err)
		}
		return nil
	}

	stop := context.AfterFunc(ctx, func() { _ = options.Terminate() })
	defer stop()
	runErr := session.RunWithOptions(inputs, outputs, options)
	if err := ctx.Err(); err != nil {
		return errors.Join(err, runErr)
	}
	if runErr != nil {
		return fmt.Errorf("infergo: run model: %w", runErr)
	}
	return nil
}

func toORTValue(tensor Tensor) (ort.Value, error) {
	shape := ort.NewShape(tensor.shape...)
	switch data := tensor.data.(type) {
	case []bool:
		return ort.NewTensor(shape, data)
	case []string:
		value, err := ort.NewStringTensor(shape)
		if err != nil {
			return nil, err
		}
		if err := value.SetContents(data); err != nil {
			return nil, errors.Join(err, value.Destroy())
		}
		return value, nil
	case []float32:
		return ort.NewTensor(shape, data)
	case []float64:
		return ort.NewTensor(shape, data)
	case []int8:
		return ort.NewTensor(shape, data)
	case []int16:
		return ort.NewTensor(shape, data)
	case []int32:
		return ort.NewTensor(shape, data)
	case []int64:
		return ort.NewTensor(shape, data)
	case []uint8:
		return ort.NewTensor(shape, data)
	case []uint16:
		return ort.NewTensor(shape, data)
	case []uint32:
		return ort.NewTensor(shape, data)
	case []uint64:
		return ort.NewTensor(shape, data)
	default:
		return nil, fmt.Errorf("unsupported tensor type %T", tensor.data)
	}
}

func fromORTValue(value ort.Value) (Tensor, error) {
	if value == nil {
		return Tensor{}, errors.New("ONNX Runtime returned a nil value")
	}
	shape := []int64(value.GetShape())
	switch value := value.(type) {
	case *ort.Tensor[bool]:
		return NewTensor(shape, value.GetData())
	case *ort.StringTensor:
		data, err := value.GetContents()
		if err != nil {
			return Tensor{}, err
		}
		return NewTensor(shape, data)
	case *ort.Tensor[float32]:
		return NewTensor(shape, value.GetData())
	case *ort.Tensor[float64]:
		return NewTensor(shape, value.GetData())
	case *ort.Tensor[int8]:
		return NewTensor(shape, value.GetData())
	case *ort.Tensor[int16]:
		return NewTensor(shape, value.GetData())
	case *ort.Tensor[int32]:
		return NewTensor(shape, value.GetData())
	case *ort.Tensor[int64]:
		return NewTensor(shape, value.GetData())
	case *ort.Tensor[uint8]:
		return NewTensor(shape, value.GetData())
	case *ort.Tensor[uint16]:
		return NewTensor(shape, value.GetData())
	case *ort.Tensor[uint32]:
		return NewTensor(shape, value.GetData())
	case *ort.Tensor[uint64]:
		return NewTensor(shape, value.GetData())
	default:
		return Tensor{}, fmt.Errorf("unsupported ONNX value type %T", value)
	}
}

func destroyValues(values []ort.Value) error {
	var result error
	for _, value := range values {
		if value != nil {
			result = errors.Join(result, value.Destroy())
		}
	}
	return result
}
