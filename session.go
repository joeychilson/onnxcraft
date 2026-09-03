package infergo

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"os"
	"runtime"
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

// ExecutionMode controls whether independent graph nodes may execute in
// parallel. Sequential execution is the default.
type ExecutionMode int

// Supported graph execution modes.
const (
	ExecutionSequential ExecutionMode = iota
	ExecutionParallel
)

// LoggingLevel controls ONNX Runtime session log verbosity.
type LoggingLevel int

// Supported session logging levels.
const (
	LoggingVerbose LoggingLevel = iota
	LoggingInfo
	LoggingWarning
	LoggingError
	LoggingFatal
)

// SessionOption configures a Session.
type SessionOption func(*sessionConfig) error

type sessionConfig struct {
	intraOpThreads int
	interOpThreads int
	optimization   OptimizationLevel
	executionMode  ExecutionMode
	executionSet   bool
	directML       bool
	memoryPattern  *bool
	cpuArena       *bool
	logging        *LoggingLevel
	profilePrefix  string
	optimizedModel string
	customOps      []string
	configEntries  map[string]string
	providerNames  map[string]struct{}
	providers      []func(*ort.SessionOptions) error
}

// Session runs one ONNX model. Run may be called concurrently. Close waits
// for active runs and is safe to call more than once.
type Session struct {
	mu          sync.RWMutex
	active      sync.WaitGroup
	runMu       sync.Mutex
	raw         *ort.DynamicAdvancedSession
	release     func() error
	inputNames  []string
	outputNames []string
	inputInfo   []ValueInfo
	outputInfo  []ValueInfo
	serialize   bool
	closeDone   chan struct{}
	closeErr    error
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

// WithExecutionMode sets graph execution to sequential or parallel. Setting
// inter-op threads implicitly selects parallel execution unless this option is
// used explicitly.
func WithExecutionMode(mode ExecutionMode) SessionOption {
	return func(config *sessionConfig) error {
		if mode < ExecutionSequential || mode > ExecutionParallel {
			return fmt.Errorf("infergo: invalid execution mode %d", mode)
		}
		config.executionMode = mode
		config.executionSet = true
		return nil
	}
}

// WithMemoryPattern controls ONNX Runtime memory-pattern optimization.
func WithMemoryPattern(enabled bool) SessionOption {
	return func(config *sessionConfig) error {
		config.memoryPattern = &enabled
		return nil
	}
}

// WithCPUMemoryArena controls ONNX Runtime's CPU memory arena.
func WithCPUMemoryArena(enabled bool) SessionOption {
	return func(config *sessionConfig) error {
		config.cpuArena = &enabled
		return nil
	}
}

// WithLogging sets the ONNX Runtime session log severity threshold.
func WithLogging(level LoggingLevel) SessionOption {
	return func(config *sessionConfig) error {
		if level < LoggingVerbose || level > LoggingFatal {
			return fmt.Errorf("infergo: invalid logging level %d", level)
		}
		config.logging = &level
		return nil
	}
}

// WithProfiling writes ONNX Runtime profiling output using prefix.
func WithProfiling(prefix string) SessionOption {
	return func(config *sessionConfig) error {
		if prefix == "" {
			return errors.New("infergo: profiling prefix cannot be empty")
		}
		config.profilePrefix = prefix
		return nil
	}
}

// WithOptimizedModel writes the optimized graph to path while loading.
func WithOptimizedModel(path string) SessionOption {
	return func(config *sessionConfig) error {
		if path == "" {
			return errors.New("infergo: optimized model path cannot be empty")
		}
		config.optimizedModel = path
		return nil
	}
}

// WithCustomOperators registers a custom-operator shared library.
func WithCustomOperators(path string) SessionOption {
	return func(config *sessionConfig) error {
		if path == "" {
			return errors.New("infergo: custom-operator library path cannot be empty")
		}
		config.customOps = append(config.customOps, path)
		return nil
	}
}

// WithSessionConfig sets an ONNX Runtime session configuration entry.
func WithSessionConfig(key, value string) SessionOption {
	return func(config *sessionConfig) error {
		if key == "" {
			return errors.New("infergo: session configuration key cannot be empty")
		}
		if config.configEntries == nil {
			config.configEntries = make(map[string]string)
		}
		if _, exists := config.configEntries[key]; exists {
			return fmt.Errorf("infergo: duplicate session configuration key %q", key)
		}
		config.configEntries[key] = value
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
		return appendProvider(config, "CoreML", func(options *ort.SessionOptions) error {
			if err := options.AppendExecutionProviderCoreMLV2(settings); err != nil {
				return fmt.Errorf("infergo: enable Core ML execution provider: %w", err)
			}
			return nil
		})
	}
}

// WithCUDA enables NVIDIA's CUDA execution provider. Provider settings use
// the keys documented by ONNX Runtime. A CUDA-enabled native runtime and its
// dependencies must be supplied with WithLibraryPath.
func WithCUDA(settings map[string]string) SessionOption {
	settings = maps.Clone(settings)
	return func(config *sessionConfig) error {
		return appendProvider(config, "CUDA", func(options *ort.SessionOptions) error {
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
	}
}

// WithTensorRT enables NVIDIA's TensorRT execution provider. A TensorRT-enabled
// native runtime and its CUDA/TensorRT dependencies must be supplied with
// WithLibraryPath.
func WithTensorRT(settings map[string]string) SessionOption {
	settings = maps.Clone(settings)
	return func(config *sessionConfig) error {
		return appendProvider(config, "TensorRT", func(options *ort.SessionOptions) error {
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
	}
}

// WithOpenVINO enables Intel's OpenVINO execution provider. An OpenVINO-enabled
// native runtime and its dependencies must be supplied with WithLibraryPath.
func WithOpenVINO(settings map[string]string) SessionOption {
	settings = maps.Clone(settings)
	return func(config *sessionConfig) error {
		return appendProvider(config, "OpenVINO", func(options *ort.SessionOptions) error {
			if err := options.AppendExecutionProviderOpenVINO(settings); err != nil {
				return fmt.Errorf("infergo: enable OpenVINO execution provider: %w", err)
			}
			return nil
		})
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
		return appendProvider(config, name, func(options *ort.SessionOptions) error {
			if err := options.AppendExecutionProvider(name, settings); err != nil {
				return fmt.Errorf("infergo: enable %s execution provider: %w", name, err)
			}
			return nil
		})
	}
}

// WithDirectML enables Microsoft's DirectML execution provider on deviceID.
// A DirectML-enabled native runtime must be supplied with WithLibraryPath.
func WithDirectML(deviceID int) SessionOption {
	return func(config *sessionConfig) error {
		if deviceID < 0 {
			return errors.New("infergo: DirectML device ID cannot be negative")
		}
		config.directML = true
		return appendProvider(config, "DirectML", func(options *ort.SessionOptions) error {
			if err := options.AppendExecutionProviderDirectML(deviceID); err != nil {
				return fmt.Errorf("infergo: enable DirectML execution provider: %w", err)
			}
			return nil
		})
	}
}

func appendProvider(config *sessionConfig, name string, provider func(*ort.SessionOptions) error) error {
	if config.providerNames == nil {
		config.providerNames = make(map[string]struct{})
	}
	if _, exists := config.providerNames[name]; exists {
		return fmt.Errorf("infergo: duplicate execution provider %q", name)
	}
	config.providerNames[name] = struct{}{}
	config.providers = append(config.providers, provider)
	return nil
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

// NewSessionFromInfo loads modelPath using a previously inspected graph
// schema. Inputs and outputs are validated on each run.
func (r *Runtime) NewSessionFromInfo(
	modelPath string,
	info ModelInfo,
	options ...SessionOption,
) (*Session, error) {
	if err := validateModelInfo(info); err != nil {
		return nil, err
	}
	session, err := r.NewSession(modelPath, valueNames(info.Inputs), valueNames(info.Outputs), options...)
	if err != nil {
		return nil, err
	}
	session.mu.Lock()
	session.inputInfo = cloneValueInfo(info.Inputs)
	session.outputInfo = cloneValueInfo(info.Outputs)
	session.mu.Unlock()
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

// Inputs returns the schema used to validate inputs. Sessions constructed with
// NewSession return nil because only names were supplied.
func (s *Session) Inputs() []ValueInfo {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneValueInfo(s.inputInfo)
}

// Outputs returns the schema used to validate outputs. Sessions constructed
// with NewSession return nil because only names were supplied.
func (s *Session) Outputs() []ValueInfo {
	if s == nil {
		return nil
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return cloneValueInfo(s.outputInfo)
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
	inputInfo := s.inputInfo
	outputInfo := s.outputInfo
	s.mu.RUnlock()
	defer s.active.Done()
	if s.serialize {
		s.runMu.Lock()
		defer s.runMu.Unlock()
	}

	if len(inputs) != len(inputNames) {
		return nil, fmt.Errorf("infergo: got %d input tensors, want %d", len(inputs), len(inputNames))
	}
	for index, input := range inputs {
		if len(inputInfo) > 0 {
			if err := validateTensor(input, inputInfo[index]); err != nil {
				return nil, fmt.Errorf("infergo: invalid input %q: %w", inputNames[index], err)
			}
		}
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
		if len(outputInfo) > 0 {
			if err := validateTensor(output, outputInfo[index]); err != nil {
				return nil, fmt.Errorf("infergo: invalid output %q: %w", outputNames[index], err)
			}
		}
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
		done := s.closeDone
		closeErr := s.closeErr
		s.mu.Unlock()
		if done == nil {
			return closeErr
		}
		<-done
		s.mu.RLock()
		defer s.mu.RUnlock()
		return s.closeErr
	}
	raw := s.raw
	s.raw = nil
	release := s.release
	s.release = nil
	done := make(chan struct{})
	s.closeDone = done
	s.mu.Unlock()
	s.active.Wait()
	closeErr := raw.Destroy()
	if release != nil {
		closeErr = errors.Join(closeErr, release())
	}
	s.mu.Lock()
	s.closeErr = closeErr
	close(done)
	s.mu.Unlock()
	return closeErr
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
		serialize:   config.directML,
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
	if config.interOpThreads > 0 && !config.executionSet {
		config.executionMode = ExecutionParallel
	}
	if config.interOpThreads > 0 && config.executionMode != ExecutionParallel {
		return sessionConfig{}, errors.New("infergo: inter-op threads require parallel execution")
	}
	if config.directML {
		if config.executionMode == ExecutionParallel {
			return sessionConfig{}, errors.New("infergo: DirectML requires sequential execution")
		}
		if config.memoryPattern != nil && *config.memoryPattern {
			return sessionConfig{}, errors.New("infergo: DirectML requires memory patterns to be disabled")
		}
		config.executionMode = ExecutionSequential
		disabled := false
		config.memoryPattern = &disabled
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
	executionMode := ort.ExecutionMode(ort.ExecutionModeSequential)
	if config.executionMode == ExecutionParallel {
		executionMode = ort.ExecutionMode(ort.ExecutionModeParallel)
	}
	if optionErr := options.SetExecutionMode(executionMode); optionErr != nil {
		return nil, errors.Join(fmt.Errorf("infergo: set execution mode: %w", optionErr), options.Destroy())
	}
	if config.memoryPattern != nil {
		if optionErr := options.SetMemPattern(*config.memoryPattern); optionErr != nil {
			return nil, errors.Join(fmt.Errorf("infergo: set memory patterns: %w", optionErr), options.Destroy())
		}
	}
	if config.cpuArena != nil {
		if optionErr := options.SetCpuMemArena(*config.cpuArena); optionErr != nil {
			return nil, errors.Join(fmt.Errorf("infergo: set CPU memory arena: %w", optionErr), options.Destroy())
		}
	}
	if config.logging != nil {
		logging := [...]ort.LoggingLevel{
			ort.LoggingLevel(ort.LoggingLevelVerbose),
			ort.LoggingLevel(ort.LoggingLevelInfo),
			ort.LoggingLevel(ort.LoggingLevelWarning),
			ort.LoggingLevel(ort.LoggingLevelError),
			ort.LoggingLevel(ort.LoggingLevelFatal),
		}[*config.logging]
		if optionErr := options.SetLogSeverityLevel(logging); optionErr != nil {
			return nil, errors.Join(fmt.Errorf("infergo: set logging level: %w", optionErr), options.Destroy())
		}
	}
	if config.optimizedModel != "" {
		if optionErr := options.SetOptimizedModelFilePath(config.optimizedModel); optionErr != nil {
			return nil, errors.Join(fmt.Errorf("infergo: set optimized model path: %w", optionErr), options.Destroy())
		}
	}
	if config.profilePrefix != "" {
		if optionErr := options.EnableProfiling(config.profilePrefix); optionErr != nil {
			return nil, errors.Join(fmt.Errorf("infergo: enable profiling: %w", optionErr), options.Destroy())
		}
	}
	for _, path := range config.customOps {
		if optionErr := options.RegisterCustomOpsLibrary(path); optionErr != nil {
			return nil, errors.Join(fmt.Errorf("infergo: register custom operators %q: %w", path, optionErr), options.Destroy())
		}
	}
	configKeys := slices.Sorted(maps.Keys(config.configEntries))
	for _, key := range configKeys {
		if optionErr := options.AddSessionConfigEntry(key, config.configEntries[key]); optionErr != nil {
			return nil, errors.Join(fmt.Errorf("infergo: set session configuration %q: %w", key, optionErr), options.Destroy())
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

func validateTensor(tensor Tensor, expected ValueInfo) error {
	if tensor.Type() != expected.Type {
		return fmt.Errorf("type is %s, want %s", tensor.Type(), expected.Type)
	}
	shape := tensor.shape
	if len(shape) != len(expected.Shape) {
		return fmt.Errorf("rank is %d, want %d", len(shape), len(expected.Shape))
	}
	for index, dimension := range expected.Shape {
		if dimension >= 0 && shape[index] != dimension {
			return fmt.Errorf("dimension %d is %d, want %d", index, shape[index], dimension)
		}
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

	callbackDone := make(chan struct{})
	var terminateErr error
	stop := context.AfterFunc(ctx, func() {
		terminateErr = options.Terminate()
		close(callbackDone)
	})
	runErr := session.RunWithOptions(inputs, outputs, options)
	if !stop() {
		<-callbackDone
	}
	if err := ctx.Err(); err != nil {
		return errors.Join(err, runErr, terminateErr)
	}
	if runErr != nil {
		return errors.Join(fmt.Errorf("infergo: run model: %w", runErr), terminateErr)
	}
	return terminateErr
}

func toORTValue(tensor Tensor) (ort.Value, error) {
	shape := ort.NewShape(tensor.shape...)
	if len(tensor.shape) == 0 {
		return toORTScalar(tensor)
	}
	switch data := tensor.data.(type) {
	case []bool:
		return newPinnedORTTensor(shape, data)
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
		return newPinnedORTTensor(shape, data)
	case []float64:
		return newPinnedORTTensor(shape, data)
	case []int8:
		return newPinnedORTTensor(shape, data)
	case []int16:
		return newPinnedORTTensor(shape, data)
	case []int32:
		return newPinnedORTTensor(shape, data)
	case []int64:
		return newPinnedORTTensor(shape, data)
	case []uint8:
		return newPinnedORTTensor(shape, data)
	case []uint16:
		return newPinnedORTTensor(shape, data)
	case []uint32:
		return newPinnedORTTensor(shape, data)
	case []uint64:
		return newPinnedORTTensor(shape, data)
	default:
		return nil, fmt.Errorf("unsupported tensor type %T", tensor.data)
	}
}

type pinnedORTValue struct {
	ort.Value
	pinner *runtime.Pinner
}

func (v *pinnedORTValue) Destroy() error {
	destroyErr := v.Value.Destroy()
	if v.pinner != nil {
		v.pinner.Unpin()
		v.pinner = nil
	}
	return destroyErr
}

func newPinnedORTTensor[T ort.TensorData](shape ort.Shape, data []T) (ort.Value, error) {
	var pinner *runtime.Pinner
	if len(data) > 0 {
		pinner = &runtime.Pinner{}
		pinner.Pin(&data[0])
	}
	value, err := ort.NewTensor(shape, data)
	if err != nil {
		if pinner != nil {
			pinner.Unpin()
		}
		return nil, err
	}
	runtime.KeepAlive(data)
	return &pinnedORTValue{Value: value, pinner: pinner}, nil
}

func toORTScalar(tensor Tensor) (ort.Value, error) {
	switch data := tensor.data.(type) {
	case []bool:
		return ort.NewScalar(data[0])
	case []string:
		return nil, errors.New("infergo: string scalar inputs are not supported by the ONNX binding")
	case []float32:
		return ort.NewScalar(data[0])
	case []float64:
		return ort.NewScalar(data[0])
	case []int8:
		return ort.NewScalar(data[0])
	case []int16:
		return ort.NewScalar(data[0])
	case []int32:
		return ort.NewScalar(data[0])
	case []int64:
		return ort.NewScalar(data[0])
	case []uint8:
		return ort.NewScalar(data[0])
	case []uint16:
		return ort.NewScalar(data[0])
	case []uint32:
		return ort.NewScalar(data[0])
	case []uint64:
		return ort.NewScalar(data[0])
	default:
		return nil, fmt.Errorf("unsupported scalar type %T", tensor.data)
	}
}

func fromORTValue(value ort.Value) (Tensor, error) {
	if value == nil {
		return Tensor{}, errors.New("ONNX Runtime returned a nil value")
	}
	shape := []int64(value.GetShape())
	switch value := value.(type) {
	case *ort.Tensor[bool]:
		return TakeTensor(shape, value.GetData())
	case *ort.StringTensor:
		data, err := value.GetContents()
		if err != nil {
			return Tensor{}, err
		}
		return TakeTensor(shape, data)
	case *ort.Tensor[float32]:
		return TakeTensor(shape, value.GetData())
	case *ort.Tensor[float64]:
		return TakeTensor(shape, value.GetData())
	case *ort.Tensor[int8]:
		return TakeTensor(shape, value.GetData())
	case *ort.Tensor[int16]:
		return TakeTensor(shape, value.GetData())
	case *ort.Tensor[int32]:
		return TakeTensor(shape, value.GetData())
	case *ort.Tensor[int64]:
		return TakeTensor(shape, value.GetData())
	case *ort.Tensor[uint8]:
		return TakeTensor(shape, value.GetData())
	case *ort.Tensor[uint16]:
		return TakeTensor(shape, value.GetData())
	case *ort.Tensor[uint32]:
		return TakeTensor(shape, value.GetData())
	case *ort.Tensor[uint64]:
		return TakeTensor(shape, value.GetData())
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
