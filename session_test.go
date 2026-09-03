package infergo

import "testing"

func TestValidateNames(t *testing.T) {
	t.Parallel()
	if err := validateNames("input", []string{"left", "right"}); err != nil {
		t.Fatal(err)
	}
	for _, names := range [][]string{nil, {""}, {"value", "value"}} {
		if err := validateNames("input", names); err == nil {
			t.Fatalf("validateNames(%q) error = nil", names)
		}
	}
}

func TestSessionOptions(t *testing.T) {
	t.Parallel()
	config := sessionConfig{}
	if err := WithIntraOpThreads(4)(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithInterOpThreads(2)(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithOptimization(OptimizationExtended)(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithExecutionMode(ExecutionParallel)(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithMemoryPattern(false)(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithCPUMemoryArena(false)(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithLogging(LoggingWarning)(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithProfiling("profile")(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithOptimizedModel("optimized.onnx")(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithCustomOperators("operators.so")(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithSessionConfig("session.test", "value")(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithCoreML(map[string]string{"ModelFormat": "MLProgram"})(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithCUDA(map[string]string{"device_id": "0"})(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithTensorRT(map[string]string{"device_id": "0"})(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithOpenVINO(map[string]string{"device_type": "CPU"})(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithExecutionProvider("XNNPACK", nil)(&config); err != nil {
		t.Fatal(err)
	}
	if err := WithDirectML(0)(&config); err != nil {
		t.Fatal(err)
	}
	if config.intraOpThreads != 4 || config.interOpThreads != 2 || config.optimization != OptimizationExtended ||
		config.executionMode != ExecutionParallel || !config.directML || len(config.providers) != 6 ||
		config.memoryPattern == nil || *config.memoryPattern || config.cpuArena == nil || *config.cpuArena ||
		config.logging == nil || *config.logging != LoggingWarning || config.profilePrefix != "profile" ||
		config.optimizedModel != "optimized.onnx" || len(config.customOps) != 1 || config.configEntries["session.test"] != "value" {
		t.Fatalf("config = %+v", config)
	}
	if err := WithIntraOpThreads(0)(&config); err == nil {
		t.Fatal("WithIntraOpThreads(0) error = nil")
	}
	if err := WithInterOpThreads(0)(&config); err == nil {
		t.Fatal("WithInterOpThreads(0) error = nil")
	}
	if err := WithOptimization(OptimizationLevel(99))(&config); err == nil {
		t.Fatal("WithOptimization(99) error = nil")
	}
	if err := WithExecutionMode(ExecutionMode(99))(&config); err == nil {
		t.Fatal("WithExecutionMode(99) error = nil")
	}
	if err := WithLogging(LoggingLevel(99))(&config); err == nil {
		t.Fatal("WithLogging(99) error = nil")
	}
	if err := WithProfiling("")(&config); err == nil {
		t.Fatal("WithProfiling() error = nil")
	}
	if err := WithOptimizedModel("")(&config); err == nil {
		t.Fatal("WithOptimizedModel() error = nil")
	}
	if err := WithCustomOperators("")(&config); err == nil {
		t.Fatal("WithCustomOperators() error = nil")
	}
	if err := WithSessionConfig("", "value")(&config); err == nil {
		t.Fatal("WithSessionConfig() error = nil")
	}
	if err := WithSessionConfig("session.test", "other")(&config); err == nil {
		t.Fatal("duplicate WithSessionConfig() error = nil")
	}
	if err := WithDirectML(-1)(&config); err == nil {
		t.Fatal("WithDirectML(-1) error = nil")
	}
	if err := WithExecutionProvider("", nil)(&config); err == nil {
		t.Fatal("WithExecutionProvider() error = nil")
	}
}

func TestResolveSessionConfig(t *testing.T) {
	t.Parallel()
	config, err := resolveSessionConfig([]SessionOption{WithInterOpThreads(2)})
	if err != nil {
		t.Fatal(err)
	}
	if config.executionMode != ExecutionParallel {
		t.Fatalf("execution mode = %d, want parallel", config.executionMode)
	}
	if _, err := resolveSessionConfig([]SessionOption{
		WithInterOpThreads(2),
		WithExecutionMode(ExecutionSequential),
	}); err == nil {
		t.Fatal("sequential execution with inter-op threads error = nil")
	}
	if _, err := resolveSessionConfig([]SessionOption{
		WithDirectML(0),
		WithExecutionMode(ExecutionParallel),
	}); err == nil {
		t.Fatal("parallel DirectML error = nil")
	}
	if _, err := resolveSessionConfig([]SessionOption{
		WithDirectML(0),
		WithMemoryPattern(true),
	}); err == nil {
		t.Fatal("DirectML with memory patterns error = nil")
	}
	if _, err := resolveSessionConfig([]SessionOption{
		WithCUDA(nil),
		WithCUDA(nil),
	}); err == nil {
		t.Fatal("duplicate provider error = nil")
	}
}

func TestNilResourcesCloseCleanly(t *testing.T) {
	t.Parallel()
	var runtime *Runtime
	if err := runtime.Close(); err != nil {
		t.Fatal(err)
	}
	var session *Session
	if err := session.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestRunNamedValidatesNames(t *testing.T) {
	t.Parallel()
	session := &Session{inputNames: []string{"input"}, outputNames: []string{"output"}}
	if _, err := session.RunNamed(t.Context(), nil); err == nil {
		t.Fatal("RunNamed() missing-input error = nil")
	}
	input := MustTensor([]int64{1}, []float32{1})
	if _, err := session.RunNamed(t.Context(), map[string]Tensor{"input": input, "extra": input}); err == nil {
		t.Fatal("RunNamed() unknown-input error = nil")
	}
}
