package infergo

import (
	"context"
	"encoding/base64"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"sync"
	"testing"
)

func TestOpenRejectsCancelledContext(t *testing.T) {
	t.Parallel()
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := Open(ctx); !errors.Is(err, context.Canceled) {
		t.Fatalf("Open() error = %v, want context.Canceled", err)
	}
}

func TestOpenOfflineCacheMiss(t *testing.T) {
	t.Parallel()
	if _, supported := runtimeArtifacts[currentOS+"/"+currentArch]; !supported {
		t.Skip("automatic runtime is unsupported on this platform")
	}
	_, err := Open(t.Context(), WithCacheDir(t.TempDir()), WithOffline(true))
	if !errors.Is(err, ErrRuntimeNotCached) {
		t.Fatalf("Open() error = %v, want ErrRuntimeNotCached", err)
	}
}

func TestExecutionProviderDevicesRejectsInvalidRuntime(t *testing.T) {
	t.Parallel()
	var runtime *Runtime
	if _, err := runtime.ExecutionProviderDevices(); err == nil {
		t.Fatal("ExecutionProviderDevices() accepted a nil runtime")
	}
	runtime = &Runtime{closed: true}
	if _, err := runtime.ExecutionProviderDevices(); err == nil {
		t.Fatal("ExecutionProviderDevices() accepted a closed runtime")
	}
}

func TestValidateLibraryResolvesSymlinks(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	target := filepath.Join(directory, "runtime")
	link := filepath.Join(directory, "runtime-link")
	if err := os.WriteFile(target, []byte("library"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, link); err != nil {
		t.Skipf("create symlink: %v", err)
	}
	got, err := validateLibrary(link)
	if err != nil {
		t.Fatal(err)
	}
	gotInfo, err := os.Stat(got)
	if err != nil {
		t.Fatal(err)
	}
	targetInfo, err := os.Stat(target)
	if err != nil {
		t.Fatal(err)
	}
	if !os.SameFile(gotInfo, targetInfo) {
		t.Fatalf("validateLibrary() = %q, want same file as %q", got, target)
	}
}

const sumModelBase64 = "CAgSB3B5dG9yY2gaBTIuMS4yOqcCCkUSEW9ubng6OlJlZHVjZVN1bV8xGgpDb25zdGFudF8wIghDb25zdGFudCoaCgV2YWx1ZSoOCAEQB0oIAQAAAAAAAACgAQQKWgoNaW5wdXRfdmVjdG9ycwoRb25ueDo6UmVkdWNlU3VtXzESDm91dHB1dF9zY2FsYXJzGgovUmVkdWNlU3VtIglSZWR1Y2VTdW0qDwoIa2VlcGRpbXMYAKABAhIKbWFpbl9ncmFwaFo7Cg1pbnB1dF92ZWN0b3JzEioKKAgBEiQKHhIcaW5wdXRfdmVjdG9yc19keW5hbWljX2F4ZXNfMQoCCApiOQoOb3V0cHV0X3NjYWxhcnMSJwolCAESIQofEh1vdXRwdXRfc2NhbGFyc19keW5hbWljX2F4ZXNfMUICEBE="

func TestRuntimeIntegration(t *testing.T) {
	if os.Getenv("INFERGO_INTEGRATION") == "" {
		t.Skip("set INFERGO_INTEGRATION=1 to download and load the native runtime")
	}
	cacheDir := os.Getenv("INFERGO_CACHE_DIR")
	if cacheDir == "" {
		cacheDir = t.TempDir()
	}
	runtime, err := Open(t.Context(), WithCacheDir(cacheDir))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if closeErr := runtime.Close(); closeErr != nil {
			t.Error(closeErr)
		}
	})
	version, err := runtime.LoadedVersion()
	if err != nil {
		t.Fatal(err)
	}
	if version != RuntimeVersion {
		t.Fatalf("LoadedVersion() = %q, want %q", version, RuntimeVersion)
	}
	devices, err := runtime.ExecutionProviderDevices()
	if err != nil {
		t.Fatal(err)
	}
	for _, device := range devices {
		if device.Provider == "" {
			t.Fatalf("ExecutionProviderDevices() returned an empty provider: %+v", device)
		}
	}

	modelData, err := base64.StdEncoding.DecodeString(sumModelBase64)
	if err != nil {
		t.Fatal(err)
	}
	modelPath := filepath.Join(t.TempDir(), "sum.onnx")
	if writeErr := os.WriteFile(modelPath, modelData, 0o600); writeErr != nil {
		t.Fatal(writeErr)
	}
	info, err := runtime.Inspect(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(info.Inputs) != 1 || info.Inputs[0].Name != "input_vectors" || info.Inputs[0].Type != DataTypeFloat32 ||
		len(info.Outputs) != 1 || info.Outputs[0].Name != "output_scalars" {
		t.Fatalf("Inspect() = %+v", info)
	}
	session, err := runtime.Load(modelPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(session.Inputs()) != 1 || len(session.Outputs()) != 1 {
		t.Fatalf("session schema = inputs %+v, outputs %+v", session.Inputs(), session.Outputs())
	}
	memorySession, err := runtime.LoadBytes(modelData)
	if err != nil {
		t.Fatal(err)
	}
	input := MustTensor([]int64{1, 10}, make([]float32, 10))
	var runs sync.WaitGroup
	for range 8 {
		runs.Go(func() {
			if _, runErr := memorySession.Run(t.Context(), input); runErr != nil {
				t.Errorf("concurrent Run(): %v", runErr)
			}
		})
	}
	runs.Wait()
	closeErrors := make(chan error, 2)
	go func() { closeErrors <- memorySession.Close() }()
	go func() { closeErrors <- memorySession.Close() }()
	for range 2 {
		if closeErr := <-closeErrors; closeErr != nil {
			t.Fatal(closeErr)
		}
	}
	if _, runErr := session.Run(t.Context(), MustTensor([]int64{2, 9}, make([]float32, 18))); runErr == nil {
		t.Fatal("Run() accepted an invalid fixed dimension")
	}
	t.Cleanup(func() {
		if closeErr := session.Close(); closeErr != nil {
			t.Error(closeErr)
		}
	})
	metadata, err := session.Metadata()
	if err != nil {
		t.Fatal(err)
	}
	if metadata.Producer != "pytorch" || metadata.Graph != "main_graph" {
		t.Fatalf("Metadata() = %+v", metadata)
	}
	if closeErr := runtime.Close(); closeErr != nil {
		t.Fatal(closeErr)
	}
	input = MustTensor([]int64{2, 10}, []float32{
		1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
		11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
	})
	outputs, err := session.RunNamed(t.Context(), map[string]Tensor{"input_vectors": input})
	if err != nil {
		t.Fatal(err)
	}
	values, err := outputs["output_scalars"].Data[float32]()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(values, []float32{55, 155}) {
		t.Fatalf("session output = %v", values)
	}
}
