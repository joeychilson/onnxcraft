package infergo

import (
	"encoding/base64"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

const sumModelBase64 = "CAgSB3B5dG9yY2gaBTIuMS4yOqcCCkUSEW9ubng6OlJlZHVjZVN1bV8xGgpDb25zdGFudF8wIghDb25zdGFudCoaCgV2YWx1ZSoOCAEQB0oIAQAAAAAAAACgAQQKWgoNaW5wdXRfdmVjdG9ycwoRb25ueDo6UmVkdWNlU3VtXzESDm91dHB1dF9zY2FsYXJzGgovUmVkdWNlU3VtIglSZWR1Y2VTdW0qDwoIa2VlcGRpbXMYAKABAhIKbWFpbl9ncmFwaFo7Cg1pbnB1dF92ZWN0b3JzEioKKAgBEiQKHhIcaW5wdXRfdmVjdG9yc19keW5hbWljX2F4ZXNfMQoCCApiOQoOb3V0cHV0X3NjYWxhcnMSJwolCAESIQofEh1vdXRwdXRfc2NhbGFyc19keW5hbWljX2F4ZXNfMUICEBE="

func TestRuntimeIntegration(t *testing.T) {
	if os.Getenv("INFERGO_INTEGRATION") == "" {
		t.Skip("set INFERGO_INTEGRATION=1 to download and load the native runtime")
	}
	runtime, err := Open(t.Context(), WithCacheDir(t.TempDir()))
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
	input := MustTensor([]int64{2, 10}, []float32{
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
