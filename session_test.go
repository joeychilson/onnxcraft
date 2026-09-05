package onnxcraft

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"sync"
	"testing"
	"time"
)

func openRuntime(tb testing.TB) *Runtime {
	tb.Helper()
	path := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")
	if path == "" {
		tb.Skip("set ONNXRUNTIME_SHARED_LIBRARY_PATH to run native integration tests")
	}
	r, err := Open(path)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() {
		if err := r.Close(); err != nil {
			tb.Error(err)
		}
	})
	return r
}

func loadSession(tb testing.TB, r *Runtime, model string) *Session {
	tb.Helper()
	s, err := r.Load(filepath.Join("testdata", model+".onnx"), &SessionOptions{IntraOpThreads: 1})
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() {
		if err := s.Close(); err != nil {
			tb.Error(err)
		}
	})
	return s
}

func tensor[T Element](tb testing.TB, shape []int64, values []T) Tensor {
	tb.Helper()
	x, err := NewTensor(shape, values)
	if err != nil {
		tb.Fatal(err)
	}
	return x
}

func checkData[T Element](tb testing.TB, x Tensor, want []T) {
	tb.Helper()
	got, err := x.Data[T]()
	if err != nil || !slices.Equal(got, want) {
		tb.Fatalf("data = %v, want %v; error: %v", got, want, err)
	}
}

func TestInference(t *testing.T) {
	r := openRuntime(t)
	s := loadSession(t, r, "add")
	if r.Version() == "" {
		t.Fatal("missing version")
	}
	info := s.Inputs()
	if len(info) != 2 || info[0].Name != "a" || info[0].Type != Float32 || !slices.Equal(info[0].Shape, []int64{-1, 3}) {
		t.Fatalf("inputs = %+v", info)
	}
	info[0].Shape[0], info[0].Name = 99, "changed"
	if s.Inputs()[0].Shape[0] != -1 || s.Inputs()[0].Name != "a" {
		t.Fatal("metadata aliased")
	}
	for _, batch := range []int{1, 2, 0, 4} {
		shape := []int64{int64(batch), 3}
		a, b, want := make([]float32, batch*3), make([]float32, batch*3), make([]float32, batch*3)
		for i := range a {
			a[i], b[i], want[i] = float32(i), 2, float32(i)+2
		}
		x, y := tensor(t, shape, a), tensor(t, shape, b)
		out, err := s.Run(t.Context(), x, y)
		if err != nil {
			t.Fatal(err)
		}
		checkData(t, out[0], want)
		if !slices.Equal(out[0].Shape(), shape) {
			t.Fatalf("shape = %v", out[0].Shape())
		}
		clear(a)
		checkData(t, out[0], want)
		buffer := tensor(t, shape, make([]float32, batch*3))
		for range 3 {
			if err := s.RunInto(t.Context(), []Tensor{buffer}, x, y); err != nil {
				t.Fatal(err)
			}
			checkData(t, buffer, b)
		}
	}
}

func identity[T Element](t *testing.T, r *Runtime, name string, values []T) {
	t.Helper()
	s := loadSession(t, r, "identity_"+name)
	input := tensor(t, []int64{int64(len(values))}, values)
	output, err := s.Run(t.Context(), input)
	if err != nil {
		t.Fatal(err)
	}
	checkData(t, output[0], values)
	buffer := tensor(t, input.Shape(), make([]T, len(values)))
	if err := s.RunInto(t.Context(), []Tensor{buffer}, input); err != nil {
		t.Fatal(err)
	}
	checkData(t, buffer, values)
	if len(values) > 0 {
		view, _ := output[0].Data[T]()
		if &view[0] == &values[0] {
			t.Fatal("Run result aliases input")
		}
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	runtime.GC()
	checkData(t, output[0], values)
}

func TestElementTypes(t *testing.T) {
	r := openRuntime(t)
	identity(t, r, "float32", []float32{1.25, -2.5})
	identity(t, r, "float64", []float64{1.25, -2.5})
	identity(t, r, "int8", []int8{-128, 127})
	identity(t, r, "uint8", []uint8{0, 255})
	identity(t, r, "int16", []int16{-32768, 32767})
	identity(t, r, "uint16", []uint16{0, 65535})
	identity(t, r, "int32", []int32{-2147483648, 2147483647})
	identity(t, r, "uint32", []uint32{0, 4294967295})
	identity(t, r, "int64", []int64{-9223372036854775808, 9223372036854775807})
	identity(t, r, "uint64", []uint64{0, 18446744073709551615})
	identity(t, r, "bool", []bool{true, false, true})
	identity(t, r, "float16", []Float16{0x3c00, 0xc000})
	identity(t, r, "bfloat16", []BFloat16{0x3f80, 0xc000})
	identity(t, r, "float32", []float32{})
	s := loadSession(t, r, "scalar")
	out, err := s.Run(t.Context(), tensor(t, nil, []float32{42}))
	if err != nil {
		t.Fatal(err)
	}
	if len(out[0].Shape()) != 0 {
		t.Fatal(out[0].Shape())
	}
	checkData(t, out[0], []float32{42})
}

func TestMixedAndManyOutputs(t *testing.T) {
	r := openRuntime(t)
	s := loadSession(t, r, "mixed")
	x := tensor(t, []int64{2}, []float32{1, 2})
	i := tensor(t, []int64{2}, []int64{3, 4})
	out, err := s.Run(t.Context(), x, i)
	if err != nil {
		t.Fatal(err)
	}
	checkData(t, out[0], []float32{1, 2})
	checkData(t, out[1], []int64{3, 4})
	s = loadSession(t, r, "fanout")
	out, err = s.Run(t.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 20 {
		t.Fatal(len(out))
	}
	for _, y := range out {
		checkData(t, y, []float32{1, 2})
	}
	view, _ := out[0].Data[float32]()
	view[0] = 99
	checkData(t, out[1], []float32{1, 2})
	if err := s.RunInto(t.Context(), out, x); err != nil {
		t.Fatal(err)
	}
	for _, y := range out {
		checkData(t, y, []float32{1, 2})
	}
}

func TestConstantAndScalarBuffers(t *testing.T) {
	r := openRuntime(t)
	s := loadSession(t, r, "constant")
	if len(s.Inputs()) != 0 {
		t.Fatal(s.Inputs())
	}
	out, err := s.Run(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	checkData(t, out[0], []float32{1, 2, 3})
	if err := s.RunInto(t.Context(), out); err != nil {
		t.Fatal(err)
	}
	checkData(t, out[0], []float32{1, 2, 3})
	s = loadSession(t, r, "scalar")
	x := tensor(t, nil, []float32{42})
	y := tensor(t, nil, make([]float32, 1))
	if err := s.RunInto(t.Context(), []Tensor{y}, x); err != nil {
		t.Fatal(err)
	}
	checkData(t, y, []float32{42})
}

func TestInvalidRuns(t *testing.T) {
	r := openRuntime(t)
	s := loadSession(t, r, "add")
	x := tensor(t, []int64{1, 3}, []float32{1, 2, 3})
	for _, inputs := range [][]Tensor{nil, {x}, {x, x, x}, {Tensor{}, x},
		{tensor(t, []int64{1, 2}, []float32{1, 2}), x},
		{tensor(t, []int64{1, 3}, []int64{1, 2, 3}), x}} {
		if out, err := s.Run(t.Context(), inputs...); err == nil || out != nil {
			t.Fatalf("invalid run returned %v, %v", out, err)
		}
	}
	for _, outputs := range [][]Tensor{nil, {x}, {Tensor{}},
		{tensor(t, []int64{1, 2}, make([]float32, 2))},
		{tensor(t, []int64{1, 3}, make([]int64, 3))}} {
		if err := s.RunInto(t.Context(), outputs, x, x); err == nil {
			t.Fatal("invalid output accepted")
		}
	}
	_, err := s.Run(t.Context(), tensor(t, []int64{1, 3}, []int64{1, 2, 3}), x)
	var native *NativeError
	if !errors.As(err, &native) || native.Code != 2 || native.Message == "" {
		t.Fatalf("native error = %v", err)
	}
	storage := make([]float32, 6)
	a := tensor(t, []int64{1, 3}, storage[:3])
	o := tensor(t, []int64{1, 3}, storage[2:5])
	if err := s.RunInto(t.Context(), []Tensor{o}, a, x); err == nil {
		t.Fatal("partially overlapping storage accepted")
	}
	if _, err := s.Run(t.Context(), x, x); err != nil {
		t.Fatal("failure poisoned session:", err)
	}
}

func TestLoadAndLifetime(t *testing.T) {
	r := openRuntime(t)
	for _, name := range []string{"identity_string", "sequence"} {
		if s, err := r.Load(filepath.Join("testdata", name+".onnx"), &SessionOptions{IntraOpThreads: 1}); err == nil {
			s.Close()
			t.Fatalf("unsupported %s accepted", name)
		}
	}
	for _, model := range [][]byte{nil, []byte("not an onnx model")} {
		if s, err := r.LoadBytes(model, nil); err == nil {
			s.Close()
			t.Fatal("invalid model accepted")
		}
	}
	model, err := os.ReadFile("testdata/add.onnx")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "模型 café.onnx")
	if err := os.WriteFile(path, model, 0600); err != nil {
		t.Fatal(err)
	}
	file, err := r.Load(path, &SessionOptions{IntraOpThreads: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	s, err := r.LoadBytes(model, &SessionOptions{IntraOpThreads: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	clear(model)
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if err := r.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := r.Load("testdata/add.onnx", nil); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	x := tensor(t, []int64{1, 3}, []float32{1, 2, 3})
	out, err := s.Run(t.Context(), x, x)
	if err != nil {
		t.Fatal(err)
	}
	checkData(t, out[0], []float32{2, 4, 6})
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Run(t.Context(), x, x); !errors.Is(err, ErrClosed) {
		t.Fatal(err)
	}
	checkData(t, out[0], []float32{2, 4, 6})
}

func TestConcurrentRunsAndClose(t *testing.T) {
	r := openRuntime(t)
	s := loadSession(t, r, "add")
	x := tensor(t, []int64{1, 3}, []float32{1, 2, 3})
	var wg sync.WaitGroup
	for range 16 {
		wg.Go(func() {
			buffer, err := NewTensor([]int64{1, 3}, make([]float32, 3))
			if err != nil {
				t.Error(err)
				return
			}
			for range 50 {
				if err := s.RunInto(context.Background(), []Tensor{buffer}, x, x); err != nil {
					t.Error(err)
					return
				}
				data, err := buffer.Data[float32]()
				if err != nil || !slices.Equal(data, []float32{2, 4, 6}) {
					t.Errorf("concurrent result: %v, %v", data, err)
					return
				}
			}
		})
	}
	wg.Wait()
	for range 16 {
		wg.Go(func() {
			for range 20 {
				if _, err := s.Run(t.Context(), x, x); err != nil && !errors.Is(err, ErrClosed) {
					t.Error(err)
				}
			}
		})
	}
	wg.Go(func() {
		if err := s.Close(); err != nil {
			t.Error(err)
		}
	})
	wg.Go(func() {
		if err := r.Close(); err != nil {
			t.Error(err)
		}
	})
	wg.Wait()
}

func TestIndependentRuntimes(t *testing.T) {
	r1, r2 := openRuntime(t), openRuntime(t)
	s1, s2 := loadSession(t, r1, "scalar"), loadSession(t, r2, "scalar")
	if err := r1.Close(); err != nil {
		t.Fatal(err)
	}
	if err := s1.Close(); err != nil {
		t.Fatal(err)
	}
	x := tensor(t, nil, []float32{42})
	out, err := s2.Run(t.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	if err := s2.Close(); err != nil {
		t.Fatal(err)
	}
	if err := r2.Close(); err != nil {
		t.Fatal(err)
	}
	runtime.GC()
	checkData(t, out[0], []float32{42})
}

func TestCancellation(t *testing.T) {
	r := openRuntime(t)
	s := loadSession(t, r, "matmul")
	x := tensor(t, []int64{512, 512}, make([]float32, 512*512))
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := s.Run(ctx, x); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	for range 5 {
		ctx, cancel := context.WithTimeout(t.Context(), 2*time.Millisecond)
		out, err := s.Run(ctx, x)
		cancel()
		if !errors.Is(err, context.DeadlineExceeded) || out != nil {
			t.Fatalf("canceled Run: %v, %v", out, err)
		}
	}
	small := tensor(t, []int64{2, 2}, []float32{1, 0, 0, 1})
	out, err := s.Run(t.Context(), small)
	if err != nil {
		t.Fatal("cancellation affected later call:", err)
	}
	checkData(t, out[0], []float32{1, 0, 0, 1})
	ctx, cancel = context.WithTimeout(t.Context(), 2*time.Millisecond)
	defer cancel()
	finished := make(chan error, 1)
	go func() { _, err := s.Run(ctx, x); finished <- err }()
	if _, err := s.Run(t.Context(), small); err != nil {
		t.Fatal("another run's cancellation leaked:", err)
	}
	if err := <-finished; !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
	buffer := tensor(t, x.Shape(), make([]float32, 512*512))
	ctx, cancel = context.WithTimeout(t.Context(), 2*time.Millisecond)
	defer cancel()
	if err := s.RunInto(ctx, []Tensor{buffer}, x); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatal(err)
	}
}

func TestQueuedProviderCancellation(t *testing.T) {
	r := openRuntime(t)
	s := loadSession(t, r, "add")
	// Occupy the same gate used by configured execution providers. This isolates
	// queue cancellation from provider availability and operator timing.
	s.gate = make(chan struct{}, 1)
	s.gate <- struct{}{}
	t.Cleanup(func() { <-s.gate })
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Millisecond)
	defer cancel()
	x := tensor(t, []int64{1, 3}, []float32{1, 2, 3})
	finished := make(chan error, 1)
	go func() { _, err := s.Run(ctx, x, x); finished <- err }()
	select {
	case err := <-finished:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatal(err)
		}
	case <-time.After(time.Second):
		t.Fatal("cancellation waited for another inference call")
	}
}

func TestOptions(t *testing.T) {
	r := openRuntime(t)
	for _, options := range []SessionOptions{
		{IntraOpThreads: -1}, {Optimization: 255},
		{Config: map[string]string{"session.use_ort_model_bytes": "1"}},
		{Config: map[string]string{"bad\x00key": "1"}},
		{Providers: []Provider{{Name: "no-such-provider"}}},
		{Providers: []Provider{{Name: "CUDA", Options: map[string]string{"enable_cuda_graph": "1"}}}},
		{Providers: []Provider{{Name: "CoreML", Options: map[string]string{"bad\x00key": "1"}}}},
	} {
		if s, err := r.Load("testdata/add.onnx", &options); err == nil {
			s.Close()
			t.Fatal("invalid options accepted")
		}
	}
	s, err := r.Load("testdata/add.onnx", &SessionOptions{IntraOpThreads: 1, InterOpThreads: 2, Parallel: true, Optimization: OptimizeBasic, Config: map[string]string{"session.intra_op.allow_spinning": "0"}})
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	x := tensor(t, []int64{1, 3}, []float32{1, 2, 3})
	if _, err := s.Run(t.Context(), x, x); err != nil {
		t.Fatal(err)
	}
}

func TestOpenErrors(t *testing.T) {
	for _, path := range []string{"", "a\x00b", filepath.Join(t.TempDir(), "missing")} {
		if r, err := Open(path); err == nil {
			r.Close()
			t.Fatalf("Open(%q) succeeded", path)
		}
	}
}

func TestSum(t *testing.T) {
	r := openRuntime(t)
	s := loadSession(t, r, "sum")
	data := make([]float32, 20)
	for i := range data {
		data[i] = float32(i + 1)
	}
	x := tensor(t, []int64{2, 10}, data)
	out, err := s.Run(t.Context(), x)
	if err != nil {
		t.Fatal(err)
	}
	checkData(t, out[0], []float32{55, 155})
	buffer := tensor(t, []int64{2}, make([]float32, 2))
	if err := s.RunInto(t.Context(), []Tensor{buffer}, x); err != nil {
		t.Fatal(err)
	}
	checkData(t, buffer, []float32{55, 155})
}

func BenchmarkNativeSession(b *testing.B) {
	r := openRuntime(b)
	model, err := os.ReadFile("testdata/sum.onnx")
	if err != nil {
		b.Fatal(err)
	}
	b.Run("LoadBytes", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			s, err := r.LoadBytes(model, &SessionOptions{IntraOpThreads: 1})
			if err != nil {
				b.Fatal(err)
			}
			s.Close()
		}
	})
	s := loadSession(b, r, "sum")
	x := tensor(b, []int64{1, 10}, make([]float32, 10))
	out := []Tensor{tensor(b, []int64{1}, make([]float32, 1))}
	b.Run("Run", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if _, err := s.Run(context.Background(), x); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("RunInto", func(b *testing.B) {
		b.ReportAllocs()
		for b.Loop() {
			if err := s.RunInto(context.Background(), out, x); err != nil {
				b.Fatal(err)
			}
		}
	})
}
