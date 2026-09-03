package infergo

import (
	"math"
	"slices"
	"testing"
)

func TestTensorCopiesInputAndOutput(t *testing.T) {
	t.Parallel()
	shape := []int64{1, 3}
	values := []float32{1, 2, 3}
	tensor, err := NewTensor(shape, values)
	if err != nil {
		t.Fatal(err)
	}
	shape[1] = 99
	values[0] = 99

	if got := tensor.Shape(); !slices.Equal(got, []int64{1, 3}) {
		t.Fatalf("Shape() = %v", got)
	}
	got, err := tensor.Data[float32]()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []float32{1, 2, 3}) {
		t.Fatalf("Data() = %v", got)
	}
	got[0] = 100
	again, err := tensor.Data[float32]()
	if err != nil {
		t.Fatal(err)
	}
	if again[0] != 1 {
		t.Fatalf("Data() exposed mutable storage: %v", again)
	}
	wrapped, err := Data[float32](tensor)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(wrapped, []float32{1, 2, 3}) {
		t.Fatalf("package Data() = %v", wrapped)
	}
	if tensor.Type() != DataTypeFloat32 || tensor.Len() != 3 {
		t.Fatalf("type = %q, len = %d", tensor.Type(), tensor.Len())
	}
}

func TestNewTensorRejectsInvalidShapes(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		shape []int64
		data  []float32
	}{
		{name: "negative", shape: []int64{-1}},
		{name: "negative after zero", shape: []int64{0, -1}},
		{name: "overflow", shape: []int64{math.MaxInt64, 2}},
		{name: "mismatch", shape: []int64{2}, data: []float32{1}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			if _, err := NewTensor(test.shape, test.data); err == nil {
				t.Fatal("NewTensor() error = nil")
			}
		})
	}
}

func TestTensorSupportsScalars(t *testing.T) {
	t.Parallel()
	tensor, err := NewTensor([]int64{}, []float32{42})
	if err != nil {
		t.Fatal(err)
	}
	if tensor.Len() != 1 {
		t.Fatalf("Len() = %d, want 1", tensor.Len())
	}
}

func TestTakeTensorAdoptsData(t *testing.T) {
	t.Parallel()
	values := []float32{1, 2}
	tensor, err := TakeTensor([]int64{2}, values)
	if err != nil {
		t.Fatal(err)
	}
	values[0] = 3
	got, err := tensor.Data[float32]()
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(got, []float32{3, 2}) {
		t.Fatalf("Data() = %v", got)
	}
}

func TestTensorSupportsZeroDimensions(t *testing.T) {
	t.Parallel()
	tensor, err := NewTensor([]int64{2, 0, 3}, []int64{})
	if err != nil {
		t.Fatal(err)
	}
	if tensor.Len() != 0 {
		t.Fatalf("Len() = %d, want 0", tensor.Len())
	}
}

func TestDataRejectsWrongType(t *testing.T) {
	t.Parallel()
	tensor := MustTensor([]int64{1}, []int64{42})
	if _, err := tensor.Data[float32](); err == nil {
		t.Fatal("Data[float32]() error = nil")
	}
}

func TestMustTensorPanics(t *testing.T) {
	t.Parallel()
	defer func() {
		if recover() == nil {
			t.Fatal("MustTensor() did not panic")
		}
	}()
	_ = MustTensor([]int64{2}, []float32{1})
}

func FuzzNewTensorShapes(f *testing.F) {
	f.Add(int64(2), int64(3), uint16(6))
	f.Add(int64(0), int64(-1), uint16(0))
	f.Fuzz(func(t *testing.T, first, second int64, length uint16) {
		data := make([]float32, int(length)%1024)
		tensor, err := NewTensor([]int64{first, second}, data)
		if err == nil && tensor.Len() != len(data) {
			t.Fatalf("Len() = %d, data length = %d", tensor.Len(), len(data))
		}
	})
}

func BenchmarkTensorConstruction(b *testing.B) {
	shape := []int64{32, 3, 224, 224}
	data := make([]float32, 32*3*224*224)
	b.Run("copy", func(b *testing.B) {
		for b.Loop() {
			if _, err := NewTensor(shape, data); err != nil {
				b.Fatal(err)
			}
		}
	})
	b.Run("take", func(b *testing.B) {
		for b.Loop() {
			if _, err := TakeTensor(shape, data); err != nil {
				b.Fatal(err)
			}
		}
	})
}
