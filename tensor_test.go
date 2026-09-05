package onnxcraft

import (
	"math"
	"slices"
	"testing"
)

func TestTensorOwnership(t *testing.T) {
	shape, data := []int64{2, 2}, []float32{1, 2, 3, 4}
	tensor, err := NewTensor(shape, data)
	if err != nil {
		t.Fatal(err)
	}
	shape[0] = 99
	gotShape := tensor.Shape()
	gotShape[1] = 99
	if !slices.Equal(tensor.Shape(), []int64{2, 2}) {
		t.Fatal("shape is aliased")
	}
	view, err := tensor.Data[float32]()
	if err != nil {
		t.Fatal(err)
	}
	data[0] = 42
	if view[0] != 42 || &view[0] != &data[0] {
		t.Fatal("data was copied")
	}
	view[1] = 10
	if data[1] != 10 {
		t.Fatal("Data is not a view")
	}
	if _, err := tensor.Data[int32](); err == nil {
		t.Fatal("wrong element type accepted")
	}
	if _, err := (Tensor{}).Data[float32](); err == nil {
		t.Fatal("zero tensor accepted")
	}
}

func TestTensorShape(t *testing.T) {
	for _, tt := range []struct {
		shape []int64
		count int
		valid bool
	}{
		{nil, 1, true}, {nil, 0, false}, {[]int64{0}, 0, true},
		{[]int64{2, 0, 3}, 0, true}, {[]int64{2, 3}, 6, true},
		{[]int64{2, 3}, 5, false}, {[]int64{-1}, 0, false},
		{[]int64{0, -1}, 0, false}, {[]int64{math.MaxInt64, 2}, 0, false},
		{[]int64{math.MaxInt64}, 0, false}, {[]int64{math.MaxInt64, 0}, 0, true},
	} {
		_, err := NewTensor(tt.shape, make([]float64, tt.count))
		if (err == nil) != tt.valid {
			t.Errorf("NewTensor(%v, %d): %v", tt.shape, tt.count, err)
		}
	}
}

func TestNamedElements(t *testing.T) {
	type Score float32
	tensor, err := NewTensor([]int64{2}, []Score{1.5, 2.5})
	if err != nil {
		t.Fatal(err)
	}
	view, err := tensor.Data[float32]()
	if err != nil || !slices.Equal(view, []float32{1.5, 2.5}) {
		t.Fatalf("view: %v, %v", view, err)
	}
	half, err := NewTensor([]int64{1}, []Float16{0x3c00})
	if err != nil {
		t.Fatal(err)
	}
	if half.Type() != Float16Type {
		t.Fatal(half.Type())
	}
	if _, err := half.Data[uint16](); err == nil {
		t.Fatal("float16 accepted as uint16")
	}
	if _, err := half.Data[BFloat16](); err == nil {
		t.Fatal("float16 accepted as bfloat16")
	}
}

func FuzzTensorShape(f *testing.F) {
	f.Add(int64(2), int64(3), uint8(6))
	f.Add(int64(math.MaxInt64), int64(0), uint8(0))
	f.Add(int64(-1), int64(0), uint8(0))
	f.Fuzz(func(t *testing.T, a, b int64, length uint8) {
		data := make([]uint64, int(length))
		tensor, err := NewTensor([]int64{a, b}, data)
		if err != nil {
			return
		}
		if a < 0 || b < 0 {
			t.Fatal("negative dimension accepted")
		}
		if a == 0 || b == 0 {
			if length != 0 {
				t.Fatal("nonempty data accepted for empty shape")
			}
		} else if a > 255 || b > 255 || a*b != int64(length) {
			t.Fatal("invalid shape accepted")
		}
		view, err := tensor.Data[uint64]()
		if err != nil || len(view) != len(data) {
			t.Fatalf("data view: %v", err)
		}
	})
}
