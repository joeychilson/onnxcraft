package onnxcraft

import (
	"fmt"
	"math"
	"reflect"
	"slices"
	"unsafe"
)

// Float16 holds the IEEE 754 binary16 bits of a tensor element.
type Float16 uint16

// BFloat16 holds the bfloat16 bits of a tensor element.
type BFloat16 uint16

// Element is a fixed-size ONNX tensor element. Named Go types are supported.
// int and uint are excluded because their width depends on the architecture.
type Element interface {
	~float32 | ~float64 | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint8 | ~uint16 | ~uint32 | ~uint64 | ~bool
}

// DataType identifies an ONNX tensor element type.
type DataType uint8

const (
	Float32      DataType = 1
	Uint8        DataType = 2
	Int8         DataType = 3
	Uint16       DataType = 4
	Int16        DataType = 5
	Int32        DataType = 6
	Int64        DataType = 7
	Bool         DataType = 9
	Float16Type  DataType = 10
	Float64      DataType = 11
	Uint32       DataType = 12
	Uint64       DataType = 13
	BFloat16Type DataType = 16
)

func (d DataType) String() string {
	names := [...]string{Float32: "float32", Uint8: "uint8", Int8: "int8",
		Uint16: "uint16", Int16: "int16", Int32: "int32", Int64: "int64",
		Bool: "bool", Float16Type: "float16", Float64: "float64",
		Uint32: "uint32", Uint64: "uint64", BFloat16Type: "bfloat16"}
	if int(d) < len(names) && names[d] != "" {
		return names[d]
	}
	return fmt.Sprintf("DataType(%d)", d)
}

func (d DataType) size() int {
	sizes := [...]int{Float32: 4, Uint8: 1, Int8: 1, Uint16: 2, Int16: 2,
		Int32: 4, Int64: 8, Bool: 1, Float16Type: 2, Float64: 8,
		Uint32: 4, Uint64: 8, BFloat16Type: 2}
	if int(d) >= len(sizes) {
		return 0
	}
	return sizes[d]
}

func elementType[T Element]() DataType {
	switch any(*new(T)).(type) {
	case Float16:
		return Float16Type
	case BFloat16:
		return BFloat16Type
	}
	types := [...]DataType{reflect.Float32: Float32, reflect.Float64: Float64,
		reflect.Int8: Int8, reflect.Int16: Int16, reflect.Int32: Int32,
		reflect.Int64: Int64, reflect.Uint8: Uint8, reflect.Uint16: Uint16,
		reflect.Uint32: Uint32, reflect.Uint64: Uint64, reflect.Bool: Bool}
	return types[reflect.TypeFor[T]().Kind()]
}

// Tensor is a dense, contiguous, row-major tensor backed by Go memory.
// Copies share data; Shape returns a copy. Its zero value is invalid.
// A tensor and slices obtained from it remain valid after a session is closed.
type Tensor struct {
	data   unsafe.Pointer
	shape  []int64
	length int
	dtype  DataType
}

// NewTensor wraps data without copying it and copies shape. A nil or empty
// shape represents a scalar and requires one element. Zero dimensions are
// allowed; negative dimensions and overflowing sizes are rejected.
//
// Do not mutate data while an inference call reads or writes it. To give a
// tensor independent storage, pass slices.Clone(data).
func NewTensor[T Element](shape []int64, data []T) (Tensor, error) {
	dtype := elementType[T]()
	n, err := tensorSize(shape, dtype.size())
	if err != nil {
		return Tensor{}, err
	}
	if len(data) != n {
		return Tensor{}, fmt.Errorf("onnxcraft: shape %v needs %d elements, got %d", shape, n, len(data))
	}
	return Tensor{data: unsafe.Pointer(unsafe.SliceData(data)), shape: slices.Clone(shape), length: n, dtype: dtype}, nil
}

// Data returns a view of the tensor's data, with no copy. T must match its
// ONNX element type. Float16 and BFloat16 are distinct from uint16.
func (t Tensor) Data[T Element]() ([]T, error) {
	if want := elementType[T](); t.dtype != want {
		return nil, fmt.Errorf("onnxcraft: tensor contains %s, requested %s", t.dtype, want)
	}
	return unsafe.Slice((*T)(t.data), t.length), nil
}

// Shape returns a copy of the tensor's dimensions.
func (t Tensor) Shape() []int64 { return slices.Clone(t.shape) }

// Type returns the tensor's element type.
func (t Tensor) Type() DataType { return t.dtype }

func tensorSize(shape []int64, size int) (int, error) {
	empty := false
	for _, d := range shape {
		if d < 0 {
			return 0, fmt.Errorf("onnxcraft: negative tensor dimension %d", d)
		}
		empty = empty || d == 0
	}
	if size == 0 {
		return 0, fmt.Errorf("onnxcraft: unsupported tensor element type")
	}
	if empty {
		return 0, nil
	}
	n := 1
	for _, d := range shape {
		if d > int64(math.MaxInt/n) {
			return 0, fmt.Errorf("onnxcraft: tensor shape overflows int")
		}
		n *= int(d)
	}
	if n > math.MaxInt/size {
		return 0, fmt.Errorf("onnxcraft: tensor byte size overflows int")
	}
	return n, nil
}
