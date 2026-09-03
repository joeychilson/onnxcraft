package infergo

import (
	"errors"
	"fmt"
	"math"
	"slices"
)

// TensorData is a Go type supported by an ONNX tensor.
type TensorData interface {
	bool | string |
		float32 | float64 |
		int8 | int16 | int32 | int64 |
		uint8 | uint16 | uint32 | uint64
}

// DataType identifies the element type stored in a Tensor.
type DataType string

// Supported tensor element types.
const (
	DataTypeUndefined      DataType = "undefined"
	DataTypeBool           DataType = "bool"
	DataTypeString         DataType = "string"
	DataTypeFloat32        DataType = "float32"
	DataTypeFloat64        DataType = "float64"
	DataTypeFloat16        DataType = "float16"
	DataTypeBFloat16       DataType = "bfloat16"
	DataTypeFloat8E4M3FN   DataType = "float8e4m3fn"
	DataTypeFloat8E4M3FNUZ DataType = "float8e4m3fnuz"
	DataTypeFloat8E5M2     DataType = "float8e5m2"
	DataTypeFloat8E5M2FNUZ DataType = "float8e5m2fnuz"
	DataTypeFloat8E8M0     DataType = "float8e8m0"
	DataTypeFloat4E2M1     DataType = "float4e2m1"
	DataTypeComplex64      DataType = "complex64"
	DataTypeComplex128     DataType = "complex128"
	DataTypeInt8           DataType = "int8"
	DataTypeInt16          DataType = "int16"
	DataTypeInt32          DataType = "int32"
	DataTypeInt64          DataType = "int64"
	DataTypeInt4           DataType = "int4"
	DataTypeInt2           DataType = "int2"
	DataTypeUint8          DataType = "uint8"
	DataTypeUint16         DataType = "uint16"
	DataTypeUint32         DataType = "uint32"
	DataTypeUint64         DataType = "uint64"
	DataTypeUint4          DataType = "uint4"
	DataTypeUint2          DataType = "uint2"
)

// Tensor is an immutable, row-major ONNX tensor.
type Tensor struct {
	shape    []int64
	data     any
	dataType DataType
}

// NewTensor constructs a tensor and copies shape and data so callers can
// safely reuse their input slices.
func NewTensor[T TensorData](shape []int64, data []T) (Tensor, error) {
	size, err := shapeSize(shape)
	if err != nil {
		return Tensor{}, err
	}
	if size != int64(len(data)) {
		return Tensor{}, fmt.Errorf("infergo: tensor shape contains %d elements, data contains %d", size, len(data))
	}

	return Tensor{
		shape:    slices.Clone(shape),
		data:     slices.Clone(data),
		dataType: dataTypeOf[T](),
	}, nil
}

// MustTensor is like NewTensor but panics if shape and data are incompatible.
// It is intended for package-level constants and tests.
func MustTensor[T TensorData](shape []int64, data []T) Tensor {
	tensor, err := NewTensor(shape, data)
	if err != nil {
		panic(err)
	}
	return tensor
}

// Data returns a copy of tensor's data as T.
func Data[T TensorData](tensor Tensor) ([]T, error) {
	data, ok := tensor.data.([]T)
	if !ok {
		return nil, fmt.Errorf("infergo: tensor contains %s, not %s", tensor.dataType, dataTypeOf[T]())
	}
	return slices.Clone(data), nil
}

// Shape returns a copy of the tensor dimensions.
func (t Tensor) Shape() []int64 {
	return slices.Clone(t.shape)
}

// Type returns the tensor element type.
func (t Tensor) Type() DataType {
	return t.dataType
}

// Len returns the flattened number of tensor elements.
func (t Tensor) Len() int {
	size, err := shapeSize(t.shape)
	if err != nil || size > int64(math.MaxInt) {
		return 0
	}
	return int(size)
}

func shapeSize(shape []int64) (int64, error) {
	if len(shape) == 0 {
		return 0, errors.New("infergo: tensor shape must contain at least one dimension")
	}
	for index, dimension := range shape {
		if dimension < 0 {
			return 0, fmt.Errorf("infergo: tensor dimension %d is negative", index)
		}
		if dimension == 0 {
			return 0, nil
		}
	}

	size := int64(1)
	for _, dimension := range shape {
		if size > math.MaxInt64/dimension {
			return 0, errors.New("infergo: tensor shape overflows int64")
		}
		size *= dimension
	}
	return size, nil
}

func dataTypeOf[T TensorData]() DataType {
	var value T
	switch any(value).(type) {
	case bool:
		return DataTypeBool
	case string:
		return DataTypeString
	case float32:
		return DataTypeFloat32
	case float64:
		return DataTypeFloat64
	case int8:
		return DataTypeInt8
	case int16:
		return DataTypeInt16
	case int32:
		return DataTypeInt32
	case int64:
		return DataTypeInt64
	case uint8:
		return DataTypeUint8
	case uint16:
		return DataTypeUint16
	case uint32:
		return DataTypeUint32
	case uint64:
		return DataTypeUint64
	default:
		panic("infergo: unsupported tensor data type")
	}
}
