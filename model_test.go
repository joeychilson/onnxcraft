package infergo

import (
	"slices"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func TestConvertValueInfo(t *testing.T) {
	t.Parallel()
	result := convertValueInfo([]ort.InputOutputInfo{{
		Name:         "input",
		OrtValueType: ort.ONNXTypeTensor,
		Dimensions:   ort.Shape{-1, 3},
		DataType:     ort.TensorElementDataTypeFloat,
	}})
	if len(result) != 1 || result[0].Name != "input" || result[0].Kind != ValueKindTensor ||
		result[0].Type != DataTypeFloat32 || !slices.Equal(result[0].Shape, []int64{-1, 3}) {
		t.Fatalf("result = %+v", result)
	}
}

func TestElementTypeSupportsModernONNXTypes(t *testing.T) {
	t.Parallel()
	tests := map[ort.TensorElementDataType]DataType{
		ort.TensorElementDataTypeFloat16:      DataTypeFloat16,
		ort.TensorElementDataTypeBFloat16:     DataTypeBFloat16,
		ort.TensorElementDataTypeFloat8E4M3FN: DataTypeFloat8E4M3FN,
		ort.TensorElementDataTypeInt4:         DataTypeInt4,
		ort.TensorElementDataTypeUint2:        DataTypeUint2,
	}
	for input, want := range tests {
		if got := elementType(input); got != want {
			t.Errorf("elementType(%v) = %q, want %q", input, got, want)
		}
	}
}
