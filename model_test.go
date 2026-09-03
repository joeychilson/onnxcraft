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

func TestValidateModelInfo(t *testing.T) {
	t.Parallel()
	valid := ModelInfo{
		Inputs:  []ValueInfo{{Name: "input", Kind: ValueKindTensor, Shape: []int64{-1, 3}, Type: DataTypeFloat32}},
		Outputs: []ValueInfo{{Name: "output", Kind: ValueKindTensor, Shape: []int64{-1}, Type: DataTypeInt64}},
	}
	if err := validateModelInfo(valid); err != nil {
		t.Fatal(err)
	}
	invalid := valid
	invalid.Outputs = []ValueInfo{{Name: "output", Kind: ValueKindSequence, Type: DataTypeFloat32}}
	if err := validateModelInfo(invalid); err == nil {
		t.Fatal("validateModelInfo(sequence) error = nil")
	}
	invalid.Outputs = []ValueInfo{{Name: "output", Kind: ValueKindTensor, Type: DataTypeFloat16}}
	if err := validateModelInfo(invalid); err == nil {
		t.Fatal("validateModelInfo(float16) error = nil")
	}
}

func TestValidateTensorAgainstSchema(t *testing.T) {
	t.Parallel()
	info := ValueInfo{Name: "input", Kind: ValueKindTensor, Shape: []int64{-1, 3}, Type: DataTypeFloat32}
	if err := validateTensor(MustTensor([]int64{2, 3}, make([]float32, 6)), info); err != nil {
		t.Fatal(err)
	}
	if err := validateTensor(MustTensor([]int64{2, 4}, make([]float32, 8)), info); err == nil {
		t.Fatal("validateTensor(shape) error = nil")
	}
	if err := validateTensor(MustTensor([]int64{2, 3}, make([]int64, 6)), info); err == nil {
		t.Fatal("validateTensor(type) error = nil")
	}
}

func TestInspectBytesRejectsEmptyModel(t *testing.T) {
	t.Parallel()
	var runtime *Runtime
	if _, err := runtime.InspectBytes(nil); err == nil {
		t.Fatal("InspectBytes() error = nil")
	}
}
