package infergo

import (
	"errors"
	"fmt"
	"slices"

	ort "github.com/yalue/onnxruntime_go"
)

// ValueKind identifies the top-level ONNX type of a model input or output.
type ValueKind string

// ONNX value kinds reported by Inspect.
const (
	ValueKindUnknown      ValueKind = "unknown"
	ValueKindTensor       ValueKind = "tensor"
	ValueKindSequence     ValueKind = "sequence"
	ValueKindMap          ValueKind = "map"
	ValueKindOpaque       ValueKind = "opaque"
	ValueKindSparseTensor ValueKind = "sparse_tensor"
	ValueKindOptional     ValueKind = "optional"
)

// ValueInfo describes one model input or output. Dynamic dimensions are
// represented by negative values, as reported by ONNX Runtime.
type ValueInfo struct {
	Name  string
	Kind  ValueKind
	Shape []int64
	Type  DataType
}

// ModelInfo describes a model's ordered inputs and outputs.
type ModelInfo struct {
	Inputs  []ValueInfo
	Outputs []ValueInfo
}

// ModelMetadata contains descriptive fields embedded in an ONNX model.
type ModelMetadata struct {
	Producer    string
	Graph       string
	Domain      string
	Description string
	Version     int64
	Custom      map[string]string
}

// Inspect reads the ordered inputs and outputs from modelPath. It creates a
// temporary ONNX session, so loading a large model may be expensive.
func (r *Runtime) Inspect(modelPath string, options ...SessionOption) (result ModelInfo, resultErr error) {
	if err := validateModel(modelPath); err != nil {
		return ModelInfo{}, err
	}
	config, err := resolveSessionConfig(options)
	if err != nil {
		return ModelInfo{}, err
	}
	release, err := r.retain()
	if err != nil {
		return ModelInfo{}, err
	}
	defer func() { resultErr = errors.Join(resultErr, release()) }()

	sessionOptions, err := newORTSessionOptions(config)
	if err != nil {
		return ModelInfo{}, err
	}
	inputs, outputs, inspectErr := ort.GetInputOutputInfoWithOptions(modelPath, sessionOptions)
	destroyErr := sessionOptions.Destroy()
	if inspectErr != nil {
		return ModelInfo{}, errors.Join(fmt.Errorf("infergo: inspect model graph: %w", inspectErr), destroyErr)
	}
	if destroyErr != nil {
		return ModelInfo{}, fmt.Errorf("infergo: close inspection options: %w", destroyErr)
	}
	return ModelInfo{Inputs: convertValueInfo(inputs), Outputs: convertValueInfo(outputs)}, nil
}

// InspectBytes reads ordered inputs and outputs from an in-memory ONNX model.
func (r *Runtime) InspectBytes(model []byte) (result ModelInfo, resultErr error) {
	if len(model) == 0 {
		return ModelInfo{}, errors.New("infergo: model data cannot be empty")
	}
	release, err := r.retain()
	if err != nil {
		return ModelInfo{}, err
	}
	defer func() { resultErr = errors.Join(resultErr, release()) }()
	inputs, outputs, err := ort.GetInputOutputInfoWithONNXData(model)
	if err != nil {
		return ModelInfo{}, fmt.Errorf("infergo: inspect in-memory model graph: %w", err)
	}
	return ModelInfo{Inputs: convertValueInfo(inputs), Outputs: convertValueInfo(outputs)}, nil
}

// Load inspects modelPath and creates a session using every graph input and
// output in model order. Use NewSession when only selected outputs are needed.
func (r *Runtime) Load(modelPath string, options ...SessionOption) (*Session, error) {
	info, err := r.Inspect(modelPath, options...)
	if err != nil {
		return nil, err
	}
	return r.NewSessionFromInfo(modelPath, info, options...)
}

// LoadBytes inspects an in-memory ONNX model and creates a schema-aware
// session using every graph input and output.
func (r *Runtime) LoadBytes(model []byte, options ...SessionOption) (*Session, error) {
	info, err := r.InspectBytes(model)
	if err != nil {
		return nil, err
	}
	session, err := r.NewSessionFromBytes(model, valueNames(info.Inputs), valueNames(info.Outputs), options...)
	if err != nil {
		return nil, err
	}
	session.mu.Lock()
	session.inputInfo = cloneValueInfo(info.Inputs)
	session.outputInfo = cloneValueInfo(info.Outputs)
	session.mu.Unlock()
	return session, nil
}

func validateModelInfo(info ModelInfo) error {
	if err := validateValueInfo("input", info.Inputs); err != nil {
		return err
	}
	return validateValueInfo("output", info.Outputs)
}

func validateValueInfo(kind string, values []ValueInfo) error {
	if err := validateNames(kind, valueNames(values)); err != nil {
		return err
	}
	for _, value := range values {
		if value.Kind != ValueKindTensor {
			return fmt.Errorf("infergo: %s %q has unsupported ONNX kind %s", kind, value.Name, value.Kind)
		}
		if !supportedTensorType(value.Type) {
			return fmt.Errorf("infergo: %s %q has unsupported tensor type %s", kind, value.Name, value.Type)
		}
	}
	return nil
}

func supportedTensorType(dataType DataType) bool {
	switch dataType {
	case DataTypeBool, DataTypeString, DataTypeFloat32, DataTypeFloat64,
		DataTypeInt8, DataTypeInt16, DataTypeInt32, DataTypeInt64,
		DataTypeUint8, DataTypeUint16, DataTypeUint32, DataTypeUint64:
		return true
	case DataTypeUndefined, DataTypeFloat16, DataTypeBFloat16,
		DataTypeComplex64, DataTypeComplex128, DataTypeFloat8E4M3FN,
		DataTypeFloat8E4M3FNUZ, DataTypeFloat8E5M2, DataTypeFloat8E5M2FNUZ,
		DataTypeInt4, DataTypeUint4, DataTypeInt2, DataTypeUint2, DataTypeFloat4E2M1,
		DataTypeFloat8E8M0:
		return false
	default:
		return false
	}
}

func cloneValueInfo(source []ValueInfo) []ValueInfo {
	result := make([]ValueInfo, len(source))
	for index, info := range source {
		result[index] = info
		result[index].Shape = slices.Clone(info.Shape)
	}
	return result
}

func convertValueInfo(source []ort.InputOutputInfo) []ValueInfo {
	result := make([]ValueInfo, len(source))
	for index, info := range source {
		result[index] = ValueInfo{
			Name:  info.Name,
			Kind:  valueKind(info.OrtValueType),
			Shape: slices.Clone(info.Dimensions),
			Type:  elementType(info.DataType),
		}
	}
	return result
}

func valueNames(info []ValueInfo) []string {
	result := make([]string, len(info))
	for index := range info {
		result[index] = info[index].Name
	}
	return result
}

func valueKind(kind ort.ONNXType) ValueKind {
	switch kind {
	case ort.ONNXTypeUnknown:
		return ValueKindUnknown
	case ort.ONNXTypeTensor:
		return ValueKindTensor
	case ort.ONNXTypeSequence:
		return ValueKindSequence
	case ort.ONNXTypeMap:
		return ValueKindMap
	case ort.ONNXTypeOpaque:
		return ValueKindOpaque
	case ort.ONNXTypeSparseTensor:
		return ValueKindSparseTensor
	case ort.ONNXTypeOptional:
		return ValueKindOptional
	default:
		return ValueKindUnknown
	}
}

func elementType(dataType ort.TensorElementDataType) DataType {
	switch dataType {
	case ort.TensorElementDataTypeUndefined:
		return DataTypeUndefined
	case ort.TensorElementDataTypeFloat:
		return DataTypeFloat32
	case ort.TensorElementDataTypeUint8:
		return DataTypeUint8
	case ort.TensorElementDataTypeInt8:
		return DataTypeInt8
	case ort.TensorElementDataTypeUint16:
		return DataTypeUint16
	case ort.TensorElementDataTypeInt16:
		return DataTypeInt16
	case ort.TensorElementDataTypeInt32:
		return DataTypeInt32
	case ort.TensorElementDataTypeInt64:
		return DataTypeInt64
	case ort.TensorElementDataTypeString:
		return DataTypeString
	case ort.TensorElementDataTypeBool:
		return DataTypeBool
	case ort.TensorElementDataTypeFloat16:
		return DataTypeFloat16
	case ort.TensorElementDataTypeDouble:
		return DataTypeFloat64
	case ort.TensorElementDataTypeUint32:
		return DataTypeUint32
	case ort.TensorElementDataTypeUint64:
		return DataTypeUint64
	case ort.TensorElementDataTypeComplex64:
		return DataTypeComplex64
	case ort.TensorElementDataTypeComplex128:
		return DataTypeComplex128
	case ort.TensorElementDataTypeBFloat16:
		return DataTypeBFloat16
	case ort.TensorElementDataTypeFloat8E4M3FN:
		return DataTypeFloat8E4M3FN
	case ort.TensorElementDataTypeFloat8E4M3FNUZ:
		return DataTypeFloat8E4M3FNUZ
	case ort.TensorElementDataTypeFloat8E5M2:
		return DataTypeFloat8E5M2
	case ort.TensorElementDataTypeFloat8E5M2FNUZ:
		return DataTypeFloat8E5M2FNUZ
	case ort.TensorElementDataTypeUint4:
		return DataTypeUint4
	case ort.TensorElementDataTypeInt4:
		return DataTypeInt4
	case ort.TensorElementDataTypeFloat4E2M1:
		return DataTypeFloat4E2M1
	case ort.TensorElementDataTypeUint2:
		return DataTypeUint2
	case ort.TensorElementDataTypeInt2:
		return DataTypeInt2
	case ort.TensorElementDataTypeFloat8E8M0:
		return DataTypeFloat8E8M0
	default:
		return DataTypeUndefined
	}
}
