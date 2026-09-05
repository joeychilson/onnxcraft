package onnxcraft

/*
// This binding exposes no callbacks into Go.
#cgo nocallback oc_run
#cgo nocallback oc_run_into
#include "bridge.h"
*/
import "C"

import (
	"context"
	"fmt"
	"runtime"
	"slices"
	"sync"
	"unsafe"
)

// TensorInfo describes a model input or output. Negative dimensions are dynamic.
type TensorInfo struct {
	Name  string
	Type  DataType
	Shape []int64
}

// Session is a loaded model. Reuse it across inference calls; it must not be
// copied. CPU runs may execute concurrently. Sessions with explicit execution
// providers serialize runs because some providers require exclusive access.
type Session struct {
	mu              sync.RWMutex
	gate            chan struct{}
	native          *C.oc_session
	runtime         *Runtime
	inputs, outputs []TensorInfo
}

// Only descriptors are pooled. Their Go and native pointers are cleared before
// reuse; the pool never retains tensor storage or native resources.
var runDescriptors = sync.Pool{New: func() any { return new([]C.oc_tensor) }}

// Inputs returns independent copies of the input metadata in Run argument order.
func (s *Session) Inputs() []TensorInfo { return cloneInfo(s.inputs) }

// Outputs returns independent copies of the output metadata in result order.
func (s *Session) Outputs() []TensorInfo { return cloneInfo(s.outputs) }

func cloneInfo(info []TensorInfo) []TensorInfo {
	result := slices.Clone(info)
	for i := range result {
		result[i].Shape = slices.Clone(result[i].Shape)
	}
	return result
}

// Close waits for active inference calls, then releases the model. It is
// idempotent. Later calls to Run and RunInto return ErrClosed.
func (s *Session) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.native != nil {
		C.oc_session_free(s.native)
		s.native = nil
		s.runtime.mu.Lock()
		s.runtime.release()
		s.runtime.mu.Unlock()
	}
	return nil
}

// Run executes all model outputs. Inputs follow Inputs order. Returned tensors
// own independent Go storage; they need no cleanup and can be reused as inputs.
// Cancellation asks ONNX Runtime to terminate and waits until native work stops.
// Cancellation latency depends on the executing operator and provider.
func (s *Session) Run(ctx context.Context, inputs ...Tensor) ([]Tensor, error) {
	return s.run(ctx, inputs, nil, false)
}

// RunInto writes all outputs into caller-provided tensors, without tensor data
// copies. Outputs must have the exact resulting types and shapes, in Outputs
// order. They must not overlap inputs or each other. Do not read or mutate
// their data during the call or use them in concurrent runs. On error or
// cancellation their contents are unspecified; the buffers remain reusable.
func (s *Session) RunInto(ctx context.Context, outputs []Tensor, inputs ...Tensor) error {
	_, err := s.run(ctx, inputs, outputs, true)
	return err
}

func (s *Session) run(ctx context.Context, inputs, outputs []Tensor, into bool) ([]Tensor, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.native == nil {
		return nil, ErrClosed
	}
	if s.gate != nil {
		select {
		case s.gate <- struct{}{}:
			defer func() { <-s.gate }()
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		if err := ctx.Err(); err != nil {
			return nil, err
		}
	}
	if len(inputs) != len(s.inputs) {
		return nil, fmt.Errorf("onnxcraft: expected %d inputs, got %d", len(s.inputs), len(inputs))
	}
	if into && len(outputs) != len(s.outputs) {
		return nil, fmt.Errorf("onnxcraft: expected %d outputs, got %d", len(s.outputs), len(outputs))
	}
	for i, out := range outputs {
		start := uintptr(out.data)
		end := start + uintptr(out.length*out.dtype.size())
		for _, group := range [2][]Tensor{inputs, outputs[:i]} {
			for _, other := range group {
				p := uintptr(other.data)
				q := p + uintptr(other.length*other.dtype.size())
				if start < end && p < q && start < q && p < end {
					return nil, fmt.Errorf("onnxcraft: output %d overlaps another tensor", i)
				}
			}
		}
	}
	n := len(s.inputs) + len(s.outputs)
	buffer := runDescriptors.Get().(*[]C.oc_tensor)
	if cap(*buffer) < n {
		*buffer = make([]C.oc_tensor, n)
	}
	descriptors := (*buffer)[:n]
	defer func() {
		clear(descriptors)
		runDescriptors.Put(buffer)
	}()
	var pins runtime.Pinner
	defer pins.Unpin()
	index := 0
	for _, group := range [2][]Tensor{inputs, outputs} {
		for _, tensor := range group {
			if tensor.dtype.size() == 0 {
				return nil, fmt.Errorf("onnxcraft: invalid tensor at index %d", index)
			}
			shape := unsafe.SliceData(tensor.shape)
			if shape != nil {
				pins.Pin(shape)
			}
			if tensor.data != nil {
				pins.Pin(tensor.data)
			}
			descriptors[index] = C.oc_tensor{data: tensor.data, shape: (*C.int64_t)(unsafe.Pointer(shape)),
				bytes: C.size_t(tensor.length * tensor.dtype.size()), rank: C.size_t(len(tensor.shape)), _type: C.ONNXTensorElementDataType(tensor.dtype)}
			index++
		}
	}
	r := s.runtime.native
	var options *C.OrtRunOptions
	if ctx.Done() != nil {
		var finish func()
		var err error
		options, finish, err = watchContext(ctx, r)
		if err != nil {
			return nil, err
		}
		defer finish()
	}
	in := unsafe.SliceData(descriptors)
	out := &descriptors[len(s.inputs)]
	var status *C.OrtStatus
	if into {
		status = C.oc_run_into(s.native, options, in, out)
	} else {
		status = C.oc_run(s.native, options, in, out)
		defer C.oc_outputs_free(s.native, out)
	}
	err := nativeError(r, status)
	if ctx.Err() != nil {
		return nil, ctx.Err()
	}
	if err != nil {
		return nil, fmt.Errorf("onnxcraft: run model: %w", err)
	}
	if into {
		return nil, nil
	}
	result := make([]Tensor, len(s.outputs))
	for i, output := range descriptors[len(s.inputs):] {
		shape := slices.Clone(unsafe.Slice((*int64)(unsafe.Pointer(output.shape)), int(output.rank)))
		dtype := DataType(output._type)
		length, err := tensorSize(shape, dtype.size())
		if err != nil {
			return nil, err
		}
		nbytes := length * dtype.size()
		if C.size_t(nbytes) != output.bytes || (nbytes != 0 && output.data == nil) {
			return nil, fmt.Errorf("onnxcraft: output %d has inconsistent storage", i)
		}
		// uint64 storage guarantees alignment for every supported element type.
		storage := make([]uint64, nbytes/8+min(nbytes%8, 1))
		ptr := unsafe.Pointer(unsafe.SliceData(storage))
		copy(unsafe.Slice((*byte)(ptr), nbytes), unsafe.Slice((*byte)(output.data), nbytes))
		result[i] = Tensor{data: ptr, shape: shape, length: length, dtype: dtype}
	}
	return result, nil
}

// Keep cancellation setup separate so passing &options to cgo only allocates
// for cancelable contexts. Background runs require no native run options.
func watchContext(ctx context.Context, r *C.oc_runtime) (*C.OrtRunOptions, func(), error) {
	var options *C.OrtRunOptions
	if err := nativeError(r, C.oc_run_options(r, &options)); err != nil {
		return nil, nil, err
	}
	done := make(chan struct{})
	stop := context.AfterFunc(ctx, func() {
		C.oc_terminate(r, options)
		close(done)
	})
	return options, func() {
		// Join an already-started callback before releasing its native handle.
		if !stop() {
			<-done
		}
		C.oc_run_options_free(r, options)
	}, nil
}
