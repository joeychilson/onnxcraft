package onnxcraft

/*
#cgo linux LDFLAGS: -ldl
#include "bridge.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"unsafe"
)

// ErrClosed is returned when using a closed runtime or session.
var ErrClosed = errors.New("onnxcraft: closed")

// NativeError preserves an ONNX Runtime error code and message.
// Use errors.As to inspect it. Codes are the OrtErrorCode values in the C API.
type NativeError struct {
	Code    int
	Message string
}

func (e *NativeError) Error() string {
	return fmt.Sprintf("onnxcraft: ONNX Runtime (%d): %s", e.Code, e.Message)
}

func nativeError(r *C.oc_runtime, status *C.OrtStatus) error {
	if status == nil {
		return nil
	}
	err := &NativeError{Code: int(C.oc_error_code(r, status)), Message: C.GoString(C.oc_error_message(r, status))}
	C.oc_error_free(r, status)
	return err
}

// Runtime owns a loaded ONNX Runtime library. It must not be copied.
// Close prevents new sessions; existing sessions keep the library alive.
type Runtime struct {
	mu      sync.Mutex
	native  *C.oc_runtime
	refs    int
	closed  bool
	version string
}

// Open loads an explicit shared library path. Relative paths are resolved
// against the working directory. The library must support C API version 29.
func Open(libraryPath string) (*Runtime, error) {
	if libraryPath == "" || strings.ContainsRune(libraryPath, 0) {
		return nil, errors.New("onnxcraft: library path must be nonempty and contain no NUL")
	}
	path, err := filepath.Abs(libraryPath)
	if err != nil {
		return nil, err
	}
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	var ptr *C.oc_runtime
	var message *C.char
	code := C.oc_open(cpath, &ptr, &message)
	if code != 0 {
		defer C.free(unsafe.Pointer(message))
		if code > 0 {
			return nil, &NativeError{Code: int(code), Message: C.GoString(message)}
		}
		return nil, fmt.Errorf("onnxcraft: open %q: %s", path, C.GoString(message))
	}
	return &Runtime{native: ptr, refs: 1, version: C.GoString(ptr.version)}, nil
}

// Version returns the loaded library's version, even after Close.
func (r *Runtime) Version() string { return r.version }

// Close releases the runtime once its sessions have closed. It is idempotent.
func (r *Runtime) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.closed {
		r.closed = true
		if r.native != nil {
			r.release()
		}
	}
	return nil
}

// release requires r.mu. Each runtime and session owns one reference.
func (r *Runtime) release() {
	r.refs--
	if r.refs == 0 {
		C.oc_close(r.native)
		r.native = nil
	}
}

// Optimization selects graph transformations during model loading.
type Optimization uint8

const (
	OptimizeAll Optimization = iota // Default: all ONNX Runtime optimizations.
	OptimizeNone
	OptimizeBasic
	OptimizeExtended
)

// Provider configures an execution provider by its ONNX Runtime name, such as
// "CUDA", "TensorRT", "CoreML", "OpenVINO", or "XNNPACK". The library must
// include that provider; unavailable providers return an error during Load.
type Provider struct {
	Name    string
	Options map[string]string
}

// SessionOptions configures model loading. Its zero value uses CPU execution,
// sequential graph scheduling, all graph optimizations, and ORT thread defaults.
type SessionOptions struct {
	IntraOpThreads int // Threads within an operator; zero lets ORT choose.
	InterOpThreads int // Threads between operators when Parallel is true.
	Parallel       bool
	Optimization   Optimization
	Providers      []Provider        // In priority order; ORT's CPU fallback remains enabled.
	Config         map[string]string // ONNX Runtime session configuration entries.
}

// Load opens an ONNX or ORT model file, including models with external weights.
// A nil options pointer selects the defaults. Options are consumed during Load.
func (r *Runtime) Load(path string, options *SessionOptions) (*Session, error) {
	if path == "" || strings.ContainsRune(path, 0) {
		return nil, errors.New("onnxcraft: model path must be nonempty and contain no NUL")
	}
	return r.load(path, nil, options)
}

// LoadBytes loads a self-contained ONNX or ORT model. The bytes may be released
// or changed after this call returns. Use Load for models with external weights.
func (r *Runtime) LoadBytes(model []byte, options *SessionOptions) (*Session, error) {
	if len(model) == 0 {
		return nil, errors.New("onnxcraft: model is empty")
	}
	return r.load("", model, options)
}

func (r *Runtime) load(path string, model []byte, options *SessionOptions) (*Session, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed || r.native == nil {
		return nil, ErrClosed
	}
	o := SessionOptions{}
	if options != nil {
		o = *options
	}
	if o.IntraOpThreads < 0 || o.IntraOpThreads > math.MaxInt32 || o.InterOpThreads < 0 || o.InterOpThreads > math.MaxInt32 || o.Optimization > OptimizeExtended {
		return nil, errors.New("onnxcraft: invalid thread count or graph optimization")
	}
	var raw *C.OrtSessionOptions
	config := C.oc_options{intra_threads: C.int(o.IntraOpThreads), inter_threads: C.int(o.InterOpThreads), optimization: C.int(o.Optimization)}
	if o.Parallel {
		config.parallel = 1
	}
	if err := nativeError(r.native, C.oc_options_new(r.native, &config, &raw)); err != nil {
		return nil, err
	}
	defer C.oc_options_free(r.native, raw)
	for key, value := range o.Config {
		// These options make ORT retain the model buffer beyond CreateSession.
		if key == "session.use_ort_model_bytes" || key == "session.use_ort_model_bytes_for_initializers" {
			return nil, fmt.Errorf("onnxcraft: session config %q is incompatible with Go-owned model bytes", key)
		}
		if key == "" || strings.ContainsRune(key, 0) || strings.ContainsRune(value, 0) {
			return nil, errors.New("onnxcraft: invalid session config key or value")
		}
		k, v := C.CString(key), C.CString(value)
		err := nativeError(r.native, C.oc_config(r.native, raw, k, v))
		C.free(unsafe.Pointer(k))
		C.free(unsafe.Pointer(v))
		if err != nil {
			return nil, fmt.Errorf("onnxcraft: session config %q: %w", key, err)
		}
	}
	for _, provider := range o.Providers {
		if err := r.addProvider(raw, provider); err != nil {
			return nil, err
		}
	}
	var cpath *C.char
	if path != "" {
		cpath = C.CString(path)
		defer C.free(unsafe.Pointer(cpath))
	}
	var ptr *C.oc_session
	err := nativeError(r.native, C.oc_load(r.native, cpath, unsafe.Pointer(unsafe.SliceData(model)), C.size_t(len(model)), raw, &ptr))
	runtime.KeepAlive(model)
	if err != nil {
		return nil, fmt.Errorf("onnxcraft: load model: %w", err)
	}
	ports := unsafe.Slice(ptr.ports, int(ptr.inputs+ptr.outputs))
	info := make([]TensorInfo, len(ports))
	for i, p := range ports {
		dtype := DataType(p._type)
		if dtype.size() == 0 {
			name := C.GoString(p.name)
			C.oc_session_free(ptr)
			return nil, fmt.Errorf("onnxcraft: %q has unsupported tensor type %d", name, p._type)
		}
		info[i] = TensorInfo{Name: C.GoString(p.name), Type: dtype, Shape: append([]int64(nil), unsafe.Slice((*int64)(unsafe.Pointer(p.shape)), int(p.rank))...)}
	}
	r.refs++
	s := &Session{native: ptr, runtime: r, inputs: info[:int(ptr.inputs)], outputs: info[int(ptr.inputs):]}
	if len(o.Providers) != 0 {
		s.gate = make(chan struct{}, 1)
	}
	return s, nil
}

func (r *Runtime) addProvider(o *C.OrtSessionOptions, provider Provider) error {
	if provider.Name == "" || strings.ContainsRune(provider.Name, 0) {
		return errors.New("onnxcraft: invalid execution provider name")
	}
	name := C.CString(provider.Name)
	defer C.free(unsafe.Pointer(name))
	keys := make([]*C.char, 0, len(provider.Options))
	values := make([]*C.char, 0, len(provider.Options))
	defer func() {
		for i := range keys {
			C.free(unsafe.Pointer(keys[i]))
			C.free(unsafe.Pointer(values[i]))
		}
	}()
	for key, value := range provider.Options {
		if (key == "enable_cuda_graph" || key == "trt_cuda_graph_enable") && value != "0" {
			return errors.New("onnxcraft: CUDA graph capture requires persistent device buffers and is not supported")
		}
		if key == "" || strings.ContainsRune(key, 0) || strings.ContainsRune(value, 0) {
			return errors.New("onnxcraft: invalid execution provider option")
		}
		keys = append(keys, C.CString(key))
		values = append(values, C.CString(value))
	}
	if err := nativeError(r.native, C.oc_provider(r.native, o, name, unsafe.SliceData(keys), unsafe.SliceData(values), C.size_t(len(keys)))); err != nil {
		return fmt.Errorf("onnxcraft: provider %q: %w", provider.Name, err)
	}
	return nil
}
