// Package onnxcraft runs ONNX models using the ONNX Runtime C API.
//
// Open a Runtime, load a Session, and pass tensors to Session.Run. Reuse the
// session across requests. Use Session.RunInto to reuse output storage.
// Tensor data belongs to Go and needs no Close; runtimes and sessions must be
// closed explicitly. A session supports concurrent runs with independent output
// buffers. Applications must synchronize access to tensor data they mutate.
//
// Building requires Go 1.27 and cgo. Running requires an ONNX Runtime 1.29 or
// newer shared library. Loading the library never downloads or installs code.
package onnxcraft
