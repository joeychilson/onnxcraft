// Package infergo runs ONNX models with a managed native runtime.
//
// Open initializes ONNX Runtime, verifies and caches the native library when
// needed, and returns a Runtime that can create context-aware sessions.
package infergo
