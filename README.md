# infergo

[![CI](https://github.com/joeychilson/infergo/actions/workflows/ci.yml/badge.svg)](https://github.com/joeychilson/infergo/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/joeychilson/infergo.svg)](https://pkg.go.dev/github.com/joeychilson/infergo)
[![Release](https://img.shields.io/github/v/release/joeychilson/infergo)](https://github.com/joeychilson/infergo/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Fast, typed ONNX inference for Go with managed native runtimes, verified model
artifacts, and production-oriented model pipelines.

Requires Go 1.27 or later and a C toolchain for cgo.

## Installation

```text
go get github.com/joeychilson/infergo
```

## Packages

- `infergo` — verified ONNX Runtime installation, lifecycle-safe sessions,
  model inspection, named or positional execution, metadata, typed tensors,
  graph optimization, context cancellation, and CPU/Core ML/CUDA/TensorRT/
  OpenVINO/DirectML execution.
- `bert` — configurable WordPiece tokenization, dynamic batches,
  masked-language prediction, and softmax/sigmoid sequence classification.
- `embedding` — batched sentence embeddings, attention-aware mean or CLS
  pooling, L2 normalization, and cosine similarity.
- `modelhub` — pinned model catalog plus atomic, SHA-256-verified downloads,
  caching, size limits, bundles, and offline mode.
- `resnet` — end-to-end, batched ImageNet image classification.
- `yolos` — end-to-end COCO object detection with YOLOS transformer models.
- `vision` — resize, crop, RGB normalization, and NCHW tensor conversion.
- `postprocess` — softmax classification and class-aware non-max suppression.
- `labels` — immutable access to the standard ImageNet-1K and COCO labels.

## Quick start

Create normalized MiniLM sentence embeddings and compare them. The first run
downloads both the pinned model and matching ONNX Runtime, verifies their
SHA-256 digests, and reuses them from the user cache afterward.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/joeychilson/infergo"
	"github.com/joeychilson/infergo/embedding"
	"github.com/joeychilson/infergo/modelhub"
)

func main() {
	ctx := context.Background()
	hub, err := modelhub.New()
	if err != nil {
		log.Fatal(err)
	}
	modelPath, err := hub.Fetch(ctx, modelhub.AllMiniLML6V2())
	if err != nil {
		log.Fatal(err)
	}

	runtime, err := infergo.Open(ctx)
	if err != nil {
		log.Fatal(err)
	}
	defer runtime.Close()

	model, err := embedding.New(runtime, modelPath)
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	vectors, err := model.Embed(ctx, []string{
		"A dog is playing in the park.",
		"A puppy plays outside.",
	}, embedding.EmbedOptions{MaxLength: 256})
	if err != nil {
		log.Fatal(err)
	}
	similarity, err := embedding.CosineSimilarity(vectors[0], vectors[1])
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("similarity: %.4f\n", similarity)
}
```

Runnable BERT, embedding, ResNet, and YOLOS programs are in
[`examples`](examples). They download the matching verified model when no
model path is supplied.

## Verified model catalog

The catalog pins every artifact to an immutable repository commit, exact byte
count, and SHA-256 digest. It is deliberately small; any HTTP or Hugging Face
artifact can use the same `modelhub.Artifact` and `modelhub.HuggingFace` APIs.

| Function | Task | Size | Compatible API |
| --- | --- | ---: | --- |
| `modelhub.BERTBaseUncased()` | fill mask | 110.8 MB | `bert.New` |
| `modelhub.AllMiniLML6V2()` | sentence embedding | 90.4 MB | `embedding.New` |
| `modelhub.ResNet50()` | ImageNet classification | 102.2 MB | `resnet.New` |
| `modelhub.YOLOSSmall()` | COCO object detection | 123.0 MB | `yolos.New` |

Pass `modelhub.WithOffline(true)` to guarantee that a run never accesses the
network. `FetchBundle` keeps an ONNX graph and its external tensor-data files
in one immutable directory; `FetchAll` handles independent batch downloads.
Model weights remain under their upstream licenses and are not copied into
this module.

## Language models

BERT model constructors inspect graph inputs and automatically supply a
zero-valued `token_type_ids` tensor when an export requires it. DistilBERT
two-input exports work without configuration. Custom graph names remain
available through `WithTensorNames` and `WithTokenTypeIDs`.

`bert.NewClassifier` supports batched single-label softmax, multi-label
sigmoid, and raw-score classification. Tokenizers can load any line-delimited
WordPiece vocabulary with `bert.NewTokenizerFromFile`; lowercasing, accent
stripping, and all special tokens are configurable for cased and multilingual
models.

## Native runtime

Infergo vNext uses ONNX Runtime 1.29.0 and verifies every automatic download
against the digest published with the official release. Automatic installation
supports:

- macOS on ARM64;
- Linux on AMD64 or ARM64; and
- Windows on AMD64 or ARM64.

For a system installation, a custom build, or another platform, set
`ONNXRUNTIME_SHARED_LIBRARY_PATH` or pass
`infergo.WithLibraryPath("/path/to/library")` to `infergo.Open`.

Automatic downloads are CPU builds. Core ML can be enabled on supported Apple
builds with `infergo.WithCoreML(nil)`. CUDA, TensorRT, OpenVINO, and DirectML
require an appropriate native runtime supplied through
`infergo.WithLibraryPath`, followed by the matching session option. The generic
`infergo.WithExecutionProvider` supports other providers such as QNN and
XNNPACK when compiled into that runtime. Provider options compose in priority
order and work through every high-level model's `WithSessionOptions` option.

The native ONNX Runtime environment is process-wide. Multiple runtimes and
sessions share it safely through reference counting, and active sessions keep
it alive even if their parent runtime is closed.

## Low-level sessions

Model-specific packages are the easiest entry point. Arbitrary ONNX tensor
models can be inspected, loaded without hard-coded graph names, and run by
name:

```go
info, err := runtime.Inspect("model.onnx")
if err != nil {
	log.Fatal(err)
}
fmt.Println(info.Inputs, info.Outputs)

session, err := runtime.Load("model.onnx")
if err != nil {
	log.Fatal(err)
}
defer session.Close()

input := infergo.MustTensor([]int64{1, 2}, []float32{1, 2})
outputs, err := session.RunNamed(ctx, map[string]infergo.Tensor{"input": input})
if err != nil {
	log.Fatal(err)
}
values, err := infergo.Data[float32](outputs["output"])
if err != nil {
	log.Fatal(err)
}
```

`Session.Metadata` exposes producer, graph, domain, description, version, and
custom metadata. `NewSession` remains available for positional execution and
for selecting only the outputs a large model should calculate. `Run` and
`RunNamed` may be called concurrently; returned tensors own independent Go
memory.

Model inspection reports tensors, sequences, maps, optional values, sparse
tensors, dynamic dimensions, and modern ONNX element types. Dense execution
currently supports booleans, strings, and the standard 8/16/32/64-bit integer
and 32/64-bit floating-point tensor types.

## Migration from the original API

This rebuild intentionally removes the old `pkg/*` and `models/*` layout.
Replace:

- `pkg/onnx.New` with `infergo.Open`;
- `models/bert` plus manual tokenization with `bert.FillMask`;
- `models/resnet` plus manual preprocessing with `resnet.Classify`; and
- `models/yolo` with `yolos.Detect`.

Prediction confidence is now consistently named `Score`. The new API validates
tensor shapes and options, honors context cancellation, applies
`MaxDetections`, avoids mutating caller data, and keeps native resources alive
for their complete lifetime. This rebuild is a breaking major-version change;
pin the original module version while migrating existing applications.

## Development

```text
golangci-lint run ./...
go test -shuffle=on ./...
go test -race -shuffle=on ./...
go build ./...
```

Set `INFERGO_INTEGRATION=1` to run the native ONNX Runtime smoke test. It
downloads the official runtime into a temporary cache, inspects and loads a
real ONNX graph, reads metadata, executes it by name, and verifies the result.

## Release

Push a semantic version tag in the form `vMAJOR.MINOR.PATCH`, optionally with a
prerelease suffix, to create a GitHub Release with generated notes. The release
publishes the Go module source and does not include binary artifacts.

## License

[MIT](LICENSE)
