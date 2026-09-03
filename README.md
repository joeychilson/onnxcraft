# Infergo

[![CI](https://github.com/joeychilson/infergo/actions/workflows/ci.yml/badge.svg)](https://github.com/joeychilson/infergo/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/joeychilson/infergo.svg)](https://pkg.go.dev/github.com/joeychilson/infergo)
[![Release](https://img.shields.io/github/v/release/joeychilson/infergo)](https://github.com/joeychilson/infergo/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Infergo is a Go library for running ONNX models without manually managing the
native runtime. It provides typed tensors and sessions, verified model
downloads, and ready-to-use pipelines for common language and vision tasks.

Use the low-level API with any supported ONNX tensor model, or start with the
built-in BERT, sentence-embedding, ResNet, and YOLOS packages.

## Requirements

- Go 1.27 or later
- A C toolchain for cgo

## Install

```sh
go get github.com/joeychilson/infergo
```

## Quick start

This example downloads a pinned MiniLM model, creates sentence embeddings, and
compares them. Models and ONNX Runtime are cached after the first download.

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

More runnable examples are available in [`examples`](examples).

## Packages

| Package | Purpose |
| --- | --- |
| `infergo` | Runtime management, model inspection, typed tensors, and ONNX sessions |
| `modelhub` | Verified model downloads, caching, bundles, and offline mode |
| `bert` | WordPiece tokenization, fill-mask inference, and text classification |
| `embedding` | Batched sentence embeddings, pooling, normalization, and similarity |
| `resnet` | Batched ImageNet image classification |
| `yolos` | COCO object detection with YOLOS models |
| `vision` | Image resizing, normalization, and NCHW conversion |
| `postprocess` | Classification and object-detection postprocessing |
| `labels` | ImageNet-1K and COCO labels |

## Included models

The small built-in catalog pins each model to an immutable revision, expected
size, and SHA-256 digest. Custom HTTP and Hugging Face artifacts use the same
`modelhub` API.

| Model | Task | API |
| --- | --- | --- |
| `modelhub.BERTBaseUncased()` | Fill mask | `bert.New` |
| `modelhub.AllMiniLML6V2()` | Sentence embeddings | `embedding.New` |
| `modelhub.ResNet50()` | Image classification | `resnet.New` |
| `modelhub.YOLOSSmall()` | Object detection | `yolos.New` |

Use `modelhub.WithOffline(true)` when network access must be disabled. Model
weights remain subject to their upstream licenses and are not included in this
module.

## ONNX Runtime

`infergo.Open` automatically downloads and verifies the CPU build of ONNX
Runtime 1.29.0 on:

- macOS ARM64
- Linux AMD64 and ARM64
- Windows AMD64 and ARM64

Set `ONNXRUNTIME_SHARED_LIBRARY_PATH` or use `infergo.WithLibraryPath` to load a
system or custom runtime. Core ML is available on supported Apple builds;
CUDA, TensorRT, OpenVINO, DirectML, and other providers require a compatible
custom runtime and the corresponding session option.

The native environment is shared safely across runtimes and sessions. Sessions
remain usable until closed, even if their parent runtime has already closed.

## Low-level API

Use `Runtime.Inspect` to discover model inputs and outputs, `Runtime.Load` to
create a session from that schema, and `Session.Run` or `Session.RunNamed` for
inference. Sessions support concurrent calls, context cancellation, graph
optimization, metadata, and configurable execution providers.

Dense execution supports booleans, strings, standard integer types, `float32`,
and `float64` tensors.

## Development

```sh
golangci-lint run ./...
go test -race -shuffle=on ./...
go build ./...
```

Run the native ONNX Runtime integration test with:

```sh
INFERGO_INTEGRATION=1 go test ./...
```

## License

[MIT](LICENSE)
