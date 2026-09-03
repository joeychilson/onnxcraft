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

- Go 1.27.0 or later
- A C toolchain for cgo

Infergo is tested with the race detector on Linux AMD64/ARM64, macOS ARM64,
and Windows AMD64. Windows ARM64 is covered without the race detector. The
native boundary also runs under Go's strict cgo pointer checks and AddressSanitizer.

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
module. Catalog artifacts expose their source repository, immutable revision,
SPDX license identifier, byte size, and SHA-256 digest.

`modelhub.Client` deduplicates concurrent fetches across goroutines and
processes, installs bundles transactionally, retries transient failures, and
resumes interrupted downloads with verified HTTP ranges. Production clients
can configure download concurrency, retry count, maximum size, authentication
headers, progress reporting, cache location, and offline operation:

```go
hub, err := modelhub.New(
	modelhub.WithConcurrency(4),
	modelhub.WithRetries(3),
	modelhub.WithRequestHeader("Authorization", "Bearer "+token),
	modelhub.WithProgress(func(update modelhub.Progress) {
		fmt.Printf("%s: %d/%d\n", update.Artifact.Name, update.Downloaded, update.Total)
	}),
)
```

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
`Runtime.Info` reports the actual loaded version and library path, while
`Runtime.ExecutionProviderDevices` reports hardware advertised by registered
execution-provider plugins. Use `infergo.WithOffline(true)` to require a
previously verified runtime and `infergo.WithDownloadRetries` to tune transient
download handling.

## Low-level API

Use `Runtime.Inspect` to discover model inputs and outputs, `Runtime.Load` to
create a session from that schema, and `Session.Run` or `Session.RunNamed` for
inference. `InspectBytes` and `LoadBytes` provide the same workflow for models
already held in memory. Schema-aware sessions validate input type, rank, and
fixed dimensions before crossing the native boundary and validate outputs on
return.

```go
session, err := runtime.Load("model.onnx")
if err != nil {
	log.Fatal(err)
}
defer session.Close()

input, err := infergo.NewTensor([]int64{1, 4}, []float32{1, 2, 3, 4})
if err != nil {
	log.Fatal(err)
}
outputs, err := session.Run(ctx, input)
if err != nil {
	log.Fatal(err)
}
values, err := outputs[0].Data[float32]()
if err != nil {
	log.Fatal(err)
}
```

Sessions support concurrent calls, context cancellation, model metadata,
sequential or parallel graph execution, thread counts, graph optimization,
memory arenas and patterns, profiling, optimized-model output, custom
operators, arbitrary session configuration, and these execution providers:

- Core ML
- CUDA
- TensorRT
- OpenVINO
- DirectML
- Generic provider names supported by the supplied ONNX Runtime build

DirectML sessions are automatically configured for its required sequential
execution and memory-pattern behavior, and their `Run` calls are serialized.

## Memory ownership

`NewTensor` copies both shape and data. `TakeTensor` copies the shape but adopts
a caller-owned data slice, avoiding a large allocation when the buffer was
created solely for inference. Do not mutate an adopted slice afterward.

`Tensor.Data` returns an independent copy. Performance-sensitive read-only
code can use `infergo.BorrowData`, which returns a zero-allocation view that
must not be modified. Numeric and boolean input buffers are pinned for the
entire native call, and returned tensors own Go memory independent of the
session.

## Cancellation and concurrency

All operations that can block or perform substantial work accept a
`context.Context`. Cancellation propagates through downloads, file-lock waits,
tokenization, image preprocessing, pooling, postprocessing, and ONNX Runtime
execution. A session can be used by multiple goroutines. `Session.Close` waits
for active calls, is safe to call concurrently, and returns the same teardown
result to every caller.

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

Set `INFERGO_CACHE_DIR` to reuse a downloaded runtime across runs.

Run the opt-in end-to-end suite against every pinned catalog model with:

```sh
INFERGO_MODEL_INTEGRATION=1 INFERGO_CACHE_DIR="$PWD/.cache/infergo" \
  go test -race -run '^TestCatalogModels$' .
```

The catalog suite also runs weekly and on demand in GitHub Actions. Fuzz targets
and allocation-reporting benchmarks are part of the test packages:

```sh
go test -fuzz=Fuzz -fuzztime=30s ./...
go test -run '^$' -bench=. -benchmem ./...
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full development and commit
conventions, [CHANGELOG.md](CHANGELOG.md) for notable changes, and
[SECURITY.md](SECURITY.md) for vulnerability reporting.

## License

[MIT](LICENSE)
