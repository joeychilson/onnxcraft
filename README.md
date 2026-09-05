# onnxcraft

[![CI](https://github.com/joeychilson/onnxcraft/actions/workflows/ci.yml/badge.svg)](https://github.com/joeychilson/onnxcraft/actions/workflows/ci.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/joeychilson/onnxcraft.svg)](https://pkg.go.dev/github.com/joeychilson/onnxcraft)
[![Release](https://img.shields.io/github/v/release/joeychilson/onnxcraft)](https://github.com/joeychilson/onnxcraft/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An ONNX inference library for Go, powered by ONNX Runtime.

Requires Go 1.27 or later, cgo, a C compiler, and ONNX Runtime 1.29 or later.
No Go module dependencies.

## Installation

```text
go get github.com/joeychilson/onnxcraft
```

Install an [ONNX Runtime shared library](https://github.com/microsoft/onnxruntime/releases)
for your platform and execution provider. Pass its path to `Open`, keeping any
provider libraries alongside it. The C headers are included in this module.

## Package

- `onnxcraft` — model loading, typed tensors, inference, reusable outputs, and execution provider configuration.

Supports dense numeric and boolean tensors, including float16, bfloat16,
scalars, empty tensors, and dynamic dimensions.

## Example

Add two tensors using the included `testdata/add.onnx` model. Run from the
repository root with `ONNXRUNTIME_SHARED_LIBRARY_PATH` set to your native library.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/joeychilson/onnxcraft"
)

func main() {
	rt, err := onnxcraft.Open(os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"))
	if err != nil {
		log.Fatal(err)
	}
	defer rt.Close()

	session, err := rt.Load("testdata/add.onnx", nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	a, err := onnxcraft.NewTensor([]int64{1, 3}, []float32{1, 2, 3})
	if err != nil {
		log.Fatal(err)
	}
	b, err := onnxcraft.NewTensor([]int64{1, 3}, []float32{10, 20, 30})
	if err != nil {
		log.Fatal(err)
	}

	outputs, err := session.Run(context.Background(), a, b)
	if err != nil {
		log.Fatal(err)
	}

	values, err := outputs[0].Data[float32]()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(values) // [11 22 33]
}
```

See [examples/basic](examples/basic/main.go) for the runnable example. Use
`session.Inputs()` and `session.Outputs()` to inspect your model's types,
dimensions, and argument order. `LoadBytes` accepts self-contained model data;
`Load` also supports models with external weights.

## Usage

Load a session once and reuse it across requests. `SessionOptions` configures
thread counts, graph optimizations, and execution providers such as CUDA,
TensorRT, CoreML, and OpenVINO. Providers require a compatible native library.

| Operation | Behavior |
| --- | --- |
| `NewTensor(shape, data)` | Copies the shape and shares the Go data slice. |
| `tensor.Data[T]()` | Returns a mutable view without copying. |
| `session.Run(ctx, inputs...)` | Returns independent, Go-owned output tensors. |
| `session.RunInto(ctx, outputs, inputs...)` | Writes into reusable output tensors with exact result shapes and types. |

Tensors need no cleanup. Close runtimes and sessions when finished; sessions
keep their runtime alive. CPU sessions allow concurrent runs. Sessions with
explicit execution providers serialize runs. Use independent output buffers for
concurrent callers, and do not access tensor data while inference writes it or
mutate data while inference reads it.

Cancellation affects only the current run and waits for native execution to
finish. Its latency depends on the operator and provider. After a failed or
canceled `RunInto`, output contents are unspecified.

## Benchmarks

Median of five runs on an Apple M2, macOS ARM64, Go 1.27.0, and ONNX Runtime
1.29.0 CPU. Uses `testdata/sum.onnx`, a float32 input of shape `[1, 10]`, one
intra-op thread, and `context.Background()`.

| Operation | Time/op | Go bytes/op | Go allocs/op |
| --- | ---: | ---: | ---: |
| LoadBytes | 74.48 µs | 256 | 9 |
| Run | 1.454 µs | 64 | 3 |
| RunInto | 1.303 µs | 0 | 0 |

These measure binding overhead on a small model. Go allocation counts exclude
native allocations; the warmed `RunInto` path reuses internal storage.
Cancelable contexts require additional bookkeeping.

## Development

Set `ONNXRUNTIME_SHARED_LIBRARY_PATH` to enable native integration tests.

```text
go vet ./...
GOEXPERIMENT=cgocheck2 go test -race -shuffle=on ./...
go build ./...
go test -run '^$' -bench BenchmarkNativeSession -benchmem -count=5 .
```

CI runs native tests on Linux AMD64/ARM64, macOS ARM64, and Windows AMD64.
Generate test models with `python3 testdata/generate.py`; no Python packages
are needed.

## Release

Push a semantic version tag in the form `vMAJOR.MINOR.PATCH`, optionally with a
prerelease suffix, to run CI and create a GitHub Release with generated notes.
The release publishes the Go module source and does not include binary artifacts.

## License

[MIT](LICENSE). The included ONNX Runtime headers are distributed under
[Microsoft's MIT license](internal/ort/LICENSE).
