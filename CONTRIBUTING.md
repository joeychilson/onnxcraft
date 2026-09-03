# Contributing to Infergo

Thank you for improving Infergo. Correctness and predictable ownership come
before convenience; measured performance comes before speculative complexity.

## Development setup

Install Go 1.27 or later, a C toolchain, and `golangci-lint` v2.13 or later.
Then run:

```sh
go mod verify
go mod tidy -diff
golangci-lint run ./...
go test -race -shuffle=on ./...
go build ./...
```

Native-boundary changes must also pass:

```sh
INFERGO_INTEGRATION=1 GOEXPERIMENT=cgocheck2 go test -race ./...
go test -asan ./...
```

Changes to catalog models, preprocessing, tokenization, or end-to-end pipeline
semantics should run the pinned-model suite:

```sh
INFERGO_MODEL_INTEGRATION=1 INFERGO_CACHE_DIR="$PWD/.cache/infergo" \
  go test -race -run '^TestCatalogModels$' .
```

## Change guidelines

- Preserve backward compatibility unless a breaking change is explicitly
  justified and scheduled for a major release.
- Add tests that fail before the fix and cover error paths as well as success.
- Add or update a benchmark for performance-sensitive code, and include the
  before/after result in the pull request.
- Propagate contexts and wrapped errors; never silently discard cleanup errors.
- Document slice ownership, concurrency guarantees, and native-resource
  lifetimes for exported APIs.
- Keep dependencies minimal and pin downloadable artifacts by immutable
  revision, exact byte size, and SHA-256 digest.
- Update `CHANGELOG.md` for user-visible changes.

Run `gofmt` on edited Go files. The repository uses Conventional Commits such
as `fix: harden session teardown`, `feat: support in-memory models`, and
`perf: avoid output copies`. Keep commits focused so each can be reviewed and
reverted independently.

## Pull requests

Describe the observable behavior, why the design is correct, and how it was
verified. Note platform or execution-provider limitations explicitly. Pull
requests must pass lint, unit/race tests, builds, vulnerability scanning, and
the relevant native or catalog integration jobs.

By contributing, you agree that your contribution is licensed under the
repository's MIT license.
