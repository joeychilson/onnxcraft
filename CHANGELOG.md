# Changelog

All notable changes to Infergo are documented here. This project follows
[Semantic Versioning](https://semver.org/) and
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- Schema-aware sessions with input/output validation and model metadata.
- In-memory model inspection and loading.
- Session controls for execution mode, memory behavior, logging, profiling,
  optimized graphs, custom operators, provider configuration, and thread use.
- Runtime execution-provider device discovery and offline operation.
- Context-aware tokenization and image preprocessing.
- Model download concurrency, retries, range resumption, authentication
  headers, progress callbacks, offline corruption errors, and transactional
  multi-file bundles.
- Immutable catalog provenance and license metadata.
- Real-model golden tests, fuzz targets, benchmarks, multi-platform native
  testing, strict cgo checks, and AddressSanitizer coverage.

### Changed

- Upgraded the module baseline and continuous integration to Go 1.27.
- Aligned ResNet preprocessing and YOLOS no-object/NMS behavior with their
  upstream processors.
- Reduced allocation and algorithmic overhead throughout vision,
  tokenization, classification, detection, and tensor conversion paths.
- Pinned third-party GitHub Actions by immutable commit.

### Fixed

- Tensor scalar, zero-dimension, overflow, and ownership behavior.
- Go-pointer lifetime safety at the cgo boundary.
- Concurrent session close/run ordering, cancellation teardown, and native
  environment reference accounting.
- WordPiece Unicode controls, custom special tokens, duplicate tokens, and
  sequence limits.
- Cache races, partial bundle visibility, corrupt offline entries, native
  runtime identity checks, and interrupted downloads.
- Stable softmax, cosine similarity, top-k ties, DETR no-object handling, and
  non-finite numeric validation.

[Unreleased]: https://github.com/joeychilson/infergo/commits/main
