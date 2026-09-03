## Summary

Describe the observable change and why this design is correct.

## Verification

List the exact tests, race runs, integration runs, fuzzing, profiles, or
benchmarks performed. Include before/after numbers for performance changes.

## Checklist

- [ ] The change is focused and uses a Conventional Commit title.
- [ ] Public API compatibility and error semantics were considered.
- [ ] Exported API documents ownership, concurrency, and resource lifetimes.
- [ ] New behavior and important failure paths have tests.
- [ ] `go test -race -shuffle=on ./...` passes.
- [ ] `golangci-lint run ./...` passes.
- [ ] Native or catalog integration tests were run when relevant.
- [ ] `CHANGELOG.md` and user documentation were updated when relevant.
- [ ] Downloaded artifacts are pinned by revision, size, SHA-256, and license.
