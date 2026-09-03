# Security policy

## Supported versions

Security fixes are applied to the latest released minor version. Users should
upgrade to the newest release before reporting an issue that may already be
fixed.

## Reporting a vulnerability

Do not disclose a suspected vulnerability in a public issue. Use GitHub's
[private vulnerability reporting](https://github.com/joeychilson/infergo/security/advisories/new)
to provide:

- the affected Infergo and Go versions;
- operating system, architecture, and ONNX Runtime version;
- a minimal reproducer or malformed artifact when safe to share;
- impact, prerequisites, and any known mitigations.

You should receive an acknowledgment within seven days. A fix, coordinated
disclosure date, and credit will be discussed after validation.

## Security boundaries

Infergo verifies automatically downloaded runtime and catalog artifacts by
exact size and SHA-256 before installation. Cache installation is atomic and
coordinated across threads and processes. Offline mode never initiates a
download.

Custom ONNX Runtime libraries, custom operator libraries, execution-provider
plugins, and ONNX model files are native-code or native-parser trust inputs.
Only load them from trusted sources. A digest proves content identity, not that
the content is safe. Applications processing untrusted models should isolate
inference at the process or container boundary and keep ONNX Runtime current.

Authentication values supplied with `modelhub.WithRequestHeader` are held in
memory and attached to artifact requests. Use a dedicated HTTP client with an
appropriate redirect policy when credentials must never cross hosts.
