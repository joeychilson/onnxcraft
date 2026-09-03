package infergo

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func TestDownloadArtifactVerifiesDigest(t *testing.T) {
	t.Parallel()
	content := []byte("native runtime")
	digest := sha256.Sum256(content)
	client := &http.Client{Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
		return &http.Response{
			Status:        "200 OK",
			StatusCode:    http.StatusOK,
			Body:          io.NopCloser(strings.NewReader(string(content))),
			ContentLength: int64(len(content)),
			Header:        make(http.Header),
		}, nil
	})}

	path, err := downloadArtifact(
		t.Context(),
		client,
		"https://example.test/runtime",
		t.TempDir(),
		hex.EncodeToString(digest[:]),
		int64(len(content)),
	)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Remove(path) })
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(content) {
		t.Fatalf("download = %q", got)
	}
}

func TestDownloadArtifactRejectsBadResponses(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		status int
		digest string
	}{
		{name: "status", status: http.StatusNotFound, digest: "unused"},
		{name: "digest", status: http.StatusOK, digest: "incorrect"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			client := &http.Client{Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
				return &http.Response{
					Status:        http.StatusText(test.status),
					StatusCode:    test.status,
					Body:          io.NopCloser(strings.NewReader("response")),
					ContentLength: -1,
					Header:        make(http.Header),
				}, nil
			})}
			if _, err := downloadArtifact(t.Context(), client, "https://example.test/runtime", t.TempDir(), test.digest, 8); err == nil {
				t.Fatal("downloadArtifact() error = nil")
			}
		})
	}
}

func TestExtractLibrary(t *testing.T) {
	t.Parallel()
	for _, format := range []string{"zip", "tgz"} {
		t.Run(format, func(t *testing.T) {
			t.Parallel()
			directory := t.TempDir()
			archivePath := filepath.Join(directory, "download")
			libraryName := "libonnxruntime.test"
			content := []byte("library bytes")
			if format == "zip" {
				writeZip(t, archivePath, "runtime/lib/"+libraryName, content)
			} else {
				writeTarGzip(t, archivePath, "runtime/lib/"+libraryName, content)
			}
			destination := filepath.Join(directory, "library")
			file, createErr := os.Create(destination)
			if createErr != nil {
				t.Fatal(createErr)
			}
			if closeErr := file.Close(); closeErr != nil {
				t.Fatal(closeErr)
			}
			if extractErr := extractLibrary(archivePath, "runtime."+format, libraryName, destination); extractErr != nil {
				t.Fatal(extractErr)
			}
			got, err := os.ReadFile(destination)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != string(content) {
				t.Fatalf("extracted = %q", got)
			}
		})
	}
}

func TestVerifiedRuntimeExists(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	library := filepath.Join(directory, "library")
	marker := library + ".verified"
	if err := os.WriteFile(library, []byte("library"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(marker, []byte("digest\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if !verifiedRuntimeExists(library, marker, "digest") {
		t.Fatal("verifiedRuntimeExists() = false")
	}
	if verifiedRuntimeExists(library, marker, "other") {
		t.Fatal("verifiedRuntimeExists() accepted a stale marker")
	}
}

func TestReplaceFileReplacesExistingDestination(t *testing.T) {
	t.Parallel()
	directory := t.TempDir()
	source := filepath.Join(directory, "source")
	destination := filepath.Join(directory, "destination")
	if err := os.WriteFile(source, []byte("new"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(destination, []byte("old"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := replaceFile(source, destination); err != nil {
		t.Fatal(err)
	}
	content, err := os.ReadFile(destination)
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "new" {
		t.Fatalf("destination = %q", content)
	}
}

func TestRuntimeOptionsValidateInputs(t *testing.T) {
	t.Parallel()
	config := runtimeConfig{}
	if err := WithCacheDir("")(&config); err == nil {
		t.Fatal("WithCacheDir() error = nil")
	}
	if err := WithLibraryPath("")(&config); err == nil {
		t.Fatal("WithLibraryPath() error = nil")
	}
	if err := WithHTTPClient(nil)(&config); err == nil {
		t.Fatal("WithHTTPClient() error = nil")
	}
}

func writeZip(t *testing.T, path, name string, content []byte) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	archive := zip.NewWriter(file)
	entry, err := archive.Create(name)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := entry.Write(content); err != nil {
		t.Fatal(err)
	}
	if err := archive.Close(); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}

func writeTarGzip(t *testing.T, path, name string, content []byte) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	gzipWriter := gzip.NewWriter(file)
	archive := tar.NewWriter(gzipWriter)
	if err := archive.WriteHeader(&tar.Header{Name: name, Mode: 0o600, Size: int64(len(content))}); err != nil {
		t.Fatal(err)
	}
	if _, err := archive.Write(content); err != nil {
		t.Fatal(err)
	}
	if err := archive.Close(); err != nil {
		t.Fatal(err)
	}
	if err := gzipWriter.Close(); err != nil {
		t.Fatal(err)
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
}
