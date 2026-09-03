package modelhub

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
)

func TestFetchDownloadsVerifiesAndCaches(t *testing.T) {
	t.Parallel()
	contents := []byte("verified model")
	digest := sha256.Sum256(contents)
	var requests atomic.Int32
	httpClient := &http.Client{Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
		requests.Add(1)
		return response(contents), nil
	})}
	client, err := New(WithCacheDir(t.TempDir()), WithHTTPClient(httpClient))
	if err != nil {
		t.Fatal(err)
	}
	artifact := Artifact{
		Name: "model.onnx", URL: "https://example.com/model.onnx",
		SHA256: hex.EncodeToString(digest[:]), Size: int64(len(contents)),
	}
	first, err := client.Fetch(t.Context(), artifact)
	if err != nil {
		t.Fatal(err)
	}
	second, err := client.Fetch(t.Context(), artifact)
	if err != nil {
		t.Fatal(err)
	}
	if first != second || requests.Load() != 1 {
		t.Fatalf("paths = %q, %q; requests = %d", first, second, requests.Load())
	}
	cached, err := os.ReadFile(first)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(cached, contents) {
		t.Fatalf("cached = %q", cached)
	}
}

func TestFetchRejectsBadDigest(t *testing.T) {
	t.Parallel()
	httpClient := &http.Client{Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
		return response([]byte("wrong")), nil
	})}
	client, err := New(WithCacheDir(t.TempDir()), WithHTTPClient(httpClient))
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256([]byte("expected"))
	_, err = client.Fetch(t.Context(), Artifact{
		Name: "model.onnx", URL: "https://example.com/model.onnx", SHA256: hex.EncodeToString(digest[:]),
	})
	if err == nil {
		t.Fatal("Fetch() error = nil")
	}
}

func TestOfflineCacheMiss(t *testing.T) {
	t.Parallel()
	client, err := New(WithCacheDir(t.TempDir()), WithOffline(true))
	if err != nil {
		t.Fatal(err)
	}
	digest := sha256.Sum256([]byte("model"))
	_, err = client.Fetch(t.Context(), Artifact{
		Name: "model.onnx", URL: "https://example.com/model.onnx", SHA256: hex.EncodeToString(digest[:]),
	})
	if !errors.Is(err, ErrNotCached) {
		t.Fatalf("Fetch() error = %v", err)
	}
}

func TestHuggingFaceArtifact(t *testing.T) {
	t.Parallel()
	digest := sha256.Sum256([]byte("model"))
	artifact, err := HuggingFace(
		"sentence-transformers/all-MiniLM-L6-v2",
		"dfa9feb5cece5be2cc8fc23a3cf1f32473a9d56f",
		"onnx/model.onnx",
		hex.EncodeToString(digest[:]),
		123,
	)
	if err != nil {
		t.Fatal(err)
	}
	want := "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/dfa9feb5cece5be2cc8fc23a3cf1f32473a9d56f/onnx/model.onnx"
	if artifact.URL != want || artifact.Name != "model.onnx" {
		t.Fatalf("artifact = %+v", artifact)
	}
	if _, err := HuggingFace("invalid", "main", "model.onnx", artifact.SHA256, 1); err == nil {
		t.Fatal("HuggingFace() error = nil")
	}
}

func TestMaximumSize(t *testing.T) {
	t.Parallel()
	if _, err := New(WithMaximumSize(0)); err == nil {
		t.Fatal("New() error = nil")
	}
}

func TestFetchBundleKeepsFilesAdjacent(t *testing.T) {
	t.Parallel()
	contents := map[string][]byte{
		"model.onnx":      []byte("graph"),
		"model.onnx_data": []byte("weights"),
	}
	httpClient := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		return response(contents[request.URL.Path[1:]]), nil
	})}
	client, err := New(WithCacheDir(t.TempDir()), WithHTTPClient(httpClient))
	if err != nil {
		t.Fatal(err)
	}
	artifacts := make([]Artifact, 0, len(contents))
	for name, data := range contents {
		digest := sha256.Sum256(data)
		artifacts = append(artifacts, Artifact{
			Name: name, URL: "https://example.com/" + name,
			SHA256: hex.EncodeToString(digest[:]), Size: int64(len(data)),
		})
	}
	result, err := client.FetchBundle(t.Context(), Bundle{Name: "external-model", Artifacts: artifacts})
	if err != nil {
		t.Fatal(err)
	}
	for name, path := range result.Files {
		if filepath.Dir(path) != result.Directory {
			t.Fatalf("%s is not in %s", path, result.Directory)
		}
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			t.Fatal(readErr)
		}
		if !slices.Equal(data, contents[name]) {
			t.Fatalf("%s = %q", name, data)
		}
	}
}

func TestCatalogArtifactsAreValidAndPinned(t *testing.T) {
	t.Parallel()
	artifacts := []Artifact{BERTBaseUncased(), AllMiniLML6V2(), ResNet50(), YOLOSSmall()}
	for _, artifact := range artifacts {
		if _, err := validateArtifact(artifact); err != nil {
			t.Errorf("%s: %v", artifact.Name, err)
		}
		if artifact.Size < 1 || !strings.Contains(artifact.URL, "/resolve/") {
			t.Errorf("artifact is not pinned: %+v", artifact)
		}
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (function roundTripFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return function(request)
}

func response(contents []byte) *http.Response {
	return &http.Response{
		StatusCode:    http.StatusOK,
		Status:        "200 OK",
		Body:          io.NopCloser(bytes.NewReader(contents)),
		ContentLength: int64(len(contents)),
		Header:        make(http.Header),
	}
}
