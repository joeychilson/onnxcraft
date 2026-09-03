package modelhub

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"
	"testing"
	"time"
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

func TestOfflineCacheCorruption(t *testing.T) {
	t.Parallel()
	cache := t.TempDir()
	contents := []byte("model")
	digest := sha256.Sum256(contents)
	digestText := hex.EncodeToString(digest[:])
	artifact := Artifact{Name: "model.onnx", URL: "https://example.com/model.onnx", SHA256: digestText, Size: int64(len(contents))}
	target := filepath.Join(cache, digestText, artifact.Name)
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(target, []byte("wrong"), 0o600); err != nil {
		t.Fatal(err)
	}
	client, err := New(WithCacheDir(cache), WithOffline(true))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := client.Fetch(t.Context(), artifact); !errors.Is(err, ErrCorrupt) {
		t.Fatalf("Fetch() error = %v, want ErrCorrupt", err)
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
	if _, err := New(WithMaximumSize(math.MaxInt64)); err == nil {
		t.Fatal("New(WithMaximumSize(MaxInt64)) error = nil")
	}
	if _, err := New(WithConcurrency(0)); err == nil {
		t.Fatal("New(WithConcurrency(0)) error = nil")
	}
	if _, err := New(WithRetries(-1)); err == nil {
		t.Fatal("New(WithRetries(-1)) error = nil")
	}
}

func TestFetchRetriesTransientResponses(t *testing.T) {
	t.Parallel()
	contents := []byte("model")
	digest := sha256.Sum256(contents)
	var requests atomic.Int32
	httpClient := &http.Client{Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
		if requests.Add(1) == 1 {
			response := response(nil)
			response.StatusCode = http.StatusServiceUnavailable
			response.Status = "503 Service Unavailable"
			return response, nil
		}
		return response(contents), nil
	})}
	client, err := New(WithCacheDir(t.TempDir()), WithHTTPClient(httpClient), WithRetries(1))
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Fetch(t.Context(), Artifact{
		Name: "model.onnx", URL: "https://example.com/model.onnx",
		SHA256: hex.EncodeToString(digest[:]), Size: int64(len(contents)),
	})
	if err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 2 {
		t.Fatalf("requests = %d, want 2", requests.Load())
	}
}

func TestFetchDoesNotRetryPermanentResponses(t *testing.T) {
	t.Parallel()
	digest := sha256.Sum256([]byte("model"))
	var requests atomic.Int32
	httpClient := &http.Client{Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
		requests.Add(1)
		response := response(nil)
		response.StatusCode = http.StatusNotFound
		response.Status = "404 Not Found"
		return response, nil
	})}
	client, err := New(WithCacheDir(t.TempDir()), WithHTTPClient(httpClient))
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Fetch(t.Context(), Artifact{
		Name: "model.onnx", URL: "https://example.com/model.onnx", SHA256: hex.EncodeToString(digest[:]),
	})
	var httpErr *HTTPError
	if !errors.As(err, &httpErr) || httpErr.StatusCode != http.StatusNotFound {
		t.Fatalf("Fetch() error = %v", err)
	}
	if requests.Load() != 1 {
		t.Fatalf("requests = %d, want 1", requests.Load())
	}
}

func TestFetchAllUsesBoundedConcurrency(t *testing.T) {
	t.Parallel()
	started := make(chan struct{}, 3)
	release := make(chan struct{})
	contents := []byte("model")
	digest := sha256.Sum256(contents)
	httpClient := &http.Client{Transport: roundTripFunc(func(_ *http.Request) (*http.Response, error) {
		started <- struct{}{}
		<-release
		return response(contents), nil
	})}
	client, err := New(WithCacheDir(t.TempDir()), WithHTTPClient(httpClient), WithConcurrency(2))
	if err != nil {
		t.Fatal(err)
	}
	artifacts := make([]Artifact, 3)
	for index := range artifacts {
		artifacts[index] = Artifact{
			Name: fmt.Sprintf("model-%d.onnx", index), URL: fmt.Sprintf("https://example.com/model-%d.onnx", index),
			SHA256: hex.EncodeToString(digest[:]), Size: int64(len(contents)),
		}
	}
	done := make(chan error, 1)
	go func() {
		_, fetchErr := client.FetchAll(t.Context(), artifacts)
		done <- fetchErr
	}()
	<-started
	<-started
	select {
	case <-started:
		close(release)
		t.Fatal("FetchAll() exceeded its concurrency limit")
	case <-time.After(25 * time.Millisecond):
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
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
	marker, err := os.ReadFile(filepath.Join(result.Directory, ".verified"))
	if err != nil || len(strings.TrimSpace(string(marker))) != sha256.Size*2 {
		t.Fatalf("bundle marker = %q, %v", marker, err)
	}
}

func TestFetchBundleDoesNotExposePartialDirectory(t *testing.T) {
	t.Parallel()
	cache := t.TempDir()
	good := []byte("good")
	digest := sha256.Sum256(good)
	httpClient := &http.Client{Transport: roundTripFunc(func(request *http.Request) (*http.Response, error) {
		if strings.HasSuffix(request.URL.Path, "bad") {
			return response([]byte("bad")), nil
		}
		return response(good), nil
	})}
	client, err := New(WithCacheDir(cache), WithHTTPClient(httpClient))
	if err != nil {
		t.Fatal(err)
	}
	artifacts := []Artifact{
		{Name: "good", URL: "https://example.com/good", SHA256: hex.EncodeToString(digest[:]), Size: int64(len(good))},
		{Name: "bad", URL: "https://example.com/bad", SHA256: hex.EncodeToString(digest[:]), Size: int64(len(good))},
	}
	if _, fetchErr := client.FetchBundle(t.Context(), Bundle{Name: "transaction", Artifacts: artifacts}); fetchErr == nil {
		t.Fatal("FetchBundle() error = nil")
	}
	entries, err := os.ReadDir(filepath.Join(cache, "bundles"))
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if entry.IsDir() {
			t.Fatalf("FetchBundle() exposed partial directory %q", entry.Name())
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

func FuzzSafePathParts(f *testing.F) {
	f.Add("owner/model")
	f.Add("../model")
	f.Add("owner\\model")
	f.Fuzz(func(t *testing.T, path string) {
		parts, err := safePathParts(path)
		if err != nil {
			return
		}
		for _, part := range parts {
			if part == "" || part == "." || part == ".." || strings.ContainsAny(part, `/\\`) {
				t.Fatalf("unsafe accepted path part %q from %q", part, path)
			}
		}
	})
}
