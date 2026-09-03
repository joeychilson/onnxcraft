package modelhub

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"math"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/joeychilson/infergo/internal/atomicfile"
	"github.com/joeychilson/infergo/internal/filelock"
)

const (
	defaultMaximumSize = int64(10 << 30)
	defaultConcurrency = 4
)

// ErrNotCached is returned in offline mode when an artifact is unavailable.
var ErrNotCached = errors.New("modelhub: artifact is not cached")

// ErrCorrupt is returned when an offline cached artifact fails verification.
var ErrCorrupt = errors.New("modelhub: cached artifact is corrupt")

// Artifact identifies one immutable model file.
type Artifact struct {
	Name   string
	URL    string
	SHA256 string
	Size   int64
}

// Bundle groups artifacts that must share a directory, such as an ONNX graph
// and its external tensor data.
type Bundle struct {
	Name      string
	Artifacts []Artifact
}

// BundlePaths contains the verified local paths for a fetched bundle.
type BundlePaths struct {
	Directory string
	Files     map[string]string
}

// Option configures a Client.
type Option func(*clientConfig) error

type clientConfig struct {
	cacheDir    string
	client      *http.Client
	offline     bool
	maxSize     int64
	concurrency int
}

// Client downloads verified model artifacts into a local cache.
type Client struct {
	cacheDir    string
	client      *http.Client
	offline     bool
	maxSize     int64
	concurrency int
}

var artifactLocks = struct {
	sync.Mutex
	entries map[string]*artifactLock
}{entries: make(map[string]*artifactLock)}

type artifactLock struct {
	mutex sync.Mutex
	users int
}

// WithCacheDir stores artifacts beneath path.
func WithCacheDir(path string) Option {
	return func(config *clientConfig) error {
		if path == "" {
			return errors.New("modelhub: cache directory cannot be empty")
		}
		config.cacheDir = path
		return nil
	}
}

// WithHTTPClient sets the client used for downloads.
func WithHTTPClient(client *http.Client) Option {
	return func(config *clientConfig) error {
		if client == nil {
			return errors.New("modelhub: HTTP client cannot be nil")
		}
		config.client = client
		return nil
	}
}

// WithOffline disables network access and only returns cached artifacts.
func WithOffline(enabled bool) Option {
	return func(config *clientConfig) error {
		config.offline = enabled
		return nil
	}
}

// WithMaximumSize limits a single download. The default is 10 GiB.
func WithMaximumSize(bytes int64) Option {
	return func(config *clientConfig) error {
		if bytes < 1 || bytes == math.MaxInt64 {
			return errors.New("modelhub: maximum size must be positive")
		}
		config.maxSize = bytes
		return nil
	}
}

// WithConcurrency limits concurrent artifact downloads. The default is four.
func WithConcurrency(count int) Option {
	return func(config *clientConfig) error {
		if count < 1 {
			return errors.New("modelhub: concurrency must be positive")
		}
		config.concurrency = count
		return nil
	}
}

// New creates a model artifact client.
func New(options ...Option) (*Client, error) {
	config := clientConfig{
		client:      &http.Client{Timeout: 30 * time.Minute},
		maxSize:     defaultMaximumSize,
		concurrency: defaultConcurrency,
	}
	for _, option := range options {
		if option == nil {
			return nil, errors.New("modelhub: option cannot be nil")
		}
		if err := option(&config); err != nil {
			return nil, err
		}
	}
	if config.cacheDir == "" {
		cacheDir, err := os.UserCacheDir()
		if err != nil {
			return nil, fmt.Errorf("modelhub: locate user cache directory: %w", err)
		}
		config.cacheDir = filepath.Join(cacheDir, "infergo", "models")
	}
	absolute, err := filepath.Abs(config.cacheDir)
	if err != nil {
		return nil, fmt.Errorf("modelhub: resolve cache directory: %w", err)
	}
	return &Client{
		cacheDir:    absolute,
		client:      config.client,
		offline:     config.offline,
		maxSize:     config.maxSize,
		concurrency: config.concurrency,
	}, nil
}

// HuggingFace creates an artifact for a file hosted in a Hugging Face model
// repository. A pinned revision and the file's SHA-256 digest are recommended.
func HuggingFace(repository, revision, file, digest string, size int64) (Artifact, error) {
	repositoryParts, err := safePathParts(repository)
	if err != nil || len(repositoryParts) != 2 {
		return Artifact{}, errors.New("modelhub: Hugging Face repository must be owner/name")
	}
	if revision == "" || strings.ContainsAny(revision, "/\\") {
		return Artifact{}, errors.New("modelhub: invalid Hugging Face revision")
	}
	fileParts, err := safePathParts(file)
	if err != nil {
		return Artifact{}, errors.New("modelhub: invalid Hugging Face file path")
	}
	parts := make([]string, 0, len(repositoryParts)+len(fileParts)+2)
	parts = append(parts, repositoryParts...)
	parts = append(parts, "resolve", revision)
	parts = append(parts, fileParts...)
	for index := range parts {
		parts[index] = url.PathEscape(parts[index])
	}
	artifact := Artifact{
		Name:   fileParts[len(fileParts)-1],
		URL:    "https://huggingface.co/" + strings.Join(parts, "/"),
		SHA256: digest,
		Size:   size,
	}
	if _, err := validateArtifact(artifact); err != nil {
		return Artifact{}, err
	}
	return artifact, nil
}

// Fetch returns a verified local path, downloading artifact when necessary.
func (c *Client) Fetch(ctx context.Context, artifact Artifact) (string, error) {
	if c == nil {
		return "", errors.New("modelhub: nil client")
	}
	if ctx == nil {
		return "", errors.New("modelhub: context cannot be nil")
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	digest, err := validateArtifact(artifact)
	if err != nil {
		return "", err
	}
	target := filepath.Join(c.cacheDir, digest, artifact.Name)
	if err := c.fetchTarget(ctx, artifact, digest, target); err != nil {
		return "", err
	}
	return target, nil
}

func (c *Client) fetchTarget(ctx context.Context, artifact Artifact, digest, target string) error {
	if artifact.Size > c.maxSize {
		return fmt.Errorf("modelhub: artifact %s exceeds the %d-byte limit", artifact.Name, c.maxSize)
	}
	unlock := lockArtifact(target)
	defer unlock()
	if err := ctx.Err(); err != nil {
		return err
	}

	if valid, cacheErr := validFile(target, artifact, digest); cacheErr != nil && !errors.Is(cacheErr, ErrCorrupt) {
		return cacheErr
	} else if valid {
		return nil
	} else if c.offline {
		if cacheErr != nil {
			return cacheErr
		}
		return fmt.Errorf("%w: %s", ErrNotCached, artifact.Name)
	}
	if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
		return fmt.Errorf("modelhub: create cache directory: %w", err)
	}
	processLock, err := filelock.Acquire(ctx, target+".lock")
	if err != nil {
		return fmt.Errorf("modelhub: lock artifact cache: %w", err)
	}
	defer func() { _ = processLock.Close() }()
	if valid, cacheErr := validFile(target, artifact, digest); cacheErr != nil && !errors.Is(cacheErr, ErrCorrupt) {
		return cacheErr
	} else if valid {
		return nil
	}
	if err := c.download(ctx, artifact, digest, target); err != nil {
		return err
	}
	return nil
}

// FetchAll fetches independent artifacts and preserves their input order.
func (c *Client) FetchAll(ctx context.Context, artifacts []Artifact) ([]string, error) {
	if c == nil {
		return nil, errors.New("modelhub: nil client")
	}
	if ctx == nil {
		return nil, errors.New("modelhub: context cannot be nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	paths := make([]string, len(artifacts))
	errorsByIndex := parallel(c.concurrency, len(artifacts), func(index int) error {
		artifact := artifacts[index]
		path, err := c.Fetch(ctx, artifact)
		if err != nil {
			return fmt.Errorf("modelhub: fetch artifact %d: %w", index, err)
		}
		paths[index] = path
		return nil
	})
	for _, err := range errorsByIndex {
		if err != nil {
			return nil, err
		}
	}
	return paths, nil
}

// FetchBundle fetches artifacts into one immutable directory and returns each
// path by artifact name. This preserves ONNX external-data file references.
func (c *Client) FetchBundle(ctx context.Context, bundle Bundle) (BundlePaths, error) {
	if c == nil {
		return BundlePaths{}, errors.New("modelhub: nil client")
	}
	if ctx == nil {
		return BundlePaths{}, errors.New("modelhub: context cannot be nil")
	}
	if err := ctx.Err(); err != nil {
		return BundlePaths{}, err
	}
	if !safeFileName(bundle.Name) {
		return BundlePaths{}, errors.New("modelhub: bundle name must be a file-name-safe identifier")
	}
	if len(bundle.Artifacts) == 0 {
		return BundlePaths{}, errors.New("modelhub: bundle must contain at least one artifact")
	}
	identities := make([]string, len(bundle.Artifacts))
	digests := make([]string, len(bundle.Artifacts))
	seen := make(map[string]struct{}, len(bundle.Artifacts))
	for index, artifact := range bundle.Artifacts {
		digest, err := validateArtifact(artifact)
		if err != nil {
			return BundlePaths{}, fmt.Errorf("modelhub: invalid bundle artifact %d: %w", index, err)
		}
		if _, exists := seen[artifact.Name]; exists {
			return BundlePaths{}, fmt.Errorf("modelhub: duplicate bundle artifact name %q", artifact.Name)
		}
		seen[artifact.Name] = struct{}{}
		digests[index] = digest
		identities[index] = artifact.Name + ":" + digest
	}
	slices.Sort(identities)
	hasher := sha256.New()
	if _, err := io.WriteString(hasher, bundle.Name+"\n"+strings.Join(identities, "\n")); err != nil {
		return BundlePaths{}, fmt.Errorf("modelhub: hash bundle identity: %w", err)
	}
	key := hex.EncodeToString(hasher.Sum(nil))
	directory := filepath.Join(c.cacheDir, "bundles", bundle.Name+"-"+key[:16])
	result := BundlePaths{Directory: directory, Files: make(map[string]string, len(bundle.Artifacts))}
	targets := make([]string, len(bundle.Artifacts))
	errorsByIndex := parallel(c.concurrency, len(bundle.Artifacts), func(index int) error {
		artifact := bundle.Artifacts[index]
		target := filepath.Join(directory, artifact.Name)
		if err := c.fetchTarget(ctx, artifact, digests[index], target); err != nil {
			return fmt.Errorf("modelhub: fetch bundle artifact %q: %w", artifact.Name, err)
		}
		targets[index] = target
		return nil
	})
	for _, err := range errorsByIndex {
		if err != nil {
			return BundlePaths{}, err
		}
	}
	for index, artifact := range bundle.Artifacts {
		result.Files[artifact.Name] = targets[index]
	}
	return result, nil
}

func (c *Client) download(ctx context.Context, artifact Artifact, digest, target string) (resultErr error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, artifact.URL, nil)
	if err != nil {
		return fmt.Errorf("modelhub: create request: %w", err)
	}
	response, err := c.client.Do(request)
	if err != nil {
		return fmt.Errorf("modelhub: download %s: %w", artifact.Name, err)
	}
	defer func() { resultErr = errors.Join(resultErr, response.Body.Close()) }()
	if response.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 4<<10))
		return fmt.Errorf("modelhub: download %s: unexpected HTTP status %s", artifact.Name, response.Status)
	}
	if response.ContentLength > c.maxSize {
		return fmt.Errorf("modelhub: download %s exceeds the %d-byte limit", artifact.Name, c.maxSize)
	}
	if artifact.Size > 0 && response.ContentLength >= 0 && response.ContentLength != artifact.Size {
		return fmt.Errorf("modelhub: download %s reports %d bytes, want %d", artifact.Name, response.ContentLength, artifact.Size)
	}
	temporary, err := os.CreateTemp(filepath.Dir(target), ".download-*")
	if err != nil {
		return fmt.Errorf("modelhub: create temporary file: %w", err)
	}
	temporaryPath := temporary.Name()
	defer func() {
		if removeErr := os.Remove(temporaryPath); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
			resultErr = errors.Join(resultErr, fmt.Errorf("modelhub: remove temporary file: %w", removeErr))
		}
	}()

	hasher := sha256.New()
	written, copyErr := copyLimited(temporary, hasher, response.Body, c.maxSize)
	if copyErr == nil {
		copyErr = temporary.Sync()
	}
	closeErr := temporary.Close()
	if copyErr != nil {
		return errors.Join(fmt.Errorf("modelhub: download %s: %w", artifact.Name, copyErr), closeErr)
	}
	if closeErr != nil {
		return fmt.Errorf("modelhub: close downloaded file: %w", closeErr)
	}
	if artifact.Size > 0 && written != artifact.Size {
		return fmt.Errorf("modelhub: download %s contains %d bytes, want %d", artifact.Name, written, artifact.Size)
	}
	actualDigest := hex.EncodeToString(hasher.Sum(nil))
	if actualDigest != digest {
		return fmt.Errorf("modelhub: SHA-256 mismatch for %s: got %s, want %s", artifact.Name, actualDigest, digest)
	}
	if err := os.Chmod(temporaryPath, 0o644); err != nil {
		return fmt.Errorf("modelhub: set artifact permissions: %w", err)
	}
	if err := atomicfile.Replace(temporaryPath, target); err != nil {
		return fmt.Errorf("modelhub: cache artifact: %w", err)
	}
	return nil
}

func validateArtifact(artifact Artifact) (string, error) {
	if !safeFileName(artifact.Name) {
		return "", errors.New("modelhub: artifact name must be a file name")
	}
	parsedURL, err := url.Parse(artifact.URL)
	if err != nil || (parsedURL.Scheme != "https" && parsedURL.Scheme != "http") || parsedURL.Host == "" {
		return "", errors.New("modelhub: artifact URL must be an absolute HTTP or HTTPS URL")
	}
	if artifact.Size < 0 {
		return "", errors.New("modelhub: artifact size cannot be negative")
	}
	digest := strings.ToLower(artifact.SHA256)
	decoded, err := hex.DecodeString(digest)
	if err != nil || len(decoded) != sha256.Size {
		return "", errors.New("modelhub: artifact SHA-256 must contain 64 hexadecimal characters")
	}
	return digest, nil
}

func safeFileName(name string) bool {
	return name != "" && name == filepath.Base(name) && name != "." && name != ".." && !strings.ContainsAny(name, `/\\`)
}

func validFile(path string, artifact Artifact, digest string) (valid bool, resultErr error) {
	file, err := os.Open(path)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("modelhub: open cached artifact: %w", err)
	}
	defer func() { resultErr = errors.Join(resultErr, file.Close()) }()
	info, err := file.Stat()
	if err != nil {
		return false, fmt.Errorf("modelhub: inspect cached artifact: %w", err)
	}
	if !info.Mode().IsRegular() {
		return false, fmt.Errorf("%w: %s is not a regular file", ErrCorrupt, path)
	}
	if artifact.Size > 0 && info.Size() != artifact.Size {
		return false, fmt.Errorf("%w: %s has size %d, want %d", ErrCorrupt, path, info.Size(), artifact.Size)
	}
	hasher := sha256.New()
	if _, err := io.Copy(hasher, file); err != nil {
		return false, fmt.Errorf("modelhub: verify cached artifact: %w", err)
	}
	actual := hex.EncodeToString(hasher.Sum(nil))
	if actual != digest {
		return false, fmt.Errorf("%w: %s has SHA-256 %s, want %s", ErrCorrupt, path, actual, digest)
	}
	return true, nil
}

func copyLimited(destination io.Writer, hasher hash.Hash, source io.Reader, maximum int64) (int64, error) {
	writer := io.MultiWriter(destination, hasher)
	written, err := io.CopyN(writer, source, maximum)
	if err != nil && !errors.Is(err, io.EOF) {
		return written, err
	}
	if errors.Is(err, io.EOF) {
		return written, nil
	}
	var extra [1]byte
	read, readErr := source.Read(extra[:])
	if read > 0 {
		return written + int64(read), fmt.Errorf("modelhub: artifact exceeds the %d-byte limit", maximum)
	}
	if readErr != nil && !errors.Is(readErr, io.EOF) {
		return written, readErr
	}
	return written, nil
}

func lockArtifact(path string) func() {
	artifactLocks.Lock()
	entry := artifactLocks.entries[path]
	if entry == nil {
		entry = &artifactLock{}
		artifactLocks.entries[path] = entry
	}
	entry.users++
	artifactLocks.Unlock()
	entry.mutex.Lock()
	return func() {
		entry.mutex.Unlock()
		artifactLocks.Lock()
		entry.users--
		if entry.users == 0 {
			delete(artifactLocks.entries, path)
		}
		artifactLocks.Unlock()
	}
}

func parallel(concurrency, count int, function func(int) error) []error {
	errorsByIndex := make([]error, count)
	jobs := make(chan int)
	var workers sync.WaitGroup
	for range min(concurrency, count) {
		workers.Go(func() {
			for index := range jobs {
				errorsByIndex[index] = function(index)
			}
		})
	}
	for index := range count {
		jobs <- index
	}
	close(jobs)
	workers.Wait()
	return errorsByIndex
}

func safePathParts(path string) ([]string, error) {
	if path == "" || strings.ContainsRune(path, '\\') {
		return nil, errors.New("modelhub: invalid path")
	}
	parts := strings.Split(path, "/")
	for _, part := range parts {
		if part == "" || part == "." || part == ".." {
			return nil, errors.New("modelhub: invalid path")
		}
	}
	return parts, nil
}
