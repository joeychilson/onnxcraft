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
	defaultRetries     = 2
	initialRetryDelay  = 100 * time.Millisecond
)

// ErrNotCached is returned in offline mode when an artifact is unavailable.
var ErrNotCached = errors.New("modelhub: artifact is not cached")

// ErrCorrupt is returned when an offline cached artifact fails verification.
var ErrCorrupt = errors.New("modelhub: cached artifact is corrupt")

// HTTPError reports a non-successful artifact response.
type HTTPError struct {
	URL        string
	Status     string
	StatusCode int
}

// Error implements error.
func (e *HTTPError) Error() string {
	return fmt.Sprintf("modelhub: download %s: unexpected HTTP status %s", e.URL, e.Status)
}

// Artifact identifies one immutable model file.
type Artifact struct {
	Name       string
	URL        string
	SHA256     string
	Size       int64
	Repository string
	Revision   string
	License    string
}

// Progress reports the state of one download attempt. Total is -1 when the
// server and artifact metadata do not provide a size. Attempt is one-based.
type Progress struct {
	Artifact   Artifact
	Downloaded int64
	Total      int64
	Attempt    int
}

// ProgressFunc receives download progress. FetchAll and FetchBundle may call
// it concurrently, so implementations must be safe for concurrent use.
type ProgressFunc func(Progress)

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
	retries     int
	headers     http.Header
	progress    ProgressFunc
}

// Client downloads verified model artifacts into a local cache.
type Client struct {
	cacheDir    string
	client      *http.Client
	offline     bool
	maxSize     int64
	concurrency int
	retries     int
	headers     http.Header
	progress    ProgressFunc
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

// WithRetries sets the number of retries after a transient download failure.
// The default is two retries, for at most three requests.
func WithRetries(count int) Option {
	return func(config *clientConfig) error {
		if count < 0 || count > 10 {
			return errors.New("modelhub: retries must be between zero and ten")
		}
		config.retries = count
		return nil
	}
}

// WithRequestHeader adds a header to artifact download requests. It is useful
// for authentication against private model registries.
func WithRequestHeader(name, value string) Option {
	return func(config *clientConfig) error {
		if !validHeaderName(name) {
			return errors.New("modelhub: invalid request header name")
		}
		if strings.ContainsAny(value, "\r\n") {
			return errors.New("modelhub: invalid request header value")
		}
		if config.headers == nil {
			config.headers = make(http.Header)
		}
		config.headers.Set(name, value)
		return nil
	}
}

// WithProgress registers a callback for download progress.
func WithProgress(progress ProgressFunc) Option {
	return func(config *clientConfig) error {
		if progress == nil {
			return errors.New("modelhub: progress callback cannot be nil")
		}
		config.progress = progress
		return nil
	}
}

// New creates a model artifact client.
func New(options ...Option) (*Client, error) {
	config := clientConfig{
		client:      &http.Client{Timeout: 30 * time.Minute},
		maxSize:     defaultMaximumSize,
		concurrency: defaultConcurrency,
		retries:     defaultRetries,
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
		retries:     config.retries,
		headers:     config.headers.Clone(),
		progress:    config.progress,
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
		Name:       fileParts[len(fileParts)-1],
		URL:        "https://huggingface.co/" + strings.Join(parts, "/"),
		SHA256:     digest,
		Size:       size,
		Repository: repository,
		Revision:   revision,
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
	if validBundle(directory, bundle.Artifacts, digests, key) {
		return bundlePaths(directory, bundle.Artifacts), nil
	}
	parent := filepath.Dir(directory)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return BundlePaths{}, fmt.Errorf("modelhub: create bundle cache: %w", err)
	}
	unlock := lockArtifact(directory)
	defer unlock()
	processLock, err := filelock.Acquire(ctx, directory+".lock")
	if err != nil {
		return BundlePaths{}, fmt.Errorf("modelhub: lock bundle cache: %w", err)
	}
	defer func() { _ = processLock.Close() }()
	if validBundle(directory, bundle.Artifacts, digests, key) {
		return bundlePaths(directory, bundle.Artifacts), nil
	}

	sources, err := c.FetchAll(ctx, bundle.Artifacts)
	if err != nil {
		return BundlePaths{}, err
	}
	temporary, err := os.MkdirTemp(parent, ".bundle-*")
	if err != nil {
		return BundlePaths{}, fmt.Errorf("modelhub: create temporary bundle: %w", err)
	}
	defer func() { _ = os.RemoveAll(temporary) }()
	for index, artifact := range bundle.Artifacts {
		target := filepath.Join(temporary, artifact.Name)
		if err := linkOrCopy(sources[index], target); err != nil {
			return BundlePaths{}, fmt.Errorf("modelhub: install bundle artifact %q: %w", artifact.Name, err)
		}
	}
	if err := atomicfile.Write(filepath.Join(temporary, ".verified"), []byte(key+"\n"), 0o644); err != nil {
		return BundlePaths{}, fmt.Errorf("modelhub: write bundle marker: %w", err)
	}
	if err := replaceDirectory(temporary, directory); err != nil {
		return BundlePaths{}, fmt.Errorf("modelhub: install bundle: %w", err)
	}
	return bundlePaths(directory, bundle.Artifacts), nil
}

func validBundle(directory string, artifacts []Artifact, digests []string, key string) bool {
	marker, err := os.ReadFile(filepath.Join(directory, ".verified"))
	if err != nil || strings.TrimSpace(string(marker)) != key {
		return false
	}
	for index, artifact := range artifacts {
		valid, err := validFile(filepath.Join(directory, artifact.Name), artifact, digests[index])
		if err != nil || !valid {
			return false
		}
	}
	return true
}

func bundlePaths(directory string, artifacts []Artifact) BundlePaths {
	result := BundlePaths{Directory: directory, Files: make(map[string]string, len(artifacts))}
	for _, artifact := range artifacts {
		result.Files[artifact.Name] = filepath.Join(directory, artifact.Name)
	}
	return result
}

func linkOrCopy(source, target string) (resultErr error) {
	if err := os.Link(source, target); err == nil {
		return nil
	}
	input, err := os.Open(source)
	if err != nil {
		return err
	}
	defer func() { resultErr = errors.Join(resultErr, input.Close()) }()
	output, err := os.OpenFile(target, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	if _, err := io.Copy(output, input); err != nil {
		_ = output.Close()
		return err
	}
	if err := output.Sync(); err != nil {
		_ = output.Close()
		return err
	}
	return output.Close()
}

func replaceDirectory(source, target string) error {
	if err := os.Rename(source, target); err == nil {
		return nil
	}
	if _, err := os.Stat(target); err != nil {
		return err
	}
	placeholder, err := os.CreateTemp(filepath.Dir(target), ".stale-bundle-*")
	if err != nil {
		return err
	}
	stale := placeholder.Name()
	if err := placeholder.Close(); err != nil {
		return err
	}
	if err := os.Remove(stale); err != nil {
		return err
	}
	if err := os.Rename(target, stale); err != nil {
		return err
	}
	if err := os.Rename(source, target); err != nil {
		_ = os.Rename(stale, target)
		return err
	}
	return os.RemoveAll(stale)
}

func (c *Client) download(ctx context.Context, artifact Artifact, digest, target string) error {
	for attempt := 0; ; attempt++ {
		err := c.downloadOnce(ctx, artifact, digest, target, attempt+1)
		if err == nil {
			return nil
		}
		var retryable *retryableDownloadError
		if attempt >= c.retries || !errors.As(err, &retryable) {
			return err
		}
		delay := retryable.delay
		if delay <= 0 {
			delay = initialRetryDelay << attempt
		}
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return errors.Join(ctx.Err(), err)
		case <-timer.C:
		}
	}
}

type retryableDownloadError struct {
	err   error
	delay time.Duration
}

func (e *retryableDownloadError) Error() string {
	return e.err.Error()
}

func (e *retryableDownloadError) Unwrap() error {
	return e.err
}

func (c *Client) downloadOnce(
	ctx context.Context,
	artifact Artifact,
	digest, target string,
	attempt int,
) (resultErr error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, artifact.URL, nil)
	if err != nil {
		return fmt.Errorf("modelhub: create request: %w", err)
	}
	request.Header = c.headers.Clone()
	response, err := c.client.Do(request)
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		return &retryableDownloadError{err: fmt.Errorf("modelhub: download %s: %w", artifact.Name, err)}
	}
	defer func() { resultErr = errors.Join(resultErr, response.Body.Close()) }()
	if response.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 4<<10))
		httpErr := &HTTPError{URL: artifact.URL, Status: response.Status, StatusCode: response.StatusCode}
		if retryableStatus(response.StatusCode) {
			return &retryableDownloadError{err: httpErr, delay: retryAfter(response.Header.Get("Retry-After"), time.Now())}
		}
		return httpErr
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
	writer := io.Writer(temporary)
	if c.progress != nil {
		total := response.ContentLength
		if artifact.Size > 0 {
			total = artifact.Size
		}
		c.progress(Progress{Artifact: artifact, Total: total, Attempt: attempt})
		writer = &progressWriter{
			writer:   temporary,
			artifact: artifact,
			total:    total,
			attempt:  attempt,
			progress: c.progress,
		}
	}
	written, copyErr := copyLimited(writer, hasher, response.Body, c.maxSize)
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

type progressWriter struct {
	writer     io.Writer
	artifact   Artifact
	total      int64
	attempt    int
	downloaded int64
	progress   ProgressFunc
}

func (w *progressWriter) Write(data []byte) (int, error) {
	written, err := w.writer.Write(data)
	w.downloaded += int64(written)
	if written > 0 {
		w.progress(Progress{
			Artifact:   w.artifact,
			Downloaded: w.downloaded,
			Total:      w.total,
			Attempt:    w.attempt,
		})
	}
	return written, err
}

func retryableStatus(status int) bool {
	switch status {
	case http.StatusRequestTimeout, http.StatusTooEarly, http.StatusTooManyRequests,
		http.StatusInternalServerError, http.StatusBadGateway, http.StatusServiceUnavailable,
		http.StatusGatewayTimeout:
		return true
	default:
		return false
	}
}

func retryAfter(value string, now time.Time) time.Duration {
	if value == "" {
		return 0
	}
	if seconds, err := time.ParseDuration(value + "s"); err == nil {
		return max(0, seconds)
	}
	when, err := http.ParseTime(value)
	if err != nil {
		return 0
	}
	return max(0, when.Sub(now))
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

func validHeaderName(name string) bool {
	if name == "" || strings.EqualFold(name, "Host") || strings.EqualFold(name, "Content-Length") {
		return false
	}
	for _, character := range []byte(name) {
		if character >= 'a' && character <= 'z' || character >= 'A' && character <= 'Z' ||
			character >= '0' && character <= '9' || strings.ContainsRune("!#$%&'*+-.^_`|~", rune(character)) {
			continue
		}
		return false
	}
	return true
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
