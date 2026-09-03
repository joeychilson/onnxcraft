package infergo

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"

	ort "github.com/yalue/onnxruntime_go"
)

// RuntimeVersion is the native ONNX Runtime version used by infergo.
const RuntimeVersion = "1.29.0"

// ErrRuntimeNotCached is returned when offline mode cannot find a verified
// bundled native runtime.
var ErrRuntimeNotCached = errors.New("infergo: native runtime is not cached")

// Runtime owns a reference to the process-wide ONNX Runtime environment.
// Close is safe to call more than once. Sessions retain their own reference,
// so closing Runtime does not invalidate sessions that are still open.
type Runtime struct {
	mu            sync.Mutex
	libraryPath   string
	loadedVersion string
	closed        bool
}

// RuntimeInfo describes the native runtime selected for this process.
type RuntimeInfo struct {
	Version     string
	LibraryPath string
	OS          string
	Arch        string
}

// ExecutionProviderDevice describes one hardware target advertised by an
// execution-provider plugin registered with ONNX Runtime.
type ExecutionProviderDevice struct {
	Provider string
	Vendor   string
}

// RuntimeOption configures Open.
type RuntimeOption func(*runtimeConfig) error

type runtimeConfig struct {
	cacheDir        string
	library         string
	httpClient      *http.Client
	downloadRetries int
	offline         bool
}

var environment struct {
	sync.Mutex
	libraryPath string
	libraryInfo os.FileInfo
	version     string
	references  int
}

// WithCacheDir stores downloaded native runtime files beneath path.
func WithCacheDir(path string) RuntimeOption {
	return func(config *runtimeConfig) error {
		if path == "" {
			return errors.New("infergo: cache directory cannot be empty")
		}
		config.cacheDir = path
		return nil
	}
}

// WithLibraryPath uses an existing ONNX Runtime shared library and disables
// automatic downloading.
func WithLibraryPath(path string) RuntimeOption {
	return func(config *runtimeConfig) error {
		if path == "" {
			return errors.New("infergo: library path cannot be empty")
		}
		config.library = path
		return nil
	}
}

// WithHTTPClient sets the client used to download ONNX Runtime.
func WithHTTPClient(client *http.Client) RuntimeOption {
	return func(config *runtimeConfig) error {
		if client == nil {
			return errors.New("infergo: HTTP client cannot be nil")
		}
		config.httpClient = client
		return nil
	}
}

// WithOffline disables native runtime downloads. A custom library or a
// previously verified bundled runtime must be available.
func WithOffline(enabled bool) RuntimeOption {
	return func(config *runtimeConfig) error {
		config.offline = enabled
		return nil
	}
}

// WithDownloadRetries sets the number of retries after a transient native
// runtime download failure. The default is two.
func WithDownloadRetries(count int) RuntimeOption {
	return func(config *runtimeConfig) error {
		if count < 0 || count > 10 {
			return errors.New("infergo: download retries must be between zero and ten")
		}
		config.downloadRetries = count
		return nil
	}
}

// Open initializes ONNX Runtime. When no library path is provided, Open uses
// ONNXRUNTIME_SHARED_LIBRARY_PATH or downloads a verified official artifact.
func Open(ctx context.Context, options ...RuntimeOption) (*Runtime, error) {
	if ctx == nil {
		return nil, errors.New("infergo: context cannot be nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	config := defaultRuntimeConfig()
	for _, option := range options {
		if option == nil {
			return nil, errors.New("infergo: runtime option cannot be nil")
		}
		if optionErr := option(&config); optionErr != nil {
			return nil, optionErr
		}
	}
	if config.library == "" {
		config.library = os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH")
	}

	libraryPath := config.library
	if libraryPath == "" {
		if config.cacheDir == "" {
			cacheDir, cacheErr := os.UserCacheDir()
			if cacheErr != nil {
				return nil, fmt.Errorf("infergo: locate user cache directory: %w", cacheErr)
			}
			config.cacheDir = filepath.Join(cacheDir, "infergo")
		}
		downloadedPath, err := ensureRuntime(ctx, config)
		if err != nil {
			return nil, err
		}
		libraryPath = downloadedPath
	}
	validatedPath, err := validateLibrary(libraryPath)
	if err != nil {
		return nil, err
	}
	libraryPath = validatedPath
	loadedVersion, err := acquireEnvironment(libraryPath)
	if err != nil {
		return nil, err
	}

	return &Runtime{libraryPath: libraryPath, loadedVersion: loadedVersion}, nil
}

// Info returns details about the selected native runtime.
func (r *Runtime) Info() (RuntimeInfo, error) {
	if r == nil {
		return RuntimeInfo{}, errors.New("infergo: nil runtime")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return RuntimeInfo{}, errors.New("infergo: runtime is closed")
	}
	return RuntimeInfo{
		Version:     r.loadedVersion,
		LibraryPath: r.libraryPath,
		OS:          currentOS,
		Arch:        currentArch,
	}, nil
}

// LoadedVersion returns the version reported by the loaded native library.
func (r *Runtime) LoadedVersion() (string, error) {
	if r == nil {
		return "", errors.New("infergo: nil runtime")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return "", errors.New("infergo: runtime is closed")
	}
	return r.loadedVersion, nil
}

// ExecutionProviderDevices returns the hardware targets currently advertised
// by registered execution-provider plugins. The returned values are detached
// from ONNX Runtime and remain safe to use after Runtime is closed.
func (r *Runtime) ExecutionProviderDevices() ([]ExecutionProviderDevice, error) {
	if r == nil {
		return nil, errors.New("infergo: nil runtime")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, errors.New("infergo: runtime is closed")
	}
	raw, err := ort.GetEpDevices()
	if err != nil {
		return nil, fmt.Errorf("infergo: discover execution-provider devices: %w", err)
	}
	devices := make([]ExecutionProviderDevice, len(raw))
	for index, device := range raw {
		devices[index] = ExecutionProviderDevice{
			Provider: device.EpName(),
			Vendor:   device.EpVendor(),
		}
	}
	return devices, nil
}

// Close releases this Runtime's reference to the native environment.
func (r *Runtime) Close() error {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil
	}
	if err := releaseEnvironment(); err != nil {
		return err
	}
	r.closed = true
	return nil
}

func (r *Runtime) retain() (func() error, error) {
	if r == nil {
		return nil, errors.New("infergo: nil runtime")
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		return nil, errors.New("infergo: runtime is closed")
	}
	if _, err := acquireEnvironment(r.libraryPath); err != nil {
		return nil, err
	}
	var once sync.Once
	var releaseErr error
	return func() error {
		once.Do(func() { releaseErr = releaseEnvironment() })
		return releaseErr
	}, nil
}

func defaultRuntimeConfig() runtimeConfig {
	return runtimeConfig{
		httpClient: &http.Client{
			Timeout: 5 * time.Minute,
		},
		downloadRetries: 2,
	}
}

func validateLibrary(path string) (string, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return "", fmt.Errorf("infergo: resolve library path: %w", err)
	}
	resolved, err := filepath.EvalSymlinks(absolute)
	if err != nil {
		return "", fmt.Errorf("infergo: resolve ONNX Runtime library symlinks: %w", err)
	}
	info, err := os.Stat(resolved)
	if err != nil {
		return "", fmt.Errorf("infergo: inspect ONNX Runtime library: %w", err)
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("infergo: ONNX Runtime library %q is not a regular file", resolved)
	}
	return resolved, nil
}

func acquireEnvironment(libraryPath string) (string, error) {
	environment.Lock()
	defer environment.Unlock()
	if environment.references > 0 {
		sameLibrary := environment.libraryPath == libraryPath
		if !sameLibrary {
			info, err := os.Stat(libraryPath)
			if err != nil {
				return "", fmt.Errorf("infergo: inspect ONNX Runtime library: %w", err)
			}
			sameLibrary = os.SameFile(environment.libraryInfo, info)
		}
		if !sameLibrary {
			return "", fmt.Errorf("infergo: ONNX Runtime is already initialized with %q", environment.libraryPath)
		}
		environment.references++
		return environment.version, nil
	}

	libraryInfo, err := os.Stat(libraryPath)
	if err != nil {
		return "", fmt.Errorf("infergo: inspect ONNX Runtime library: %w", err)
	}
	ort.SetSharedLibraryPath(libraryPath)
	if err := ort.InitializeEnvironment(ort.WithLogLevelError()); err != nil {
		return "", fmt.Errorf("infergo: initialize ONNX Runtime: %w", err)
	}
	version := ort.GetVersion()
	environment.libraryPath = libraryPath
	environment.libraryInfo = libraryInfo
	environment.version = version
	environment.references = 1
	return version, nil
}

func releaseEnvironment() error {
	environment.Lock()
	defer environment.Unlock()
	if environment.references == 0 {
		return nil
	}
	if environment.references > 1 {
		environment.references--
		return nil
	}
	if err := ort.DestroyEnvironment(); err != nil {
		return fmt.Errorf("infergo: close ONNX Runtime: %w", err)
	}
	environment.references = 0
	environment.libraryPath = ""
	environment.libraryInfo = nil
	environment.version = ""
	return nil
}
