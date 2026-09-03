package infergo

import (
	"archive/tar"
	"archive/zip"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

const releaseBaseURL = "https://github.com/microsoft/onnxruntime/releases/download/v" + RuntimeVersion

var (
	currentOS   = runtime.GOOS
	currentArch = runtime.GOARCH
)

type runtimeArtifact struct {
	archiveName string
	libraryName string
	digest      string
	size        int64
}

// Digests and sizes are from the official ONNX Runtime v1.29.0 release assets.
var runtimeArtifacts = map[string]runtimeArtifact{
	"darwin/arm64": {
		archiveName: "onnxruntime-osx-arm64-1.29.0.tgz",
		libraryName: "libonnxruntime.1.29.0.dylib",
		digest:      "d0706fc34f315d8c88639d0a8c81f2e09e815f282cabed3493c06a054352cf92",
		size:        41_578_864,
	},
	"linux/amd64": {
		archiveName: "onnxruntime-linux-x64-1.29.0.tgz",
		libraryName: "libonnxruntime.so.1.29.0",
		digest:      "c3fddc4f139a045b0c4902c57410f0694f1c2fdf9b6939fbe38b1aeae7cd14ba",
		size:        11_082_880,
	},
	"linux/arm64": {
		archiveName: "onnxruntime-linux-aarch64-1.29.0.tgz",
		libraryName: "libonnxruntime.so.1.29.0",
		digest:      "e1799098ebc054b370f6176a450f158720f297818c613e5dc99b92e2ec82346f",
		size:        10_027_600,
	},
	"windows/amd64": {
		archiveName: "onnxruntime-win-x64-1.29.0.zip",
		libraryName: "onnxruntime.dll",
		digest:      "c9b4b7086b529ad814f428c1bad028e20a25d7dc0699836775faace4ab5b78b2",
		size:        79_645_520,
	},
	"windows/arm64": {
		archiveName: "onnxruntime-win-arm64-1.29.0.zip",
		libraryName: "onnxruntime.dll",
		digest:      "a094a49c3ced0f9fca554647cc7566ae99d93a63a8ce6bf47975561c2de7608e",
		size:        81_679_033,
	},
}

func ensureRuntime(ctx context.Context, config runtimeConfig) (string, error) {
	artifact, ok := runtimeArtifacts[currentOS+"/"+currentArch]
	if !ok {
		return "", fmt.Errorf("infergo: automatic ONNX Runtime installation is unsupported on %s/%s; use WithLibraryPath", currentOS, currentArch)
	}

	runtimeDir := filepath.Join(config.cacheDir, "onnxruntime", RuntimeVersion, currentOS+"-"+currentArch)
	if err := os.MkdirAll(runtimeDir, 0o755); err != nil {
		return "", fmt.Errorf("infergo: create runtime cache: %w", err)
	}
	libraryPath := filepath.Join(runtimeDir, artifact.libraryName)
	markerPath := libraryPath + ".verified"
	if verifiedRuntimeExists(libraryPath, markerPath, artifact.digest) {
		return libraryPath, nil
	}

	archiveFile, err := downloadArtifact(
		ctx,
		config.httpClient,
		releaseBaseURL+"/"+artifact.archiveName,
		runtimeDir,
		artifact.digest,
		artifact.size,
	)
	if err != nil {
		return "", err
	}
	defer func() { _ = os.Remove(archiveFile) }()

	temporaryLibrary, err := os.CreateTemp(runtimeDir, ".onnxruntime-*")
	if err != nil {
		return "", fmt.Errorf("infergo: create temporary runtime library: %w", err)
	}
	temporaryPath := temporaryLibrary.Name()
	if err := temporaryLibrary.Close(); err != nil {
		return "", fmt.Errorf("infergo: close temporary runtime library: %w", err)
	}
	defer func() { _ = os.Remove(temporaryPath) }()

	if err := extractLibrary(archiveFile, artifact.archiveName, artifact.libraryName, temporaryPath); err != nil {
		return "", err
	}
	if err := os.Chmod(temporaryPath, 0o755); err != nil {
		return "", fmt.Errorf("infergo: make runtime library executable: %w", err)
	}
	if err := replaceFile(temporaryPath, libraryPath); err != nil {
		return "", fmt.Errorf("infergo: install runtime library: %w", err)
	}
	if err := writeMarker(markerPath, artifact.digest); err != nil {
		return "", err
	}
	return libraryPath, nil
}

func verifiedRuntimeExists(libraryPath, markerPath, digest string) bool {
	info, err := os.Stat(libraryPath)
	if err != nil || !info.Mode().IsRegular() {
		return false
	}
	marker, err := os.ReadFile(markerPath)
	return err == nil && strings.TrimSpace(string(marker)) == digest
}

func downloadArtifact(
	ctx context.Context,
	client *http.Client,
	url string,
	directory string,
	expectedDigest string,
	expectedSize int64,
) (string, error) {
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return "", fmt.Errorf("infergo: create runtime download request: %w", err)
	}
	response, err := client.Do(request)
	if err != nil {
		return "", fmt.Errorf("infergo: download ONNX Runtime: %w", err)
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusOK {
		_, _ = io.Copy(io.Discard, io.LimitReader(response.Body, 4<<10))
		return "", fmt.Errorf("infergo: download ONNX Runtime: unexpected HTTP status %s", response.Status)
	}
	if response.ContentLength >= 0 && response.ContentLength != expectedSize {
		return "", fmt.Errorf("infergo: ONNX Runtime download size is %d bytes, want %d", response.ContentLength, expectedSize)
	}

	file, err := os.CreateTemp(directory, ".onnxruntime-download-*")
	if err != nil {
		return "", fmt.Errorf("infergo: create runtime download: %w", err)
	}
	path := file.Name()
	remove := true
	defer func() {
		_ = file.Close()
		if remove {
			_ = os.Remove(path)
		}
	}()

	hash := sha256.New()
	written, err := io.Copy(io.MultiWriter(file, hash), io.LimitReader(response.Body, expectedSize+1))
	if err != nil {
		return "", fmt.Errorf("infergo: save ONNX Runtime download: %w", err)
	}
	if written != expectedSize {
		return "", fmt.Errorf("infergo: ONNX Runtime download size is %d bytes, want %d", written, expectedSize)
	}
	if err := file.Close(); err != nil {
		return "", fmt.Errorf("infergo: close ONNX Runtime download: %w", err)
	}
	actualDigest := hex.EncodeToString(hash.Sum(nil))
	if actualDigest != expectedDigest {
		return "", fmt.Errorf("infergo: ONNX Runtime checksum mismatch: got %s, want %s", actualDigest, expectedDigest)
	}
	remove = false
	return path, nil
}

func extractLibrary(archivePath, archiveName, libraryName, destination string) error {
	if strings.HasSuffix(archiveName, ".zip") {
		return extractZipLibrary(archivePath, libraryName, destination)
	}
	return extractTarLibrary(archivePath, libraryName, destination)
}

func extractZipLibrary(archivePath, libraryName, destination string) error {
	archive, err := zip.OpenReader(archivePath)
	if err != nil {
		return fmt.Errorf("infergo: open runtime zip: %w", err)
	}
	defer func() { _ = archive.Close() }()
	for _, file := range archive.File {
		if filepath.Base(filepath.ToSlash(file.Name)) != libraryName || file.FileInfo().IsDir() {
			continue
		}
		reader, err := file.Open()
		if err != nil {
			return fmt.Errorf("infergo: open runtime library in zip: %w", err)
		}
		err = copyLibrary(destination, reader)
		closeErr := reader.Close()
		return errors.Join(err, closeErr)
	}
	return fmt.Errorf("infergo: runtime library %q not found in zip", libraryName)
}

func extractTarLibrary(archivePath, libraryName, destination string) error {
	file, err := os.Open(archivePath)
	if err != nil {
		return fmt.Errorf("infergo: open runtime archive: %w", err)
	}
	defer func() { _ = file.Close() }()
	gzipReader, err := gzip.NewReader(file)
	if err != nil {
		return fmt.Errorf("infergo: open runtime gzip stream: %w", err)
	}
	defer func() { _ = gzipReader.Close() }()

	reader := tar.NewReader(gzipReader)
	for {
		header, err := reader.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return fmt.Errorf("infergo: read runtime archive: %w", err)
		}
		if filepath.Base(filepath.ToSlash(header.Name)) == libraryName && header.Typeflag == tar.TypeReg {
			return copyLibrary(destination, io.LimitReader(reader, header.Size))
		}
	}
	return fmt.Errorf("infergo: runtime library %q not found in archive", libraryName)
}

func copyLibrary(destination string, source io.Reader) error {
	file, err := os.OpenFile(destination, os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("infergo: open extracted runtime library: %w", err)
	}
	_, copyErr := io.Copy(file, source)
	closeErr := file.Close()
	if err := errors.Join(copyErr, closeErr); err != nil {
		return fmt.Errorf("infergo: extract runtime library: %w", err)
	}
	return nil
}

func replaceFile(source, destination string) error {
	if err := os.Rename(source, destination); err == nil {
		return nil
	}
	if err := os.Remove(destination); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return os.Rename(source, destination)
}

func writeMarker(path, digest string) error {
	file, err := os.CreateTemp(filepath.Dir(path), ".verified-*")
	if err != nil {
		return fmt.Errorf("infergo: create runtime verification marker: %w", err)
	}
	temporaryPath := file.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if _, err := io.WriteString(file, digest+"\n"); err != nil {
		_ = file.Close()
		return fmt.Errorf("infergo: write runtime verification marker: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("infergo: close runtime verification marker: %w", err)
	}
	if err := replaceFile(temporaryPath, path); err != nil {
		return fmt.Errorf("infergo: install runtime verification marker: %w", err)
	}
	return nil
}
