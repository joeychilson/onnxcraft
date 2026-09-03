// Package atomicfile provides atomic file installation helpers shared by the
// runtime and model download paths.
package atomicfile

import (
	"errors"
	"os"
	"path/filepath"
)

// Replace atomically installs source at target via rename, removing target
// first when the platform does not replace on rename (Windows).
func Replace(source, target string) error {
	if err := os.Rename(source, target); err == nil {
		return nil
	}
	if err := os.Remove(target); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	return os.Rename(source, target)
}

// Write installs data at path atomically with perm permissions.
func Write(path string, data []byte, perm os.FileMode) error {
	file, err := os.CreateTemp(filepath.Dir(path), ".atomic-*")
	if err != nil {
		return err
	}
	temporaryPath := file.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if _, err := file.Write(data); err != nil {
		_ = file.Close()
		return err
	}
	if err := file.Chmod(perm); err != nil {
		_ = file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	return Replace(temporaryPath, path)
}
