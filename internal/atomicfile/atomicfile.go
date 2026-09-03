// Package atomicfile provides atomic file installation helpers shared by the
// runtime and model download paths.
package atomicfile

import (
	"os"
	"path/filepath"
)

// Replace atomically installs source at target via the platform rename
// operation. Source and target must be on the same filesystem.
func Replace(source, target string) error {
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
	if err := file.Sync(); err != nil {
		_ = file.Close()
		return err
	}
	if err := file.Close(); err != nil {
		return err
	}
	if err := Replace(temporaryPath, path); err != nil {
		return err
	}
	return syncDirectory(filepath.Dir(path))
}
