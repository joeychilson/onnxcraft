//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package atomicfile

import "os"

func syncDirectory(path string) error {
	directory, err := os.Open(path)
	if err != nil {
		return err
	}
	syncErr := directory.Sync()
	closeErr := directory.Close()
	if syncErr != nil {
		return syncErr
	}
	return closeErr
}
