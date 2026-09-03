//go:build windows

// Package filelock serializes cache installation across processes.
package filelock

import (
	"context"
	"errors"
	"fmt"
	"os"
	"time"

	"golang.org/x/sys/windows"
)

// Lock is an advisory exclusive file lock.
type Lock struct {
	file       *os.File
	overlapped windows.Overlapped
}

// Acquire waits until path can be locked or ctx is cancelled.
func Acquire(ctx context.Context, path string) (*Lock, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open lock file: %w", err)
	}
	lock := &Lock{file: file}
	for {
		err = windows.LockFileEx(
			windows.Handle(file.Fd()),
			windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY,
			0, 1, 0, &lock.overlapped,
		)
		if err == nil {
			return lock, nil
		}
		if !errors.Is(err, windows.ERROR_LOCK_VIOLATION) {
			_ = file.Close()
			return nil, fmt.Errorf("lock file: %w", err)
		}
		timer := time.NewTimer(25 * time.Millisecond)
		select {
		case <-ctx.Done():
			timer.Stop()
			_ = file.Close()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
}

// Close releases the lock.
func (l *Lock) Close() error {
	if l == nil || l.file == nil {
		return nil
	}
	unlockErr := windows.UnlockFileEx(windows.Handle(l.file.Fd()), 0, 1, 0, &l.overlapped)
	closeErr := l.file.Close()
	l.file = nil
	return errors.Join(unlockErr, closeErr)
}
