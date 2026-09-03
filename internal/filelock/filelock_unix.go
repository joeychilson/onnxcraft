//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

// Package filelock serializes cache installation across processes.
package filelock

import (
	"context"
	"errors"
	"fmt"
	"os"
	"time"

	"golang.org/x/sys/unix"
)

// Lock is an advisory exclusive file lock.
type Lock struct {
	file *os.File
}

// Acquire waits until path can be locked or ctx is cancelled.
func Acquire(ctx context.Context, path string) (*Lock, error) {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, fmt.Errorf("open lock file: %w", err)
	}
	for {
		err = unix.Flock(int(file.Fd()), unix.LOCK_EX|unix.LOCK_NB)
		if err == nil {
			return &Lock{file: file}, nil
		}
		if !errors.Is(err, unix.EWOULDBLOCK) {
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
	unlockErr := unix.Flock(int(l.file.Fd()), unix.LOCK_UN)
	closeErr := l.file.Close()
	l.file = nil
	return errors.Join(unlockErr, closeErr)
}
