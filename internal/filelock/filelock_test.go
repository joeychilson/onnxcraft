package filelock

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
)

func TestAcquireHonorsCancellation(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "cache.lock")
	first, err := Acquire(t.Context(), path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := first.Close(); err != nil {
			t.Errorf("Close(): %v", err)
		}
	}()
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := Acquire(ctx, path); !errors.Is(err, context.Canceled) {
		t.Fatalf("Acquire() error = %v, want context.Canceled", err)
	}
}
