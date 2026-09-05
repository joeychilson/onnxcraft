package onnxcraft

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestNativeFailureCleanup(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("fixture compiler invocation is for Unix")
	}
	compiler := os.Getenv("CC")
	if compiler == "" {
		compiler = "cc"
	}
	dir := t.TempDir()
	library := filepath.Join(dir, "failure.so")
	flags := []string{"-shared", "-fPIC", "-I.", "testdata/failing_runtime.c", "-o", library}
	if runtime.GOOS == "darwin" {
		flags[0] = "-dynamiclib"
	}
	if out, err := exec.Command(compiler, flags...).CombinedOutput(); err != nil {
		t.Fatalf("compile fixture: %v\n%s", err, out)
	}
	for _, stage := range []string{"env", "memory", "options", "session", "metadata", "tensor", "run", "output", "none"} {
		t.Run(stage, func(t *testing.T) {
			log := filepath.Join(t.TempDir(), "allocations")
			t.Setenv("OC_TEST_LOG", log)
			t.Setenv("OC_TEST_FAIL", stage)
			r, err := Open(library)
			if err == nil {
				s, loadErr := r.Load("unused.onnx", nil)
				err = loadErr
				if err == nil {
					x := tensor(t, []int64{1}, []float32{42})
					if stage == "tensor" {
						out := []Tensor{tensor(t, []int64{1}, make([]float32, 1)), tensor(t, []int64{1}, make([]float32, 1))}
						err = s.RunInto(context.Background(), out, x)
					} else {
						_, err = s.Run(context.Background(), x)
					}
					s.Close()
				}
				r.Close()
			}
			if stage == "none" {
				if err != nil {
					t.Fatal(err)
				}
			} else {
				var native *NativeError
				if !errors.As(err, &native) || native.Code != 1 || native.Message != "injected failure" {
					t.Fatalf("error = %v", err)
				}
			}
			contents, readErr := os.ReadFile(log)
			if readErr != nil {
				t.Fatal(readErr)
			}
			balance := map[string]int{}
			for entry := range strings.FieldsSeq(string(contents)) {
				name := entry[:len(entry)-1]
				if strings.HasSuffix(entry, "+") {
					balance[name]++
				} else {
					balance[name]--
				}
				if balance[name] < 0 {
					t.Fatalf("double release: %s\n%s", name, contents)
				}
			}
			for name, n := range balance {
				if n != 0 {
					t.Errorf("leaked %s: %d\n%s", name, n, contents)
				}
			}
		})
	}
}
