//go:build windows

package atomicfile

func syncDirectory(string) error {
	return nil
}
