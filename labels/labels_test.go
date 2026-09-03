package labels

import "testing"

func TestStandardLabels(t *testing.T) {
	t.Parallel()
	if label, ok := COCO(1); !ok || label != "person" {
		t.Fatalf("COCO(1) = %q, %v", label, ok)
	}
	if label, ok := ImageNet(0); !ok || label != "tench, Tinca tinca" {
		t.Fatalf("ImageNet(0) = %q, %v", label, ok)
	}
	if len(COCOMap()) != 80 {
		t.Fatalf("len(COCOMap()) = %d", len(COCOMap()))
	}
	if len(ImageNetMap()) != 1000 {
		t.Fatalf("len(ImageNetMap()) = %d", len(ImageNetMap()))
	}
}

func TestLabelMapsAreCopies(t *testing.T) {
	t.Parallel()
	labels := COCOMap()
	labels[1] = "changed"
	if label, _ := COCO(1); label != "person" {
		t.Fatalf("COCO map was mutated: %q", label)
	}
}
