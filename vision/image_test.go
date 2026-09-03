package vision

import (
	"context"
	"errors"
	"image"
	"image/color"
	"math"
	"slices"
	"testing"
)

func TestProcessContextCancellation(t *testing.T) {
	t.Parallel()
	source := image.NewRGBA(image.Rect(0, 0, 1, 1))
	options := Options{
		Width:  1,
		Height: 1,
		Mode:   ResizeStretch,
		StdDev: [3]float32{1, 1, 1},
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := ProcessContext(ctx, source, options); !errors.Is(err, context.Canceled) {
		t.Fatalf("ProcessContext() error = %v, want context.Canceled", err)
	}
}

func TestProcessProducesNCHW(t *testing.T) {
	t.Parallel()
	source := image.NewRGBA(image.Rect(10, 20, 12, 21))
	source.Set(10, 20, color.RGBA{R: 255, A: 255})
	source.Set(11, 20, color.RGBA{G: 255, A: 255})
	got, err := Process(source, Options{
		Width:  2,
		Height: 1,
		Mode:   ResizeStretch,
		StdDev: [3]float32{1, 1, 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{1, 0, 0, 1, 0, 0}
	if !slices.Equal(got.Pixels, want) {
		t.Fatalf("pixels = %v, want %v", got.Pixels, want)
	}
	if got.OriginalSize != image.Pt(2, 1) || got.Width != 2 || got.Height != 1 {
		t.Fatalf("image metadata = %+v", got)
	}
}

func TestDimensions(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name    string
		width   int
		height  int
		options Options
		want    image.Point
	}{
		{name: "stretch", width: 400, height: 200, options: Options{Mode: ResizeStretch, Width: 100, Height: 100}, want: image.Pt(100, 100)},
		{name: "fit", width: 400, height: 200, options: Options{Mode: ResizeFit, Width: 100, Height: 100}, want: image.Pt(100, 50)},
		{name: "fill", width: 400, height: 200, options: Options{Mode: ResizeFill, Width: 100, Height: 100}, want: image.Pt(200, 100)},
		{name: "short edge", width: 400, height: 200, options: Options{Mode: ResizeShortestEdge, ShortEdge: 100, LongEdge: 150}, want: image.Pt(150, 75)},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			width, height := dimensions(test.width, test.height, test.options)
			if got := image.Pt(width, height); got != test.want {
				t.Fatalf("dimensions() = %v, want %v", got, test.want)
			}
		})
	}
}

func TestProcessCenterCrops(t *testing.T) {
	t.Parallel()
	source := image.NewRGBA(image.Rect(0, 0, 4, 2))
	got, err := Process(source, Options{
		Width:      2,
		Height:     2,
		Mode:       ResizeFill,
		StdDev:     [3]float32{1, 1, 1},
		CenterCrop: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.Width != 2 || got.Height != 2 || len(got.Pixels) != 12 {
		t.Fatalf("Process() = %+v", got)
	}
}

func TestProcessRejectsInvalidOptions(t *testing.T) {
	t.Parallel()
	source := image.NewRGBA(image.Rect(0, 0, 1, 1))
	tests := []Options{
		{},
		{Mode: ResizeShortestEdge, ShortEdge: 2, LongEdge: 1, StdDev: [3]float32{1, 1, 1}},
		{Mode: ResizeMode(99), StdDev: [3]float32{1, 1, 1}},
		{Mode: ResizeStretch, Width: 1, Height: 1, StdDev: [3]float32{1, float32(math.NaN()), 1}},
	}
	for _, options := range tests {
		if _, err := Process(source, options); err == nil {
			t.Fatalf("Process(%+v) error = nil", options)
		}
	}
}

func TestProcessSupportsInterpolationFilters(t *testing.T) {
	t.Parallel()
	source := image.NewRGBA(image.Rect(0, 0, 2, 2))
	for _, interpolation := range []Interpolation{InterpolationBilinear, InterpolationBicubic, InterpolationNearest} {
		_, err := Process(source, Options{
			Width:         3,
			Height:        3,
			Mode:          ResizeStretch,
			Interpolation: interpolation,
			StdDev:        [3]float32{1, 1, 1},
		})
		if err != nil {
			t.Fatalf("Process(interpolation %d): %v", interpolation, err)
		}
	}
}

func TestProcessIntoReusesDestination(t *testing.T) {
	t.Parallel()
	source := image.NewRGBA(image.Rect(0, 0, 2, 2))
	destination := make([]float32, 12)
	got, err := ProcessInto(source, Options{
		Width: 2, Height: 2, Mode: ResizeStretch, StdDev: [3]float32{1, 1, 1},
	}, destination)
	if err != nil {
		t.Fatal(err)
	}
	if &got.Pixels[0] != &destination[0] {
		t.Fatal("ProcessInto() did not reuse destination")
	}
}

func FuzzResizeDimensions(f *testing.F) {
	f.Add(uint16(640), uint16(480), uint16(224), uint16(224))
	f.Fuzz(func(t *testing.T, sourceWidth, sourceHeight, targetWidth, targetHeight uint16) {
		width := 1 + int(sourceWidth)%2048
		height := 1 + int(sourceHeight)%2048
		options := Options{
			Width: 1 + int(targetWidth)%512, Height: 1 + int(targetHeight)%512,
			Mode: ResizeFill, StdDev: [3]float32{1, 1, 1},
		}
		resizedWidth, resizedHeight := dimensions(width, height, options)
		if resizedWidth < options.Width || resizedHeight < options.Height {
			t.Fatalf("dimensions() = %dx%d, target = %dx%d", resizedWidth, resizedHeight, options.Width, options.Height)
		}
	})
}

func BenchmarkProcessImage(b *testing.B) {
	source := image.NewRGBA(image.Rect(0, 0, 1920, 1080))
	options := Options{
		Width: 224, Height: 224, ShortEdge: 256, LongEdge: math.MaxInt,
		Mode: ResizeShortestEdge, Interpolation: InterpolationBicubic,
		StdDev: [3]float32{1, 1, 1}, CenterCrop: true,
	}
	destination := make([]float32, 3*224*224)
	for b.Loop() {
		if _, err := ProcessInto(source, options, destination); err != nil {
			b.Fatal(err)
		}
	}
}
