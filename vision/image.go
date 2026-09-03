package vision

import (
	"errors"
	"fmt"
	"image"
	"math"

	"golang.org/x/image/draw"
)

// ResizeMode controls how Process scales an image.
type ResizeMode int

// Supported image resize modes.
const (
	ResizeStretch ResizeMode = iota
	ResizeFit
	ResizeFill
	ResizeShortestEdge
)

// Options configures image resizing and channel normalization.
type Options struct {
	Width      int
	Height     int
	ShortEdge  int
	LongEdge   int
	Mode       ResizeMode
	Mean       [3]float32
	StdDev     [3]float32
	CenterCrop bool
}

// Image is normalized RGB image data in NCHW channel order.
type Image struct {
	Pixels       []float32
	Width        int
	Height       int
	OriginalSize image.Point
}

// Process resizes an image and converts it to a normalized RGB tensor in
// NCHW channel order.
func Process(source image.Image, options Options) (Image, error) {
	if source == nil {
		return Image{}, errors.New("vision: image cannot be nil")
	}
	if err := validateOptions(options); err != nil {
		return Image{}, err
	}
	originalSize := image.Pt(source.Bounds().Dx(), source.Bounds().Dy())
	if originalSize.X == 0 || originalSize.Y == 0 {
		return Image{}, errors.New("vision: image dimensions must be positive")
	}

	width, height := dimensions(originalSize.X, originalSize.Y, options)
	if width < 1 || height < 1 {
		return Image{}, errors.New("vision: resized image dimensions are invalid")
	}
	if height > math.MaxInt/4 || width > math.MaxInt/(height*4) {
		return Image{}, errors.New("vision: resized image dimensions overflow addressable memory")
	}
	resized := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.BiLinear.Scale(resized, resized.Bounds(), source, source.Bounds(), draw.Src, nil)

	processed := resized
	if options.CenterCrop {
		if width < options.Width || height < options.Height {
			return Image{}, fmt.Errorf("vision: cannot crop %dx%d image to %dx%d", width, height, options.Width, options.Height)
		}
		processed = centerCrop(resized, options.Width, options.Height)
	}

	return Image{
		Pixels:       normalizedNCHW(processed, options.Mean, options.StdDev),
		Width:        processed.Bounds().Dx(),
		Height:       processed.Bounds().Dy(),
		OriginalSize: originalSize,
	}, nil
}

func validateOptions(options Options) error {
	switch options.Mode {
	case ResizeStretch, ResizeFit, ResizeFill:
		if options.Width < 1 || options.Height < 1 {
			return errors.New("vision: width and height must be positive")
		}
	case ResizeShortestEdge:
		if options.ShortEdge < 1 || options.LongEdge < options.ShortEdge {
			return errors.New("vision: long edge must be at least the positive short edge")
		}
	default:
		return fmt.Errorf("vision: unsupported resize mode %d", options.Mode)
	}
	if options.CenterCrop && (options.Width < 1 || options.Height < 1) {
		return errors.New("vision: crop width and height must be positive")
	}
	for channel, mean := range options.Mean {
		if math.IsNaN(float64(mean)) || math.IsInf(float64(mean), 0) {
			return fmt.Errorf("vision: mean for channel %d must be finite", channel)
		}
	}
	for channel, deviation := range options.StdDev {
		if deviation == 0 || math.IsNaN(float64(deviation)) || math.IsInf(float64(deviation), 0) {
			return fmt.Errorf("vision: standard deviation for channel %d must be finite and non-zero", channel)
		}
	}
	return nil
}

func dimensions(width, height int, options Options) (int, int) {
	switch options.Mode {
	case ResizeStretch:
		return options.Width, options.Height
	case ResizeFit:
		scale := math.Min(float64(options.Width)/float64(width), float64(options.Height)/float64(height))
		return scaled(width, height, scale)
	case ResizeFill:
		scale := math.Max(float64(options.Width)/float64(width), float64(options.Height)/float64(height))
		return scaled(width, height, scale)
	case ResizeShortestEdge:
		shortest := min(width, height)
		longest := max(width, height)
		scale := float64(options.ShortEdge) / float64(shortest)
		if float64(longest)*scale > float64(options.LongEdge) {
			scale = float64(options.LongEdge) / float64(longest)
		}
		return scaled(width, height, scale)
	default:
		panic("vision: unreachable resize mode")
	}
}

func scaled(width, height int, scale float64) (int, int) {
	return max(1, int(math.Round(float64(width)*scale))), max(1, int(math.Round(float64(height)*scale)))
}

func centerCrop(source *image.RGBA, width, height int) *image.RGBA {
	x := (source.Bounds().Dx() - width) / 2
	y := (source.Bounds().Dy() - height) / 2
	destination := image.NewRGBA(image.Rect(0, 0, width, height))
	draw.Draw(destination, destination.Bounds(), source, image.Pt(x, y), draw.Src)
	return destination
}

func normalizedNCHW(source image.Image, mean, deviation [3]float32) []float32 {
	bounds := source.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	planeSize := width * height
	pixels := make([]float32, 3*planeSize)
	for y := range height {
		for x := range width {
			red, green, blue, _ := source.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			position := y*width + x
			pixels[position] = normalize(red, mean[0], deviation[0])
			pixels[planeSize+position] = normalize(green, mean[1], deviation[1])
			pixels[2*planeSize+position] = normalize(blue, mean[2], deviation[2])
		}
	}
	return pixels
}

func normalize(value uint32, mean, deviation float32) float32 {
	return (float32(value)/65535 - mean) / deviation
}
