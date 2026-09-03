package vision

import (
	"context"
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

// Interpolation controls the resampling filter used while resizing.
type Interpolation int

// Supported resize interpolation filters.
const (
	InterpolationBilinear Interpolation = iota
	InterpolationBicubic
	InterpolationNearest
)

// Options configures image resizing and channel normalization.
type Options struct {
	Width         int
	Height        int
	ShortEdge     int
	LongEdge      int
	Mode          ResizeMode
	Interpolation Interpolation
	Mean          [3]float32
	StdDev        [3]float32
	CenterCrop    bool
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
	return ProcessContext(context.Background(), source, options)
}

// ProcessContext is like Process and stops preprocessing when ctx is
// canceled.
func ProcessContext(ctx context.Context, source image.Image, options Options) (Image, error) {
	return ProcessIntoContext(ctx, source, options, nil)
}

// ProcessInto is like Process but reuses destination when it has enough
// capacity. The returned Image reports the slice that was actually used.
func ProcessInto(source image.Image, options Options, destination []float32) (Image, error) {
	return ProcessIntoContext(context.Background(), source, options, destination)
}

// ProcessIntoContext is like ProcessInto and stops preprocessing when ctx is
// canceled.
func ProcessIntoContext(
	ctx context.Context,
	source image.Image,
	options Options,
	destination []float32,
) (Image, error) {
	if ctx == nil {
		return Image{}, errors.New("vision: context cannot be nil")
	}
	if err := ctx.Err(); err != nil {
		return Image{}, err
	}
	if source == nil {
		return Image{}, errors.New("vision: image cannot be nil")
	}
	if err := validateOptions(options); err != nil {
		return Image{}, err
	}
	originalSize := image.Pt(source.Bounds().Dx(), source.Bounds().Dy())
	if originalSize.X < 1 || originalSize.Y < 1 {
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
	scaler(options.Interpolation).Scale(resized, resized.Bounds(), source, source.Bounds(), draw.Src, nil)
	if err := ctx.Err(); err != nil {
		return Image{}, err
	}

	processed := resized
	if options.CenterCrop {
		if width < options.Width || height < options.Height {
			return Image{}, fmt.Errorf("vision: cannot crop %dx%d image to %dx%d", width, height, options.Width, options.Height)
		}
		processed = centerCrop(resized, options.Width, options.Height)
	}

	pixels, err := normalizedNCHW(ctx, processed, options.Mean, options.StdDev, destination)
	if err != nil {
		return Image{}, err
	}
	return Image{
		Pixels:       pixels,
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
	if options.Interpolation < InterpolationBilinear || options.Interpolation > InterpolationNearest {
		return fmt.Errorf("vision: unsupported interpolation %d", options.Interpolation)
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

func scaler(interpolation Interpolation) draw.Scaler {
	switch interpolation {
	case InterpolationBilinear:
		return draw.BiLinear
	case InterpolationBicubic:
		return draw.CatmullRom
	case InterpolationNearest:
		return draw.NearestNeighbor
	default:
		panic("vision: unreachable interpolation")
	}
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
	minimum := source.Bounds().Min.Add(image.Pt(x, y))
	return source.SubImage(image.Rectangle{Min: minimum, Max: minimum.Add(image.Pt(width, height))}).(*image.RGBA)
}

func normalizedNCHW(
	ctx context.Context,
	source *image.RGBA,
	mean, deviation [3]float32,
	destination []float32,
) ([]float32, error) {
	bounds := source.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	planeSize := width * height
	required := 3 * planeSize
	var pixels []float32
	if cap(destination) >= required {
		pixels = destination[:required]
	} else {
		pixels = make([]float32, required)
	}
	for y := range height {
		if y&63 == 0 {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
		}
		for x := range width {
			offset := source.PixOffset(bounds.Min.X+x, bounds.Min.Y+y)
			position := y*width + x
			pixels[position] = normalize(source.Pix[offset], mean[0], deviation[0])
			pixels[planeSize+position] = normalize(source.Pix[offset+1], mean[1], deviation[1])
			pixels[2*planeSize+position] = normalize(source.Pix[offset+2], mean[2], deviation[2])
		}
	}
	return pixels, nil
}

func normalize(value uint8, mean, deviation float32) float32 {
	return (float32(value)/255 - mean) / deviation
}
