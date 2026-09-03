// Command yolos detects and annotates objects with a YOLOS model.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	"log"
	"os"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	"github.com/joeychilson/infergo"
	"github.com/joeychilson/infergo/modelhub"
	"github.com/joeychilson/infergo/postprocess"
	"github.com/joeychilson/infergo/yolos"
)

var palette = [...]color.RGBA{
	{R: 235, G: 87, B: 87, A: 255},
	{R: 39, G: 174, B: 96, A: 255},
	{R: 47, G: 128, B: 237, A: 255},
	{R: 242, G: 201, B: 76, A: 255},
	{R: 155, G: 81, B: 224, A: 255},
}

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() (err error) {
	modelPath := flag.String("model", "", "path to a YOLOS ONNX model; downloads a verified YOLOS-small when empty")
	imagePath := flag.String("image", "", "path to an input image")
	outputPath := flag.String("output", "detections.png", "path for the annotated PNG")
	minScore := flag.Float64("min-score", 0.5, "minimum confidence from 0 to 1")
	flag.Parse()
	if *imagePath == "" {
		return errors.New("the -image flag is required")
	}

	source, err := loadImage(*imagePath)
	if err != nil {
		return err
	}
	ctx := context.Background()
	if *modelPath == "" {
		hub, hubErr := modelhub.New()
		if hubErr != nil {
			return hubErr
		}
		*modelPath, hubErr = hub.Fetch(ctx, modelhub.YOLOSSmall())
		if hubErr != nil {
			return hubErr
		}
	}
	runtime, err := infergo.Open(ctx)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, runtime.Close()) }()
	model, err := yolos.New(runtime, *modelPath)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, model.Close()) }()

	detections, err := model.Detect(ctx, source, yolos.DetectOptions{MinScore: float32(*minScore)})
	if err != nil {
		return err
	}
	annotated := annotate(source, detections)
	if err := writePNG(*outputPath, annotated); err != nil {
		return err
	}
	for _, detection := range detections {
		fmt.Printf("%-20s %6.2f%%  [%.0f, %.0f, %.0f, %.0f]\n",
			detection.Label,
			detection.Score*100,
			detection.Box.X1,
			detection.Box.Y1,
			detection.Box.X2,
			detection.Box.Y2,
		)
	}
	fmt.Printf("wrote %d detections to %s\n", len(detections), *outputPath)
	return nil
}

func loadImage(path string) (image.Image, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open image: %w", err)
	}
	source, _, decodeErr := image.Decode(file)
	if err := errors.Join(decodeErr, file.Close()); err != nil {
		return nil, fmt.Errorf("decode image: %w", err)
	}
	return source, nil
}

func writePNG(path string, source image.Image) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create output: %w", err)
	}
	if err := errors.Join(png.Encode(file, source), file.Close()); err != nil {
		return fmt.Errorf("write output: %w", err)
	}
	return nil
}

func annotate(source image.Image, detections []postprocess.Detection) *image.RGBA {
	bounds := source.Bounds()
	result := image.NewRGBA(image.Rect(0, 0, bounds.Dx(), bounds.Dy()))
	draw.Draw(result, result.Bounds(), source, bounds.Min, draw.Src)
	for index, detection := range detections {
		boxColor := palette[index%len(palette)]
		drawBox(result, detection.Box, boxColor)
		label := fmt.Sprintf("%s %.0f%%", detection.Label, detection.Score*100)
		fontDrawer := font.Drawer{
			Dst:  result,
			Src:  image.NewUniform(boxColor),
			Face: basicfont.Face7x13,
			Dot: fixed.Point26_6{
				X: fixed.I(max(0, int(detection.Box.X1))),
				Y: fixed.I(max(13, int(detection.Box.Y1)-3)),
			},
		}
		fontDrawer.DrawString(label)
	}
	return result
}

func drawBox(destination *image.RGBA, box postprocess.Box, boxColor color.RGBA) {
	bounds := destination.Bounds()
	x1 := max(bounds.Min.X, min(bounds.Max.X-1, int(box.X1)))
	y1 := max(bounds.Min.Y, min(bounds.Max.Y-1, int(box.Y1)))
	x2 := max(bounds.Min.X, min(bounds.Max.X-1, int(box.X2)))
	y2 := max(bounds.Min.Y, min(bounds.Max.Y-1, int(box.Y2)))
	for thickness := range 2 {
		for x := x1; x <= x2; x++ {
			destination.Set(x, min(y1+thickness, y2), boxColor)
			destination.Set(x, max(y2-thickness, y1), boxColor)
		}
		for y := y1; y <= y2; y++ {
			destination.Set(min(x1+thickness, x2), y, boxColor)
			destination.Set(max(x2-thickness, x1), y, boxColor)
		}
	}
}
