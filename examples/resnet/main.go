// Command resnet classifies an image with a ResNet model.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"
	"os"

	"github.com/joeychilson/infergo"
	"github.com/joeychilson/infergo/modelhub"
	"github.com/joeychilson/infergo/resnet"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() (err error) {
	modelPath := flag.String("model", "", "path to a ResNet ONNX model; downloads a verified ResNet-50 when empty")
	imagePath := flag.String("image", "", "path to an input image")
	topK := flag.Int("top", 5, "number of predictions")
	minScore := flag.Float64("min-score", 0.1, "minimum confidence from 0 to 1")
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
		*modelPath, hubErr = hub.Fetch(ctx, modelhub.ResNet50())
		if hubErr != nil {
			return hubErr
		}
	}
	runtime, err := infergo.Open(ctx)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, runtime.Close()) }()
	model, err := resnet.New(runtime, *modelPath)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, model.Close()) }()

	predictions, err := model.Classify(ctx, source, resnet.ClassifyOptions{
		TopK:     *topK,
		MinScore: float32(*minScore),
	})
	if err != nil {
		return err
	}
	for index, prediction := range predictions {
		fmt.Printf("%d. %-40s %6.2f%%\n", index+1, prediction.Label, prediction.Score*100)
	}
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
