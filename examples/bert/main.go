// Command bert predicts replacements for masked tokens with a BERT model.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"

	"github.com/joeychilson/infergo"
	"github.com/joeychilson/infergo/bert"
	"github.com/joeychilson/infergo/modelhub"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() (err error) {
	modelPath := flag.String("model", "", "path to a masked-language ONNX model; downloads a verified BERT model when empty")
	text := flag.String("text", "The [MASK] is a large animal that lives in the [MASK].", "text containing one or more [MASK] tokens")
	topK := flag.Int("top", 5, "predictions per mask")
	flag.Parse()

	ctx := context.Background()
	if *modelPath == "" {
		hub, hubErr := modelhub.New()
		if hubErr != nil {
			return hubErr
		}
		*modelPath, hubErr = hub.Fetch(ctx, modelhub.BERTBaseUncased())
		if hubErr != nil {
			return hubErr
		}
	}
	runtime, err := infergo.Open(ctx)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, runtime.Close()) }()

	model, err := bert.New(runtime, *modelPath)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, model.Close()) }()

	results, err := model.FillMask(ctx, *text, bert.FillMaskOptions{TopK: *topK})
	if err != nil {
		return err
	}
	for _, result := range results {
		fmt.Printf("mask at token %d:\n", result.Position)
		for _, prediction := range result.Predictions {
			fmt.Printf("  %-16s %6.2f%%\n", prediction.Label, prediction.Score*100)
		}
	}
	return nil
}
