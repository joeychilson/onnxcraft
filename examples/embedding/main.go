// Command embedding compares the meanings of two sentences with MiniLM.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"

	"github.com/joeychilson/infergo"
	"github.com/joeychilson/infergo/embedding"
	"github.com/joeychilson/infergo/modelhub"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() (err error) {
	left := flag.String("left", "A dog is playing in the park.", "first sentence")
	right := flag.String("right", "A puppy plays outside.", "second sentence")
	flag.Parse()
	ctx := context.Background()

	hub, err := modelhub.New()
	if err != nil {
		return err
	}
	modelPath, err := hub.Fetch(ctx, modelhub.AllMiniLML6V2())
	if err != nil {
		return err
	}
	runtime, err := infergo.Open(ctx)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, runtime.Close()) }()
	model, err := embedding.New(runtime, modelPath)
	if err != nil {
		return err
	}
	defer func() { err = errors.Join(err, model.Close()) }()

	vectors, err := model.Embed(ctx, []string{*left, *right}, embedding.EmbedOptions{MaxLength: 256})
	if err != nil {
		return err
	}
	similarity, err := embedding.CosineSimilarity(vectors[0], vectors[1])
	if err != nil {
		return err
	}
	fmt.Printf("cosine similarity: %.4f\n", similarity)
	return nil
}
