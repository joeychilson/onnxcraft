package infergo_test

import (
	"errors"
	"image"
	_ "image/jpeg"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/joeychilson/infergo"
	"github.com/joeychilson/infergo/bert"
	"github.com/joeychilson/infergo/embedding"
	"github.com/joeychilson/infergo/modelhub"
	"github.com/joeychilson/infergo/resnet"
	"github.com/joeychilson/infergo/yolos"
)

func TestCatalogModels(t *testing.T) {
	if os.Getenv("INFERGO_MODEL_INTEGRATION") == "" {
		t.Skip("set INFERGO_MODEL_INTEGRATION=1 to run catalog model tests")
	}
	cacheRoot := os.Getenv("INFERGO_CACHE_DIR")
	if cacheRoot == "" {
		cacheRoot = t.TempDir()
	}
	hub, err := modelhub.New(modelhub.WithCacheDir(filepath.Join(cacheRoot, "models")))
	if err != nil {
		t.Fatal(err)
	}
	runtime, err := infergo.Open(t.Context(), infergo.WithCacheDir(cacheRoot))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if closeErr := runtime.Close(); closeErr != nil {
			t.Error(closeErr)
		}
	})

	t.Run("bert", func(t *testing.T) {
		path, fetchErr := hub.Fetch(t.Context(), modelhub.BERTBaseUncased())
		if fetchErr != nil {
			t.Fatal(fetchErr)
		}
		model, modelErr := bert.New(runtime, path)
		if modelErr != nil {
			t.Fatal(modelErr)
		}
		defer func() {
			if closeErr := model.Close(); closeErr != nil {
				t.Error(closeErr)
			}
		}()
		predictions, predictErr := model.FillMask(
			t.Context(),
			"The capital of France is [MASK].",
			bert.FillMaskOptions{TopK: 3},
		)
		if predictErr != nil {
			t.Fatal(predictErr)
		}
		if len(predictions) != 1 || len(predictions[0].Predictions) != 3 || predictions[0].Predictions[0].Label != "paris" {
			t.Fatalf("FillMask() = %+v", predictions)
		}
	})

	t.Run("embedding", func(t *testing.T) {
		path, fetchErr := hub.Fetch(t.Context(), modelhub.AllMiniLML6V2())
		if fetchErr != nil {
			t.Fatal(fetchErr)
		}
		model, modelErr := embedding.New(runtime, path)
		if modelErr != nil {
			t.Fatal(modelErr)
		}
		defer func() {
			if closeErr := model.Close(); closeErr != nil {
				t.Error(closeErr)
			}
		}()
		vectors, embedErr := model.Embed(t.Context(), []string{
			"A dog is playing in the park.",
			"A puppy plays outside.",
		}, embedding.EmbedOptions{})
		if embedErr != nil {
			t.Fatal(embedErr)
		}
		if len(vectors) != 2 || len(vectors[0]) != 384 || len(vectors[1]) != 384 {
			t.Fatalf("embedding dimensions = %d x %d", len(vectors), len(vectors[0]))
		}
		similarity, similarityErr := embedding.CosineSimilarity(vectors[0], vectors[1])
		if similarityErr != nil {
			t.Fatal(similarityErr)
		}
		if similarity < 0.5 || similarity > 0.6 {
			t.Fatalf("CosineSimilarity() = %f, want [0.5, 0.6]", similarity)
		}
		for index, vector := range vectors {
			if norm := vectorNorm(vector); math.Abs(norm-1) > 1e-5 {
				t.Fatalf("embedding %d norm = %f, want 1", index, norm)
			}
		}
	})

	t.Run("resnet", func(t *testing.T) {
		path, fetchErr := hub.Fetch(t.Context(), modelhub.ResNet50())
		if fetchErr != nil {
			t.Fatal(fetchErr)
		}
		model, modelErr := resnet.New(runtime, path)
		if modelErr != nil {
			t.Fatal(modelErr)
		}
		defer func() {
			if closeErr := model.Close(); closeErr != nil {
				t.Error(closeErr)
			}
		}()
		source := openImage(t, "examples/data/airport.jpg")
		predictions, predictErr := model.Classify(t.Context(), source, resnet.ClassifyOptions{TopK: 3})
		if predictErr != nil {
			t.Fatal(predictErr)
		}
		if len(predictions) != 3 || predictions[0].Label != "trolleybus, trolley coach, trackless trolley" ||
			predictions[0].Score < 0.7 {
			t.Fatalf("Classify() = %+v", predictions)
		}
	})

	t.Run("yolos", func(t *testing.T) {
		path, fetchErr := hub.Fetch(t.Context(), modelhub.YOLOSSmall())
		if fetchErr != nil {
			t.Fatal(fetchErr)
		}
		model, modelErr := yolos.New(runtime, path)
		if modelErr != nil {
			t.Fatal(modelErr)
		}
		defer func() {
			if closeErr := model.Close(); closeErr != nil {
				t.Error(closeErr)
			}
		}()
		source := openImage(t, "examples/data/football-match.jpg")
		detections, detectErr := model.Detect(t.Context(), source, yolos.DetectOptions{MinScore: 0.5})
		if detectErr != nil {
			t.Fatal(detectErr)
		}
		if len(detections) < 5 || detections[0].Label != "person" || detections[0].Score < 0.99 {
			t.Fatalf("Detect() = %+v", detections)
		}
		foundBall := false
		for _, detection := range detections {
			if detection.Label == "sports ball" && detection.Score > 0.99 {
				foundBall = true
				break
			}
		}
		if !foundBall {
			t.Fatalf("Detect() did not find the sports ball: %+v", detections)
		}
	})
}

func openImage(t *testing.T, path string) image.Image {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	source, _, decodeErr := image.Decode(file)
	if err := errors.Join(decodeErr, file.Close()); err != nil {
		t.Fatal(err)
	}
	return source
}

func vectorNorm(vector []float32) float64 {
	var squared float64
	for _, value := range vector {
		squared += float64(value) * float64(value)
	}
	return math.Sqrt(squared)
}
