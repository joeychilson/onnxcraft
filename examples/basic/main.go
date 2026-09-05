package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/joeychilson/onnxcraft"
)

func main() {
	library := flag.String("runtime", os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"), "ONNX Runtime shared library path")
	flag.Parse()

	rt, err := onnxcraft.Open(*library)
	if err != nil {
		log.Fatal(err)
	}
	defer rt.Close()

	session, err := rt.Load("testdata/add.onnx", nil)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	a, err := onnxcraft.NewTensor([]int64{1, 3}, []float32{1, 2, 3})
	if err != nil {
		log.Fatal(err)
	}
	b, err := onnxcraft.NewTensor([]int64{1, 3}, []float32{10, 20, 30})
	if err != nil {
		log.Fatal(err)
	}

	outputs, err := session.Run(context.Background(), a, b)
	if err != nil {
		log.Fatal(err)
	}

	values, err := outputs[0].Data[float32]()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(values)
}
