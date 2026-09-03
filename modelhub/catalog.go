package modelhub

// BERTBaseUncased returns a pinned, dynamically quantized masked-language
// model from Xenova/bert-base-uncased for use with bert.New.
func BERTBaseUncased() Artifact {
	return Artifact{
		Name:   "bert-base-uncased.onnx",
		URL:    "https://huggingface.co/Xenova/bert-base-uncased/resolve/ab680a327acc2d9c3bd279ffb1cd43454181f743/onnx/model_quantized.onnx",
		SHA256: "f2c1ce78f723dfa260bf743c56596ddd5097e45081cdffa24820aab900855ed0",
		Size:   110_848_579,
	}
}

// AllMiniLML6V2 returns a pinned full-precision sentence-embedding model from
// sentence-transformers/all-MiniLM-L6-v2. Use it with embedding.New and a
// maximum input length of 256.
func AllMiniLML6V2() Artifact {
	return Artifact{
		Name:   "all-MiniLM-L6-v2.onnx",
		URL:    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/onnx/model.onnx",
		SHA256: "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
		Size:   90_405_214,
	}
}

// ResNet50 returns a pinned full-precision ImageNet classifier from
// onnx-community/resnet-50-ONNX for use with resnet.New.
func ResNet50() Artifact {
	return Artifact{
		Name:   "resnet-50.onnx",
		URL:    "https://huggingface.co/onnx-community/resnet-50-ONNX/resolve/b6ef686fe842388c9449c7acb7269d02c69bacfb/onnx/model.onnx",
		SHA256: "de5d34d98e29c8f436264685f0b0e6a83b0d023516e40df0f96e02d5dd8e4c50",
		Size:   102_182_062,
	}
}

// YOLOSSmall returns a pinned full-precision object detector from
// Xenova/yolos-small for use with yolos.New.
func YOLOSSmall() Artifact {
	return Artifact{
		Name:   "yolos-small.onnx",
		URL:    "https://huggingface.co/Xenova/yolos-small/resolve/45d2b28b7d2f3ddc80b2543c1520c612602bb9ef/onnx/model.onnx",
		SHA256: "08af01badf2adf787dc473d290b5597d12989be4f0bedeae1afa089e2b0a13b5",
		Size:   123_020_648,
	}
}
