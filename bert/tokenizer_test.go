package bert

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
)

func TestTokenizerContextCancellation(t *testing.T) {
	t.Parallel()
	tokenizer, err := NewTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(t.Context())
	cancel()
	if _, err := tokenizer.EncodeContext(ctx, "hello", 8); !errors.Is(err, context.Canceled) {
		t.Fatalf("EncodeContext() error = %v, want context.Canceled", err)
	}
	if _, err := tokenizer.EncodeBatchContext(ctx, []string{"hello"}, 8); !errors.Is(err, context.Canceled) {
		t.Fatalf("EncodeBatchContext() error = %v, want context.Canceled", err)
	}
}

func TestTokenizerEncodesUnicodeAndSpecialTokens(t *testing.T) {
	t.Parallel()
	tokenizer, err := NewTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	encoding, err := tokenizer.Encode("Héllo, [mask]!", 16)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"[CLS]", "hello", ",", "[MASK]", "!", "[SEP]"}
	if !slices.Equal(encoding.Tokens, want) {
		t.Fatalf("tokens = %q, want %q", encoding.Tokens, want)
	}
	if !slices.Equal(tokenizer.MaskPositions(encoding), []int{3}) {
		t.Fatalf("mask positions = %v", tokenizer.MaskPositions(encoding))
	}
	if len(encoding.IDs) != len(want) || len(encoding.AttentionMask) != len(want) {
		t.Fatalf("encoding lengths = %d, %d", len(encoding.IDs), len(encoding.AttentionMask))
	}
}

func TestTokenizerTruncatesBeforeSeparator(t *testing.T) {
	t.Parallel()
	tokenizer, err := NewTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	encoding, err := tokenizer.Encode("one two three four", 4)
	if err != nil {
		t.Fatal(err)
	}
	want := []string{"[CLS]", "one", "two", "[SEP]"}
	if !slices.Equal(encoding.Tokens, want) {
		t.Fatalf("tokens = %q, want %q", encoding.Tokens, want)
	}
}

func TestTokenizerPadsToMaximumLength(t *testing.T) {
	t.Parallel()
	tokenizer, err := NewTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	encoding, err := tokenizer.EncodePadded("hello", 5)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(encoding.Tokens, []string{"[CLS]", "hello", "[SEP]", "[PAD]", "[PAD]"}) {
		t.Fatalf("tokens = %q", encoding.Tokens)
	}
	if !slices.Equal(encoding.AttentionMask, []int64{1, 1, 1, 0, 0}) {
		t.Fatalf("attention mask = %v", encoding.AttentionMask)
	}
}

func TestTokenizerUsesUnknownForLongWords(t *testing.T) {
	t.Parallel()
	tokenizer, err := NewTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	encoding, err := tokenizer.Encode(strings.Repeat("a", maxInputCharactersPerWord+1), 8)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(encoding.Tokens, []string{"[CLS]", "[UNK]", "[SEP]"}) {
		t.Fatalf("tokens = %q", encoding.Tokens)
	}
}

func TestTokenizerVocabulary(t *testing.T) {
	t.Parallel()
	tokenizer, err := NewTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	if tokenizer.VocabularySize() != 30_522 {
		t.Fatalf("VocabularySize() = %d", tokenizer.VocabularySize())
	}
	if token, ok := tokenizer.Token(103); !ok || token != "[MASK]" {
		t.Fatalf("Token(103) = %q, %v", token, ok)
	}
	if _, ok := tokenizer.Token(-1); ok {
		t.Fatal("Token(-1) succeeded")
	}
}

func TestTokenizerRejectsInvalidLength(t *testing.T) {
	t.Parallel()
	tokenizer, err := NewTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := tokenizer.Encode("text", 1); err == nil {
		t.Fatal("Encode() error = nil")
	}
	if _, err := tokenizer.Encode("text", maxSequenceLength+1); err == nil {
		t.Fatal("Encode(oversized) error = nil")
	}
}

func TestTokenizerLoadsCustomCasedVocabulary(t *testing.T) {
	t.Parallel()
	vocabulary := strings.Join([]string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "Hello", "hello"}, "\n")
	tokenizer, err := NewTokenizerFromReader(strings.NewReader(vocabulary), WithLowerCase(false), WithStripAccents(false))
	if err != nil {
		t.Fatal(err)
	}
	encoding, err := tokenizer.Encode("Hello", 8)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(encoding.Tokens, []string{"[CLS]", "Hello", "[SEP]"}) {
		t.Fatalf("tokens = %v", encoding.Tokens)
	}
	if id, ok := tokenizer.TokenID("Hello"); !ok || id != 5 {
		t.Fatalf("TokenID(Hello) = %d, %t", id, ok)
	}
}

func TestTokenizerEncodesDynamicBatch(t *testing.T) {
	t.Parallel()
	tokenizer, err := NewTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	batch, err := tokenizer.EncodeBatch([]string{"hello", "hello world"}, 8)
	if err != nil {
		t.Fatal(err)
	}
	if batch.BatchSize != 2 || batch.SequenceLength != 4 || len(batch.IDs) != 8 ||
		!slices.Equal(batch.AttentionMask, []int64{1, 1, 1, 0, 1, 1, 1, 1}) {
		t.Fatalf("batch = %+v", batch)
	}
}

func TestTokenizerRejectsBlankVocabularyLines(t *testing.T) {
	t.Parallel()
	vocabulary := "[PAD]\n[UNK]\n\n[CLS]\n[SEP]\n[MASK]"
	if _, err := NewTokenizerFromReader(strings.NewReader(vocabulary)); err == nil {
		t.Fatal("NewTokenizerFromReader() error = nil")
	}
}

func TestTokenizerMatchesLongCustomSpecialToken(t *testing.T) {
	t.Parallel()
	special := SpecialTokens{
		Padding: "<padding-token-long>", Unknown: "<unknown-token-long>",
		Classifier: "<classifier-token-long>", Separator: "<separator-token-long>",
		Mask: "<mask-token-long>",
	}
	vocabulary := strings.Join([]string{
		special.Padding, special.Unknown, special.Classifier, special.Separator, special.Mask, "hello",
	}, "\n")
	tokenizer, err := NewTokenizerFromReader(strings.NewReader(vocabulary), WithSpecialTokens(special))
	if err != nil {
		t.Fatal(err)
	}
	encoding, err := tokenizer.Encode("hello <MASK-TOKEN-LONG>", 8)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(encoding.Tokens, []string{special.Classifier, "hello", special.Mask, special.Separator}) {
		t.Fatalf("tokens = %q", encoding.Tokens)
	}
}

func TestTokenizerDropsUnicodeFormatControls(t *testing.T) {
	t.Parallel()
	tokenizer, err := NewTokenizer()
	if err != nil {
		t.Fatal(err)
	}
	encoding, err := tokenizer.Encode("hello\u200eworld", 8)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(encoding.Tokens, []string{"[CLS]", "hello", "##world", "[SEP]"}) {
		t.Fatalf("tokens = %q", encoding.Tokens)
	}
}

func TestSpecialTokensRejectCaseFoldedDuplicates(t *testing.T) {
	t.Parallel()
	tokens := DefaultSpecialTokens()
	tokens.Mask = "[pad]"
	if err := WithSpecialTokens(tokens)(&tokenizerConfig{}); err == nil {
		t.Fatal("WithSpecialTokens() error = nil")
	}
}

func FuzzTokenizerEncode(f *testing.F) {
	tokenizer, err := NewTokenizer()
	if err != nil {
		f.Fatal(err)
	}
	f.Add("Hello, 世界 [MASK]", uint16(32))
	f.Add("\xff\xfe", uint16(2))
	f.Fuzz(func(t *testing.T, text string, rawLength uint16) {
		maximum := 2 + int(rawLength)%510
		encoding, err := tokenizer.Encode(text, maximum)
		if err != nil {
			t.Fatal(err)
		}
		if len(encoding.IDs) != len(encoding.Tokens) || len(encoding.IDs) != len(encoding.AttentionMask) {
			t.Fatalf("inconsistent encoding lengths: %+v", encoding)
		}
		if len(encoding.IDs) > maximum {
			t.Fatalf("encoding length = %d, maximum = %d", len(encoding.IDs), maximum)
		}
	})
}

func BenchmarkTokenizerEncodeBatch(b *testing.B) {
	tokenizer, err := NewTokenizer()
	if err != nil {
		b.Fatal(err)
	}
	texts := make([]string, 32)
	for index := range texts {
		texts[index] = "A fast and reliable inference library for production workloads."
	}
	for b.Loop() {
		if _, err := tokenizer.EncodeBatch(texts, 256); err != nil {
			b.Fatal(err)
		}
	}
}
