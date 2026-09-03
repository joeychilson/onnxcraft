package bert

import (
	"slices"
	"strings"
	"testing"
)

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
