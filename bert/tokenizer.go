package bert

import (
	"bufio"
	"bytes"
	"embed"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

const maxInputCharactersPerWord = 100

//go:embed vocab.txt
var vocabularyFS embed.FS

// SpecialTokens names the control tokens used by a BERT vocabulary.
type SpecialTokens struct {
	Padding    string
	Unknown    string
	Classifier string
	Separator  string
	Mask       string
}

// Encoding is one tokenized model input.
type Encoding struct {
	IDs           []int64
	AttentionMask []int64
	Tokens        []string
}

// BatchEncoding is a padded, row-major batch of tokenized inputs.
type BatchEncoding struct {
	IDs            []int64
	AttentionMask  []int64
	Tokens         [][]string
	BatchSize      int
	SequenceLength int
}

// Tokenizer implements uncased BERT basic and WordPiece tokenization.
type Tokenizer struct {
	vocabulary   map[string]int
	tokens       []string
	special      SpecialTokens
	specialMap   map[string]string
	lowercase    bool
	stripAccents bool
}

// TokenizerOption configures WordPiece tokenization.
type TokenizerOption func(*tokenizerConfig) error

type tokenizerConfig struct {
	special      SpecialTokens
	lowercase    bool
	stripAccents bool
}

// DefaultSpecialTokens returns the conventional BERT control tokens.
func DefaultSpecialTokens() SpecialTokens {
	return SpecialTokens{
		Padding:    "[PAD]",
		Unknown:    "[UNK]",
		Classifier: "[CLS]",
		Separator:  "[SEP]",
		Mask:       "[MASK]",
	}
}

// WithSpecialTokens overrides the vocabulary's control-token names.
func WithSpecialTokens(tokens SpecialTokens) TokenizerOption {
	return func(config *tokenizerConfig) error {
		values := [...]string{tokens.Padding, tokens.Unknown, tokens.Classifier, tokens.Separator, tokens.Mask}
		for _, token := range values {
			if token == "" {
				return errors.New("bert: special token cannot be empty")
			}
		}
		config.special = tokens
		return nil
	}
}

// WithLowerCase controls lowercasing before WordPiece lookup. It defaults to
// true for compatibility with the embedded bert-base-uncased vocabulary.
func WithLowerCase(enabled bool) TokenizerOption {
	return func(config *tokenizerConfig) error {
		config.lowercase = enabled
		return nil
	}
}

// WithStripAccents controls removal of Unicode combining marks. It defaults
// to true for compatibility with the embedded bert-base-uncased vocabulary.
func WithStripAccents(enabled bool) TokenizerOption {
	return func(config *tokenizerConfig) error {
		config.stripAccents = enabled
		return nil
	}
}

// NewTokenizer loads the embedded bert-base-uncased vocabulary.
func NewTokenizer(options ...TokenizerOption) (*Tokenizer, error) {
	contents, err := vocabularyFS.ReadFile("vocab.txt")
	if err != nil {
		return nil, fmt.Errorf("bert: read embedded vocabulary: %w", err)
	}
	return NewTokenizerFromReader(bytes.NewReader(contents), options...)
}

// NewTokenizerFromFile loads a line-delimited WordPiece vocabulary from path.
func NewTokenizerFromFile(path string, options ...TokenizerOption) (*Tokenizer, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("bert: open vocabulary: %w", err)
	}
	tokenizer, tokenizerErr := NewTokenizerFromReader(file, options...)
	if closeErr := file.Close(); closeErr != nil {
		return nil, errors.Join(tokenizerErr, fmt.Errorf("bert: close vocabulary: %w", closeErr))
	}
	return tokenizer, tokenizerErr
}

// NewTokenizerFromReader loads a line-delimited WordPiece vocabulary.
func NewTokenizerFromReader(reader io.Reader, options ...TokenizerOption) (*Tokenizer, error) {
	if reader == nil {
		return nil, errors.New("bert: vocabulary reader cannot be nil")
	}
	config := tokenizerConfig{
		special:      DefaultSpecialTokens(),
		lowercase:    true,
		stripAccents: true,
	}
	for _, option := range options {
		if option == nil {
			return nil, errors.New("bert: tokenizer option cannot be nil")
		}
		if err := option(&config); err != nil {
			return nil, err
		}
	}

	vocabulary := make(map[string]int)
	tokens := make([]string, 0)
	scanner := bufio.NewScanner(reader)
	for scanner.Scan() {
		token := scanner.Text()
		if token == "" {
			continue
		}
		if _, exists := vocabulary[token]; exists {
			return nil, fmt.Errorf("bert: duplicate vocabulary token %q", token)
		}
		vocabulary[token] = len(tokens)
		tokens = append(tokens, token)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("bert: read vocabulary: %w", err)
	}

	special := config.special
	for _, token := range [...]string{special.Padding, special.Unknown, special.Classifier, special.Separator, special.Mask} {
		if _, ok := vocabulary[token]; !ok {
			return nil, fmt.Errorf("bert: vocabulary is missing %s", token)
		}
	}
	specialMap := map[string]string{
		strings.ToUpper(special.Padding):    special.Padding,
		strings.ToUpper(special.Unknown):    special.Unknown,
		strings.ToUpper(special.Classifier): special.Classifier,
		strings.ToUpper(special.Separator):  special.Separator,
		strings.ToUpper(special.Mask):       special.Mask,
	}
	return &Tokenizer{
		vocabulary:   vocabulary,
		tokens:       tokens,
		special:      special,
		specialMap:   specialMap,
		lowercase:    config.lowercase,
		stripAccents: config.stripAccents,
	}, nil
}

// Encode tokenizes text, adds classifier and separator tokens, and truncates
// to maxLength while always preserving the final separator.
func (t *Tokenizer) Encode(text string, maxLength int) (Encoding, error) {
	if t == nil {
		return Encoding{}, errors.New("bert: nil tokenizer")
	}
	if maxLength < 2 {
		return Encoding{}, errors.New("bert: maximum length must be at least two")
	}
	wordPieces := make([]string, 0)
	for _, token := range t.basicTokens(text) {
		if special, ok := t.specialMap[strings.ToUpper(token)]; ok {
			wordPieces = append(wordPieces, special)
			continue
		}
		wordPieces = append(wordPieces, t.wordPiece(token)...)
	}
	wordPieces = wordPieces[:min(len(wordPieces), maxLength-2)]

	tokens := make([]string, 0, len(wordPieces)+2)
	tokens = append(tokens, t.special.Classifier)
	tokens = append(tokens, wordPieces...)
	tokens = append(tokens, t.special.Separator)
	ids := make([]int64, len(tokens))
	attentionMask := make([]int64, len(tokens))
	for index, token := range tokens {
		id, ok := t.vocabulary[token]
		if !ok {
			id = t.vocabulary[t.special.Unknown]
			tokens[index] = t.special.Unknown
		}
		ids[index] = int64(id)
		attentionMask[index] = 1
	}
	return Encoding{IDs: ids, AttentionMask: attentionMask, Tokens: tokens}, nil
}

// EncodePadded is like Encode and appends padding tokens until the encoding
// reaches maxLength. Padding positions have an attention value of zero.
func (t *Tokenizer) EncodePadded(text string, maxLength int) (Encoding, error) {
	encoding, err := t.Encode(text, maxLength)
	if err != nil {
		return Encoding{}, err
	}
	return t.Pad(encoding, maxLength)
}

// EncodeBatch tokenizes texts and pads every row to the longest encoded
// sequence in the batch, up to maxLength.
func (t *Tokenizer) EncodeBatch(texts []string, maxLength int) (BatchEncoding, error) {
	if t == nil {
		return BatchEncoding{}, errors.New("bert: nil tokenizer")
	}
	if len(texts) == 0 {
		return BatchEncoding{}, errors.New("bert: text batch cannot be empty")
	}
	rows := make([]Encoding, len(texts))
	sequenceLength := 0
	for index, text := range texts {
		encoding, err := t.Encode(text, maxLength)
		if err != nil {
			return BatchEncoding{}, fmt.Errorf("bert: encode batch item %d: %w", index, err)
		}
		rows[index] = encoding
		sequenceLength = max(sequenceLength, len(encoding.IDs))
	}
	result := BatchEncoding{
		IDs:            make([]int64, 0, len(texts)*sequenceLength),
		AttentionMask:  make([]int64, 0, len(texts)*sequenceLength),
		Tokens:         make([][]string, len(texts)),
		BatchSize:      len(texts),
		SequenceLength: sequenceLength,
	}
	for index, row := range rows {
		padded, err := t.Pad(row, sequenceLength)
		if err != nil {
			return BatchEncoding{}, fmt.Errorf("bert: pad batch item %d: %w", index, err)
		}
		result.IDs = append(result.IDs, padded.IDs...)
		result.AttentionMask = append(result.AttentionMask, padded.AttentionMask...)
		result.Tokens[index] = padded.Tokens
	}
	return result, nil
}

// Pad returns a copy of encoding padded to length. It rejects encodings that
// are already longer than length.
func (t *Tokenizer) Pad(encoding Encoding, length int) (Encoding, error) {
	if t == nil {
		return Encoding{}, errors.New("bert: nil tokenizer")
	}
	if length < len(encoding.IDs) || len(encoding.IDs) != len(encoding.AttentionMask) || len(encoding.IDs) != len(encoding.Tokens) {
		return Encoding{}, errors.New("bert: cannot pad malformed or oversized encoding")
	}
	encoding.IDs = append([]int64(nil), encoding.IDs...)
	encoding.AttentionMask = append([]int64(nil), encoding.AttentionMask...)
	encoding.Tokens = append([]string(nil), encoding.Tokens...)
	paddingID := int64(t.vocabulary[t.special.Padding])
	for len(encoding.IDs) < length {
		encoding.IDs = append(encoding.IDs, paddingID)
		encoding.AttentionMask = append(encoding.AttentionMask, 0)
		encoding.Tokens = append(encoding.Tokens, t.special.Padding)
	}
	return encoding, nil
}

// TokenID returns the vocabulary ID for token.
func (t *Tokenizer) TokenID(token string) (int, bool) {
	if t == nil {
		return 0, false
	}
	id, ok := t.vocabulary[token]
	return id, ok
}

// SpecialTokens returns the tokenizer's configured control tokens.
func (t *Tokenizer) SpecialTokens() SpecialTokens {
	if t == nil {
		return SpecialTokens{}
	}
	return t.special
}

// Token returns the vocabulary token for id.
func (t *Tokenizer) Token(id int) (string, bool) {
	if t == nil || id < 0 || id >= len(t.tokens) {
		return "", false
	}
	return t.tokens[id], true
}

// VocabularySize returns the number of tokens in the vocabulary.
func (t *Tokenizer) VocabularySize() int {
	if t == nil {
		return 0
	}
	return len(t.tokens)
}

// MaskPositions returns all mask-token offsets in an encoding.
func (t *Tokenizer) MaskPositions(encoding Encoding) []int {
	if t == nil {
		return nil
	}
	positions := make([]int, 0)
	for index, token := range encoding.Tokens {
		if token == t.special.Mask {
			positions = append(positions, index)
		}
	}
	return positions
}

func (t *Tokenizer) labels() map[int]string {
	labels := make(map[int]string, len(t.tokens))
	for id, token := range t.tokens {
		labels[id] = token
	}
	return labels
}

func (t *Tokenizer) basicTokens(text string) []string {
	runes := []rune(text)
	tokens := make([]string, 0)
	current := make([]rune, 0)
	flush := func() {
		if len(current) == 0 {
			return
		}
		tokens = append(tokens, t.normalize(string(current)))
		current = current[:0]
	}

	for index := 0; index < len(runes); index++ {
		character := runes[index]
		if character == '[' {
			end := index + 1
			for end < len(runes) && end-index <= 16 && runes[end] != ']' {
				end++
			}
			if end < len(runes) && runes[end] == ']' {
				candidate := string(runes[index : end+1])
				if special, ok := t.specialMap[strings.ToUpper(candidate)]; ok {
					flush()
					tokens = append(tokens, special)
					index = end
					continue
				}
			}
		}
		switch {
		case unicode.IsSpace(character):
			flush()
		case unicode.IsControl(character) || character == utf8.RuneError:
			flush()
		case unicode.IsPunct(character) || isCJK(character):
			flush()
			tokens = append(tokens, t.normalize(string(character)))
		default:
			current = append(current, character)
		}
	}
	flush()
	return tokens
}

func (t *Tokenizer) wordPiece(token string) []string {
	if _, ok := t.vocabulary[token]; ok {
		return []string{token}
	}
	characters := []rune(token)
	if len(characters) > maxInputCharactersPerWord {
		return []string{t.special.Unknown}
	}

	pieces := make([]string, 0)
	for start := 0; start < len(characters); {
		found := ""
		end := len(characters)
		for end > start {
			candidate := string(characters[start:end])
			if start > 0 {
				candidate = "##" + candidate
			}
			if _, ok := t.vocabulary[candidate]; ok {
				found = candidate
				break
			}
			end--
		}
		if found == "" {
			return []string{t.special.Unknown}
		}
		pieces = append(pieces, found)
		start = end
	}
	return pieces
}

func (t *Tokenizer) normalize(token string) string {
	if t.lowercase {
		token = strings.ToLower(token)
	}
	if !t.stripAccents {
		return token
	}
	return strings.Map(func(character rune) rune {
		if unicode.Is(unicode.Mn, character) {
			return -1
		}
		return character
	}, norm.NFD.String(token))
}

func isCJK(character rune) bool {
	return character >= 0x4E00 && character <= 0x9FFF ||
		character >= 0x3400 && character <= 0x4DBF ||
		character >= 0x20000 && character <= 0x2A6DF ||
		character >= 0x2A700 && character <= 0x2B73F ||
		character >= 0x2B740 && character <= 0x2B81F ||
		character >= 0x2B820 && character <= 0x2CEAF ||
		character >= 0xF900 && character <= 0xFAFF ||
		character >= 0x2F800 && character <= 0x2FA1F
}
