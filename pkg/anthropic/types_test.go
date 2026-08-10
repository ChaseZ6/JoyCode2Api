package anthropic

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestContentBlockMarshal_OmitsTextOnToolUse(t *testing.T) {
	b, err := json.Marshal(ContentBlock{
		Type: "tool_use", ID: "call_1", Name: "get_weather",
		Input: map[string]any{"city": "Beijing"},
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if bytes.Contains(b, []byte(`"text"`)) {
		t.Fatalf("tool_use block must not carry a text field: %s", b)
	}
	for _, want := range []string{`"type":"tool_use"`, `"id":"call_1"`, `"name":"get_weather"`, `"city":"Beijing"`} {
		if !bytes.Contains(b, []byte(want)) {
			t.Errorf("missing %s in %s", want, b)
		}
	}
}

func TestContentBlockMarshal_KeepsEmptyTextOnTextBlock(t *testing.T) {
	b, err := json.Marshal(ContentBlock{Type: "text", Text: ""})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !bytes.Contains(b, []byte(`"text":""`)) {
		t.Fatalf(`text block must keep "text":"" for content_block_start: %s`, b)
	}
}
