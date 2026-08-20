package openai

import (
	"context"
	"strings"
	"testing"
)

func collectChatReads(input string) []chatStreamRead {
	var out []chatStreamRead
	for item := range readChatSSE(context.Background(), strings.NewReader(input)) {
		out = append(out, item)
	}
	return out
}

func TestReadChatSSE_StrictDoneAndFinish(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"literal [DONE] text"},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":11,"completion_tokens":5}}

data: [DONE]

`
	items := collectChatReads(input)
	if len(items) != 3 {
		t.Fatalf("items = %d, want 3: %#v", len(items), items)
	}
	if items[0].Done {
		t.Fatal("[DONE] inside JSON content must not terminate the stream")
	}
	if items[1].FinishReason != "stop" || items[1].InputTokens != 11 || items[1].OutputTokens != 5 {
		t.Errorf("finish item = %#v", items[1])
	}
	if !items[2].Done {
		t.Errorf("final item = %#v, want Done", items[2])
	}
}

func TestReadChatSSE_BusinessError(t *testing.T) {
	items := collectChatReads(`data: {"code":"1","echo":"gateway failed"}` + "\n\n")
	if len(items) != 1 || items[0].BusinessErr == nil || !strings.Contains(items[0].BusinessErr.Error(), "gateway failed") {
		t.Fatalf("items = %#v", items)
	}
}

func TestReadChatSSE_InvalidJSON(t *testing.T) {
	items := collectChatReads("data: {bad}\n\n")
	if len(items) != 1 || items[0].Err == nil {
		t.Fatalf("items = %#v", items)
	}
}

func TestReadChatSSE_MultilineData(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"hello"},` + "\n" +
		`data: "finish_reason":null}]}` + "\n\n" +
		"data: [DONE]\n\n"
	items := collectChatReads(input)
	if len(items) != 2 || items[0].Err != nil || !strings.Contains(items[0].Data, "hello") || !items[1].Done {
		t.Fatalf("items = %#v", items)
	}
}
