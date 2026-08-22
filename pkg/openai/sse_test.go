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

// TestReadChatSSE_UsageAfterDone: with stream_options.include_usage the
// usage-only chunk arrives AFTER [DONE]. The reader must not stop at [DONE] —
// it must forward the Done marker and keep reading so the trailing usage chunk
// is captured (otherwise token stats are always 0).
func TestReadChatSSE_UsageAfterDone(t *testing.T) {
	input := `data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]

data: {"choices":[],"usage":{"prompt_tokens":120,"completion_tokens":34}}

`
	items := collectChatReads(input)
	// content, finish, Done marker, trailing usage chunk.
	if len(items) != 4 {
		t.Fatalf("items = %d, want 4 (content, finish, Done, usage): %#v", len(items), items)
	}
	if !items[2].Done {
		t.Errorf("third item = %#v, want Done marker", items[2])
	}
	usage := items[3]
	if usage.InputTokens != 120 || usage.OutputTokens != 34 {
		t.Errorf("trailing usage chunk not captured: %#v", usage)
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
