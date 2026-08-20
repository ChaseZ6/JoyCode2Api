package anthropic

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestPreemptiveTruncateReservesOutputBudget(t *testing.T) {
	message := strings.Repeat("context ", 550000)
	req := &MessageRequest{
		MaxTokens: 64000,
		Messages: []MessageParam{
			{Role: "user", Content: json.RawMessage(`"start"`)},
			{Role: "assistant", Content: json.RawMessage(`"` + message + `"`)},
			{Role: "user", Content: json.RawMessage(`"middle 1"`)},
			{Role: "assistant", Content: json.RawMessage(`"middle 2"`)},
			{Role: "user", Content: json.RawMessage(`"middle 3"`)},
			{Role: "user", Content: json.RawMessage(`"latest"`)},
		},
	}
	before := estimateTokens(req)
	if before <= contextWindowSize-64000-contextSafetyTokens {
		t.Fatalf("fixture must exceed output-aware threshold: %d", before)
	}
	if rounds := PreemptiveTruncate(req); rounds <= 0 {
		t.Fatalf("expected output-aware truncation, got %d", rounds)
	}
}

func TestFindToolPairBoundary_NoToolUse(t *testing.T) {
	msgs := []MessageParam{
		{Role: "user", Content: json.RawMessage(`"hello"`)},
		{Role: "assistant", Content: json.RawMessage(`"hi"`)},
		{Role: "user", Content: json.RawMessage(`"how are you"`)},
		{Role: "assistant", Content: json.RawMessage(`"fine"`)},
	}
	got := findToolPairBoundary(msgs, 2)
	if got != 2 {
		t.Errorf("expected 2, got %d", got)
	}
}

func TestFindToolPairBoundary_SplitsToolPair(t *testing.T) {
	msgs := []MessageParam{
		{Role: "user", Content: json.RawMessage(`"hello"`)},
		{Role: "assistant", Content: json.RawMessage(`[{"type":"tool_use","id":"tu_1","name":"Write","input":{"file_path":"/a.go","content":"..."}}]`)},
		{Role: "user", Content: json.RawMessage(`[{"type":"tool_result","tool_use_id":"tu_1","content":"ok"}]`)},
		{Role: "assistant", Content: json.RawMessage(`"done"`)},
		{Role: "user", Content: json.RawMessage(`"next"`)},
	}
	got := findToolPairBoundary(msgs, 2)
	if got != 1 {
		t.Errorf("expected 1 (include tool_use with its tool_result), got %d", got)
	}
}

func TestFindToolPairBoundary_OrphanedToolResult(t *testing.T) {
	msgs := []MessageParam{
		{Role: "user", Content: json.RawMessage(`"hello"`)},
		{Role: "assistant", Content: json.RawMessage(`"some text"`)},
		{Role: "user", Content: json.RawMessage(`[{"type":"tool_result","tool_use_id":"tu_1","content":"ok"}]`)},
		{Role: "assistant", Content: json.RawMessage(`"done"`)},
	}
	got := findToolPairBoundary(msgs, 2)
	if got != 3 {
		t.Errorf("expected 3 (skip orphaned tool_result), got %d", got)
	}
}

func TestTruncateMessages_PreservesToolPairs(t *testing.T) {
	msgs := []MessageParam{
		{Role: "user", Content: json.RawMessage(`"first user message"`)},
		{Role: "assistant", Content: json.RawMessage(`"response 1"`)},
		{Role: "user", Content: json.RawMessage(`"msg 2"`)},
		{Role: "assistant", Content: json.RawMessage(`[{"type":"tool_use","id":"tu_1","name":"Write","input":{"file_path":"/a.go","content":"package main"}}]`)},
		{Role: "user", Content: json.RawMessage(`[{"type":"tool_result","tool_use_id":"tu_1","content":"File written successfully"}]`)},
		{Role: "assistant", Content: json.RawMessage(`"done"`)},
		{Role: "user", Content: json.RawMessage(`"latest"`)},
	}

	req := &MessageRequest{Messages: msgs}
	result := truncateMessages(req)
	if !result {
		t.Fatal("expected truncation to succeed")
	}

	toolUseIDs := map[string]bool{}
	for _, m := range req.Messages {
		if m.Role == "assistant" {
			var blocks []contentBlock
			if json.Unmarshal(m.Content, &blocks) == nil {
				for _, b := range blocks {
					if b.Type == "tool_use" && b.ID != "" {
						toolUseIDs[b.ID] = true
					}
				}
			}
		}
	}
	for _, m := range req.Messages {
		if m.Role == "user" {
			var blocks []contentBlock
			if json.Unmarshal(m.Content, &blocks) == nil {
				for _, b := range blocks {
					if b.Type == "tool_result" && b.ToolUseID != "" {
						if !toolUseIDs[b.ToolUseID] {
							t.Errorf("orphaned tool_result found: tool_use_id=%s", b.ToolUseID)
						}
					}
				}
			}
		}
	}
}
