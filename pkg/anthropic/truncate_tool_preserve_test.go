package anthropic

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestPreemptiveTruncateDoesNotTreatMaxTokensAsGuaranteedOutput(t *testing.T) {
	message := strings.Repeat("x", 520000)
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
	oldThreshold := contextWindowSize - req.MaxTokens - contextSafetyTokens
	if before <= oldThreshold {
		t.Fatalf("fixture must exceed the old 64K-reservation threshold: %d", before)
	}
	if rounds := PreemptiveTruncateWithWindow(req, 200000); rounds != 0 {
		t.Fatalf("64K max_tokens ceiling must not force early truncation, got %d", rounds)
	}
}

func TestPreemptiveTruncateUsesResolvedModelWindow(t *testing.T) {
	req := &MessageRequest{
		MaxTokens: 64000,
		Messages: []MessageParam{
			{Role: "user", Content: json.RawMessage(`"start"`)},
			{Role: "assistant", Content: json.RawMessage(`"` + strings.Repeat("x", 360000) + `"`)},
			{Role: "user", Content: json.RawMessage(`"middle"`)},
			{Role: "assistant", Content: json.RawMessage(`"response"`)},
			{Role: "user", Content: json.RawMessage(`"latest"`)},
		},
	}
	if rounds := PreemptiveTruncateWithWindow(req, 100000); rounds <= 0 {
		t.Fatalf("expected truncation for the smaller resolved context window, got %d", rounds)
	}
}

func TestFindSafeTruncationStart_PreservesToolPair(t *testing.T) {
	msgs := []MessageParam{
		{Role: "user", Content: json.RawMessage(`"hello"`)},
		{Role: "assistant", Content: json.RawMessage(`[{"type":"tool_use","id":"tu_1","name":"Write","input":{"file_path":"/a.go","content":"..."}}]`)},
		{Role: "user", Content: json.RawMessage(`[{"type":"tool_result","tool_use_id":"tu_1","content":"ok"}]`)},
		{Role: "assistant", Content: json.RawMessage(`"done"`)},
		{Role: "assistant", Content: json.RawMessage(`[{"type":"tool_use","id":"tu_2","name":"Read","input":{"file_path":"/b.go"}}]`)},
		{Role: "user", Content: json.RawMessage(`[{"type":"tool_result","tool_use_id":"tu_2","content":"ok"}]`)},
	}
	got := findSafeTruncationStart(msgs, 4)
	if got != 4 {
		t.Errorf("expected 4 (tool_use plus its tool_result), got %d", got)
	}
}

func TestFindSafeTruncationStart_SkipsOrphanedToolResult(t *testing.T) {
	msgs := []MessageParam{
		{Role: "user", Content: json.RawMessage(`"hello"`)},
		{Role: "assistant", Content: json.RawMessage(`"some text"`)},
		{Role: "user", Content: json.RawMessage(`[{"type":"tool_result","tool_use_id":"tu_1","content":"ok"}]`)},
		{Role: "assistant", Content: json.RawMessage(`"done"`)},
		{Role: "user", Content: json.RawMessage(`"next task"`)},
	}
	got := findSafeTruncationStart(msgs, 2)
	if got != 4 {
		t.Errorf("expected 4 (next regular user turn), got %d", got)
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
	if got := string(req.Messages[len(req.Messages)-1].Content); got != `"latest"` {
		t.Fatalf("latest user turn was not retained: %s", got)
	}
	if string(req.Messages[0].Content) != `"first user message"` {
		t.Fatalf("first user task was not retained: %s", req.Messages[0].Content)
	}
	if req.Messages[1].Role != "assistant" || !isCompleteToolExchangeStart(req.Messages, 1) {
		t.Fatalf("retained history must continue at a complete tool exchange: %#v", req.Messages[1])
	}
	for _, message := range req.Messages {
		if message.Role == "assistant" && strings.Contains(string(message.Content), "auto-truncated") {
			t.Fatal("truncation notice must not be inserted as a synthetic assistant reply")
		}
	}
	if !strings.Contains(string(req.System), truncationNotice) {
		t.Fatalf("system truncation notice missing: %s", req.System)
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

func TestTruncateMessages_AppendsSystemNoticeOnlyOnce(t *testing.T) {
	req := &MessageRequest{
		System: json.RawMessage(`[{"type":"text","text":"original system"}]`),
		Messages: []MessageParam{
			{Role: "user", Content: json.RawMessage(`"one"`)},
			{Role: "assistant", Content: json.RawMessage(`"one response"`)},
			{Role: "user", Content: json.RawMessage(`"two"`)},
			{Role: "assistant", Content: json.RawMessage(`"two response"`)},
			{Role: "user", Content: json.RawMessage(`"three"`)},
			{Role: "assistant", Content: json.RawMessage(`"three response"`)},
			{Role: "user", Content: json.RawMessage(`"four"`)},
			{Role: "assistant", Content: json.RawMessage(`"four response"`)},
			{Role: "user", Content: json.RawMessage(`"latest"`)},
		},
	}
	if !truncateMessages(req) {
		t.Fatal("first truncation failed")
	}
	if !truncateMessages(req) {
		t.Fatal("second truncation failed")
	}
	if count := strings.Count(string(req.System), truncationNotice); count != 1 {
		t.Fatalf("truncation notice count = %d, want 1: %s", count, req.System)
	}
}
