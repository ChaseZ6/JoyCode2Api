package openai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
)

func gptSolModel() joycode.ModelInfo {
	return joycode.ModelInfo{
		Label: "GPT-5.6 Sol", ChatAPIModel: "GPT-5.6 Sol", RespMaxTokens: 64000,
		ExtJSON: map[string]interface{}{"adapterType": joycode.AdapterOpenAIResponse},
	}
}

func TestTranslateChatToResponses_ModernParameters(t *testing.T) {
	parallel := false
	strict := true
	req := &ChatRequest{
		Model:             "GPT-5.6 Sol",
		Messages:          json.RawMessage(`[{"role":"user","content":"hello"}]`),
		MaxTokens:         64000,
		ReasoningEffort:   "high",
		ParallelToolCalls: &parallel,
		ResponseFormat: json.RawMessage(`{
			"type":"json_schema",
			"json_schema":{"name":"answer","description":"Answer schema","strict":true,"schema":{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}}
		}`),
	}
	body, err := TranslateChatToResponses(req, gptSolModel())
	if err != nil {
		t.Fatal(err)
	}
	if body["max_output_tokens"] != 64000 {
		t.Errorf("max_output_tokens = %v", body["max_output_tokens"])
	}
	reasoning := body["reasoning"].(map[string]interface{})
	if reasoning["effort"] != "high" {
		t.Errorf("reasoning = %v", reasoning)
	}
	if got := body["parallel_tool_calls"]; got != false {
		t.Errorf("parallel_tool_calls = %#v", got)
	}
	text := body["text"].(map[string]interface{})
	format := text["format"].(map[string]interface{})
	if format["type"] != "json_schema" || format["name"] != "answer" || format["strict"] != strict {
		t.Errorf("format = %#v", format)
	}
	if _, ok := format["schema"].(json.RawMessage); !ok {
		t.Errorf("schema type = %T", format["schema"])
	}
}

func TestTranslateChatToResponses_MultimodalInput(t *testing.T) {
	req := &ChatRequest{
		Messages: json.RawMessage(`[
			{"role":"system","content":"be concise"},
			{"role":"user","content":[
				{"type":"text","text":"describe"},
				{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA","detail":"high"}}
			]}
		]`),
	}
	body, err := TranslateChatToResponses(req, gptSolModel())
	if err != nil {
		t.Fatal(err)
	}
	input := body["input"].([]map[string]interface{})
	if len(input) != 2 || input[0]["role"] != "developer" || input[1]["role"] != "user" {
		t.Fatalf("input = %#v", input)
	}
	parts := input[1]["content"].([]map[string]interface{})
	if parts[0]["type"] != "input_text" || parts[0]["text"] != "describe" {
		t.Errorf("text part = %#v", parts[0])
	}
	if parts[1]["type"] != "input_image" || parts[1]["image_url"] != "data:image/png;base64,AAAA" || parts[1]["detail"] != "high" {
		t.Errorf("image part = %#v", parts[1])
	}
}

func TestTranslateChatToResponses_ToolChoice(t *testing.T) {
	req := &ChatRequest{
		Messages:   json.RawMessage(`[{"role":"user","content":"weather"}]`),
		Tools:      json.RawMessage(`[{"type":"function","function":{"name":"weather","description":"Weather","parameters":{"type":"object"}}}]`),
		ToolChoice: json.RawMessage(`{"type":"function","function":{"name":"weather"}}`),
	}
	body, err := TranslateChatToResponses(req, gptSolModel())
	if err != nil {
		t.Fatal(err)
	}
	choice := body["tool_choice"].(map[string]interface{})
	if choice["type"] != "function" || choice["name"] != "weather" {
		t.Errorf("tool_choice = %#v", choice)
	}
	tools := body["tools"].([]map[string]interface{})
	if len(tools) != 1 || tools[0]["name"] != "weather" {
		t.Errorf("tools = %#v", tools)
	}
}

func TestTranslateChatToResponses_RejectsInvalidInput(t *testing.T) {
	cases := []struct {
		name string
		req  ChatRequest
	}{
		{"missing messages", ChatRequest{}},
		{"messages object", ChatRequest{Messages: json.RawMessage(`{"role":"user"}`)}},
		{"unknown content part", ChatRequest{Messages: json.RawMessage(`[{"role":"user","content":[{"type":"audio"}]}]`)}},
		{"invalid tool choice", ChatRequest{
			Messages:   json.RawMessage(`[{"role":"user","content":"x"}]`),
			Tools:      json.RawMessage(`[{"type":"function","function":{"name":"f"}}]`),
			ToolChoice: json.RawMessage(`"sometimes"`),
		}},
		{"invalid reasoning effort", ChatRequest{Messages: json.RawMessage(`[{"role":"user","content":"x"}]`), ReasoningEffort: "extreme"}},
		{"invalid response format", ChatRequest{Messages: json.RawMessage(`[{"role":"user","content":"x"}]`), ResponseFormat: json.RawMessage(`{"type":"xml"}`)}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := TranslateChatToResponses(&tc.req, gptSolModel()); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func TestTranslateResponsesToChat_ReasoningToolsAndUsageDetails(t *testing.T) {
	raw := map[string]interface{}{
		"id": "resp_1", "status": "completed",
		"output": []interface{}{
			map[string]interface{}{"type": "reasoning", "summary": []interface{}{
				map[string]interface{}{"type": "summary_text", "text": "I checked the inputs."},
			}},
			map[string]interface{}{"type": "message", "content": []interface{}{
				map[string]interface{}{"type": "output_text", "text": "Calling tools"},
			}},
			map[string]interface{}{"type": "function_call", "id": "item_1", "call_id": "call_1", "name": "one", "arguments": `{"a":1}`},
			map[string]interface{}{"type": "function_call", "id": "item_2", "call_id": "call_2", "name": "two", "arguments": `{"b":2}`},
		},
		"usage": map[string]interface{}{
			"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
			"input_tokens_details":  map[string]interface{}{"cached_tokens": 40},
			"output_tokens_details": map[string]interface{}{"reasoning_tokens": 8},
		},
	}
	out, inTk, outTk, err := TranslateResponsesToChat(raw, "GPT-5.6 Sol")
	if err != nil {
		t.Fatal(err)
	}
	if inTk != 100 || outTk != 20 {
		t.Fatalf("usage return = %d/%d", inTk, outTk)
	}
	choice := out["choices"].([]map[string]interface{})[0]
	if choice["finish_reason"] != "tool_calls" {
		t.Errorf("finish_reason = %v", choice["finish_reason"])
	}
	message := choice["message"].(map[string]interface{})
	if message["content"] != "Calling tools" || message["reasoning_content"] != "I checked the inputs." {
		t.Errorf("message = %#v", message)
	}
	calls := message["tool_calls"].([]map[string]interface{})
	if len(calls) != 2 || calls[0]["id"] != "call_1" || calls[1]["id"] != "call_2" {
		t.Errorf("tool_calls = %#v", calls)
	}
	usage := out["usage"].(map[string]interface{})
	promptDetails := usage["prompt_tokens_details"].(map[string]interface{})
	completionDetails := usage["completion_tokens_details"].(map[string]interface{})
	if promptDetails["cached_tokens"] != 40 || completionDetails["reasoning_tokens"] != 8 {
		t.Errorf("usage details = %#v", usage)
	}
}

func TestTranslateResponsesToChat_FinishReasons(t *testing.T) {
	cases := []struct {
		name   string
		reason string
		want   string
	}{
		{"length", "max_output_tokens", "length"},
		{"content filter", "content_filter", "content_filter"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := map[string]interface{}{
				"status": "incomplete",
				"output": []interface{}{map[string]interface{}{
					"type": "message", "content": []interface{}{map[string]interface{}{"type": "output_text", "text": "partial"}},
				}},
				"incomplete_details": map[string]interface{}{"reason": tc.reason},
			}
			out, _, _, err := TranslateResponsesToChat(raw, "GPT-5.6 Sol")
			if err != nil {
				t.Fatal(err)
			}
			choice := out["choices"].([]map[string]interface{})[0]
			if choice["finish_reason"] != tc.want {
				t.Errorf("finish_reason = %v, want %s", choice["finish_reason"], tc.want)
			}
		})
	}
}

func TestTranslateResponsesToChat_FailedIsError(t *testing.T) {
	for _, raw := range []map[string]interface{}{
		{"status": "failed", "error": map[string]interface{}{"message": "model unavailable"}},
		{"status": "failed", "error": nil},
		{"status": "incomplete", "incomplete_details": map[string]interface{}{"reason": "unknown"}},
	} {
		if _, _, _, err := TranslateResponsesToChat(raw, "GPT-5.6 Sol"); err == nil {
			t.Errorf("expected error for %#v", raw)
		}
	}
}

func TestChatResponsesStream_InterleavedToolsUseStableIndices(t *testing.T) {
	rec := httptest.NewRecorder()
	state := newChatResponsesStreamState(rec, rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil), "GPT-5.6 Sol", false)
	state.emitStart()
	if err := state.addTool("item_a", "call_a", "alpha"); err != nil {
		t.Fatal(err)
	}
	if err := state.addTool("item_b", "call_b", "beta"); err != nil {
		t.Fatal(err)
	}
	if err := state.toolDelta("item_a", "", `{"a":`); err != nil {
		t.Fatal(err)
	}
	if err := state.toolDelta("item_b", "", `{"b":2}`); err != nil {
		t.Fatal(err)
	}
	if err := state.toolDelta("item_a", "", `1}`); err != nil {
		t.Fatal(err)
	}
	state.complete("")

	out := rec.Body.String()
	wantDeltas := map[int][]string{0: {`{"a":`, `1}`}, 1: {`{"b":2}`}}
	gotDeltas := map[int][]string{}
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimPrefix(line, "data: ")
		if !strings.HasPrefix(line, "{") {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					ToolCalls []struct {
						Index    int `json:"index"`
						Function struct {
							Arguments string `json:"arguments"`
						} `json:"function"`
					} `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal([]byte(line), &chunk) != nil || len(chunk.Choices) == 0 {
			continue
		}
		for _, call := range chunk.Choices[0].Delta.ToolCalls {
			if call.Function.Arguments != "" {
				gotDeltas[call.Index] = append(gotDeltas[call.Index], call.Function.Arguments)
			}
		}
	}
	if !reflect.DeepEqual(gotDeltas, wantDeltas) {
		t.Errorf("tool deltas = %#v, want %#v\n%s", gotDeltas, wantDeltas, out)
	}
	if !strings.Contains(out, `"finish_reason":"tool_calls"`) {
		t.Errorf("missing tool_calls finish:\n%s", out)
	}
	if strings.Count(out, "data: [DONE]") != 1 {
		t.Errorf("DONE count = %d", strings.Count(out, "data: [DONE]"))
	}
}

func TestChatResponsesStream_IncludeUsage(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	state := newChatResponsesStreamState(rec, rec, req, "GPT-5.6 Sol", true)
	state.inputTokens, state.outputTokens = 11, 5
	state.complete("stop")
	out := rec.Body.String()
	if !strings.Contains(out, `"choices":[],"created"`) || !strings.Contains(out, `"prompt_tokens":11`) || !strings.Contains(out, `"completion_tokens":5`) {
		t.Errorf("missing usage chunk:\n%s", out)
	}
	if strings.Index(out, `"choices":[]`) > strings.Index(out, "data: [DONE]") {
		t.Error("usage chunk must precede DONE")
	}
}

func TestChatResponsesStream_ErrorDoesNotSendDone(t *testing.T) {
	rec := httptest.NewRecorder()
	state := newChatResponsesStreamState(rec, rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil), "GPT-5.6 Sol", false)
	state.fail(`upstream said "no"`, "upstream_error")
	state.complete("stop")
	out := rec.Body.String()
	if strings.Contains(out, "data: [DONE]") {
		t.Errorf("error stream must not send DONE:\n%s", out)
	}
	var payload map[string]interface{}
	line := strings.TrimPrefix(strings.TrimSpace(out), "data: ")
	if err := json.Unmarshal([]byte(line), &payload); err != nil {
		t.Fatalf("error event is invalid JSON: %v\n%s", err, out)
	}
}

func TestHandleChatResponsesEvent_Failed(t *testing.T) {
	rec := httptest.NewRecorder()
	state := newChatResponsesStreamState(rec, rec, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil), "GPT-5.6 Sol", false)
	event := responsesStreamEvent{Type: "response.failed"}
	event.Response.Error = map[string]interface{}{"message": "model unavailable"}
	if err := handleChatResponsesEvent(state, &event); err == nil || !strings.Contains(err.Error(), "model unavailable") {
		t.Fatalf("error = %v", err)
	}
}

func TestConnectUpstreamStream_FastFailureUsesHTTPError(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	resp, committed := connectUpstreamStream(rec, rec, req, func() (*http.Response, error) {
		return nil, fmt.Errorf("fast failure")
	})
	if resp != nil || committed {
		t.Fatalf("resp=%v committed=%v", resp, committed)
	}
	if rec.Code != http.StatusBadGateway {
		t.Errorf("status = %d, want 502", rec.Code)
	}
	if strings.Contains(rec.Body.String(), ": keepalive") {
		t.Error("fast failure must not commit an SSE heartbeat")
	}
}

func TestConnectUpstreamStream_SlowFailureUsesSSEError(t *testing.T) {
	old := upstreamConnectHeartbeatInterval
	upstreamConnectHeartbeatInterval = 5 * time.Millisecond
	defer func() { upstreamConnectHeartbeatInterval = old }()

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	resp, committed := connectUpstreamStream(rec, rec, req, func() (*http.Response, error) {
		time.Sleep(20 * time.Millisecond)
		return nil, fmt.Errorf(`slow "failure"`)
	})
	if resp != nil || !committed {
		t.Fatalf("resp=%v committed=%v", resp, committed)
	}
	out := rec.Body.String()
	if !strings.Contains(out, ": keepalive") || strings.Contains(out, "data: [DONE]") {
		t.Errorf("unexpected SSE output:\n%s", out)
	}
	var foundJSON bool
	for _, line := range strings.Split(out, "\n") {
		if !strings.HasPrefix(line, "data: {") {
			continue
		}
		foundJSON = true
		var payload map[string]interface{}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &payload); err != nil {
			t.Fatalf("error event is invalid JSON: %v\n%s", err, out)
		}
	}
	if !foundJSON {
		t.Fatalf("missing SSE error event:\n%s", out)
	}
}

func TestReadResponsesSSE_BusinessError(t *testing.T) {
	reads := readResponsesSSE(strings.NewReader(`data: {"code":"1","echo":"gateway failed"}` + "\n\n"))
	item, ok := <-reads
	if !ok || item.Err == nil || !strings.Contains(item.Err.Error(), "gateway failed") {
		t.Fatalf("item = %#v ok=%v", item, ok)
	}
}
