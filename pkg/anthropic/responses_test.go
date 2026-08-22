package anthropic

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

func solModel() joycode.ModelInfo {
	return joycode.ModelInfo{
		Label: "GPT-5.6 Sol", ChatAPIModel: "GPT-5.6 Sol",
		ExtJSON: map[string]interface{}{"adapterType": joycode.AdapterOpenAIResponse},
	}
}

// The Responses API rejects "messages" and "max_tokens"; it requires "input" and
// "max_output_tokens".
func TestTranslateAnthropicToResponses_UsesResponsesShape(t *testing.T) {
	req := &MessageRequest{
		Model:     "GPT-5.6 Sol",
		MaxTokens: 256,
		System:    json.RawMessage(`"be terse"`),
		Messages: []MessageParam{
			{Role: "user", Content: json.RawMessage(`"hello"`)},
		},
	}
	body := TranslateAnthropicToResponses(req, solModel())

	if _, bad := body["messages"]; bad {
		t.Error("body must not contain 'messages'; the Responses API rejects it")
	}
	if _, bad := body["max_tokens"]; bad {
		t.Error("body must not contain 'max_tokens'; the Responses API rejects it")
	}
	if body["max_output_tokens"] != 256 {
		t.Errorf("max_output_tokens = %v, want 256", body["max_output_tokens"])
	}
	if body["model"] != "GPT-5.6 Sol" {
		t.Errorf("model = %v, want GPT-5.6 Sol", body["model"])
	}
	if body["instructions"] != "be terse" {
		t.Errorf("instructions = %v, want the system prompt", body["instructions"])
	}
	input, ok := body["input"].([]map[string]interface{})
	if !ok || len(input) != 1 {
		t.Fatalf("input = %v, want one item", body["input"])
	}
	if input[0]["role"] != "user" || input[0]["content"] != "hello" {
		t.Errorf("unexpected input item: %v", input[0])
	}
}

// A block-array system prompt must be flattened into the instructions string.
func TestTranslateAnthropicToResponses_SystemBlocks(t *testing.T) {
	req := &MessageRequest{
		Model:  "GPT-5.6 Sol",
		System: json.RawMessage(`[{"type":"text","text":"first"},{"type":"text","text":"second"}]`),
	}
	body := TranslateAnthropicToResponses(req, solModel())
	got, _ := body["instructions"].(string)
	if !strings.Contains(got, "first") || !strings.Contains(got, "second") {
		t.Errorf("instructions = %q, want both blocks", got)
	}
}

// tool_use / tool_result blocks become top-level function_call and
// function_call_output items, not nested content.
func TestTranslateAnthropicToResponses_ToolRoundTrip(t *testing.T) {
	req := &MessageRequest{
		Model:     "GPT-5.6 Sol",
		MaxTokens: 64,
		Messages: []MessageParam{
			{Role: "user", Content: json.RawMessage(`"weather in Beijing?"`)},
			{Role: "assistant", Content: json.RawMessage(
				`[{"type":"tool_use","id":"call_1","name":"get_weather","input":{"city":"Beijing"}}]`)},
			{Role: "user", Content: json.RawMessage(
				`[{"type":"tool_result","tool_use_id":"call_1","content":"sunny"}]`)},
		},
		Tools: []Tool{{
			Name: "get_weather", Description: "Get weather",
			InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
		}},
	}
	body := TranslateAnthropicToResponses(req, solModel())

	input := body["input"].([]map[string]interface{})
	var sawCall, sawOutput bool
	for _, item := range input {
		switch item["type"] {
		case "function_call":
			sawCall = true
			if item["call_id"] != "call_1" || item["name"] != "get_weather" {
				t.Errorf("unexpected function_call: %v", item)
			}
			if item["arguments"] != `{"city":"Beijing"}` {
				t.Errorf("arguments = %v, want raw JSON string", item["arguments"])
			}
		case "function_call_output":
			sawOutput = true
			if item["call_id"] != "call_1" || item["output"] != "sunny" {
				t.Errorf("unexpected function_call_output: %v", item)
			}
		}
	}
	if !sawCall || !sawOutput {
		t.Errorf("expected both function_call and function_call_output, got %v", input)
	}

	// Responses tools are flat, not nested under "function".
	tools := body["tools"].([]map[string]interface{})
	if len(tools) != 1 || tools[0]["name"] != "get_weather" || tools[0]["type"] != "function" {
		t.Errorf("unexpected tools: %v", tools)
	}
	if _, nested := tools[0]["function"]; nested {
		t.Error("Responses tools must be flat, not nested under 'function'")
	}
}

// TestBuildResponsesInput_DanglingFunctionCallDowngraded: a tool_use with no
// matching tool_result would emit a function_call with no function_call_output
// — a sequence strict Responses implementations reject. It must be downgraded
// to a text note instead.
func TestBuildResponsesInput_DanglingFunctionCallDowngraded(t *testing.T) {
	req := &MessageRequest{
		Model:     "GPT-5.6 Sol",
		MaxTokens: 64,
		Messages: []MessageParam{
			{Role: "user", Content: json.RawMessage(`"start"`)},
			// Assistant calls a tool; the user interrupted before the result.
			{Role: "assistant", Content: json.RawMessage(
				`[{"type":"tool_use","id":"call_dangling","name":"Bash","input":{"command":"ls"}}]`)},
			{Role: "user", Content: json.RawMessage(`"wait, stop that"`)},
		},
	}
	body := TranslateAnthropicToResponses(req, solModel())
	input := body["input"].([]map[string]interface{})

	for _, item := range input {
		if item["type"] == "function_call" {
			t.Fatalf("dangling function_call must be downgraded, got: %v", input)
		}
	}
	// The omitted call should surface as a text note mentioning the tool name.
	var sawNote bool
	for _, item := range input {
		if content, ok := item["content"].([]map[string]interface{}); ok {
			for _, p := range content {
				if s, ok := p["text"].(string); ok && strings.Contains(s, "Bash") {
					sawNote = true
				}
			}
		}
	}
	if !sawNote {
		t.Errorf("expected a text note about the omitted call, got %v", input)
	}
}

// Anthropic server-side builtins have no Responses equivalent and must be dropped
// rather than sent as malformed function tools.
func TestConvertToolsToResponses_DropsBuiltins(t *testing.T) {
	tools := []Tool{
		{Type: "web_search_20250305", Name: "web_search"},
		{Name: "real_tool", InputSchema: json.RawMessage(`{"type":"object"}`)},
	}
	got := convertToolsToResponses(tools)
	if len(got) != 1 || got[0]["name"] != "real_tool" {
		t.Errorf("expected only the function tool, got %v", got)
	}
}

func TestConvertToolChoiceToResponses(t *testing.T) {
	cases := map[string]interface{}{
		`{"type":"any"}`:                    "required",
		`{"type":"auto"}`:                   "auto",
		`{"type":"none"}`:                   "none",
		`{"type":"tool","name":"get_time"}`: map[string]interface{}{"type": "function", "name": "get_time"},
	}
	for raw, want := range cases {
		got, _ := convertToolChoiceToResponses(json.RawMessage(raw))
		gotJSON, _ := json.Marshal(got)
		wantJSON, _ := json.Marshal(want)
		if string(gotJSON) != string(wantJSON) {
			t.Errorf("tool_choice %s => %s, want %s", raw, gotJSON, wantJSON)
		}
	}
}

func TestTranslateResponsesToAnthropic_Text(t *testing.T) {
	raw := map[string]interface{}{
		"id":     "resp_1",
		"status": "completed",
		"output": []interface{}{
			map[string]interface{}{
				"type": "message",
				"content": []interface{}{
					map[string]interface{}{"type": "output_text", "text": "OK"},
				},
			},
		},
		"usage": map[string]interface{}{"input_tokens": 11, "output_tokens": 5},
	}
	resp, err := TranslateResponsesToAnthropic(raw, "GPT-5.6 Sol")
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != "text" || resp.Content[0].Text != "OK" {
		t.Errorf("unexpected content: %+v", resp.Content)
	}
	if resp.Model != "GPT-5.6 Sol" {
		t.Errorf("model = %q, want the display label", resp.Model)
	}
	if *resp.StopReason != "end_turn" {
		t.Errorf("stop_reason = %q, want end_turn", *resp.StopReason)
	}
	if resp.Usage.InputTokens != 11 || resp.Usage.OutputTokens != 5 {
		t.Errorf("usage = %+v, want 11/5", resp.Usage)
	}
}

// A function_call in the output must surface as a tool_use block with
// stop_reason=tool_use, or Anthropic clients will not execute the tool.
func TestTranslateResponsesToAnthropic_ToolCall(t *testing.T) {
	raw := map[string]interface{}{
		"id": "resp_2",
		"output": []interface{}{
			map[string]interface{}{
				"type": "function_call", "name": "get_weather",
				"call_id": "call_9", "arguments": `{"city":"Beijing"}`,
			},
		},
	}
	resp, err := TranslateResponsesToAnthropic(raw, "GPT-5.6 Sol")
	if err != nil {
		t.Fatal(err)
	}
	if *resp.StopReason != "tool_use" {
		t.Errorf("stop_reason = %q, want tool_use", *resp.StopReason)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != "tool_use" {
		t.Fatalf("unexpected content: %+v", resp.Content)
	}
	blk := resp.Content[0]
	if blk.ID != "call_9" || blk.Name != "get_weather" {
		t.Errorf("unexpected tool_use block: %+v", blk)
	}
	input, ok := blk.Input.(map[string]interface{})
	if !ok || input["city"] != "Beijing" {
		t.Errorf("input = %v, want parsed arguments", blk.Input)
	}
}

// Hitting the output cap must map to stop_reason=max_tokens.
func TestTranslateResponsesToAnthropic_MaxTokens(t *testing.T) {
	raw := map[string]interface{}{
		"id":                 "resp_3",
		"output":             []interface{}{},
		"incomplete_details": map[string]interface{}{"reason": "max_output_tokens"},
	}
	resp, err := TranslateResponsesToAnthropic(raw, "GPT-5.6 Sol")
	if err != nil {
		t.Fatal(err)
	}
	if *resp.StopReason != "max_tokens" {
		t.Errorf("stop_reason = %q, want max_tokens", *resp.StopReason)
	}
}

// --- stream translation ---

// runResponsesStream feeds raw Responses SSE payloads through the translation
// state machine and returns the Anthropic SSE bytes it emitted.
func runResponsesStream(events []string) string {
	rec := httptest.NewRecorder()
	st := &responsesStreamState{
		w: rec, flusher: rec, msgID: "msg_test", displayModel: "GPT-5.6 Sol",
		toolsByItem: make(map[string]*anthropicToolStream), toolsByCall: make(map[string]*anthropicToolStream),
	}
	st.emitMessageStart(&MessageRequest{})
	for _, raw := range events {
		var ev responsesStreamEvent
		if err := json.Unmarshal([]byte(raw), &ev); err != nil {
			continue
		}
		st.handleEvent(&ev)
	}
	st.finish("end_turn")
	return rec.Body.String()
}

// A text stream must produce the Anthropic event sequence clients expect.
func TestResponsesStream_TextSequence(t *testing.T) {
	out := runResponsesStream([]string{
		`{"type":"response.output_text.delta","delta":"Hello"}`,
		`{"type":"response.output_text.delta","delta":" world"}`,
		`{"type":"response.completed","response":{"usage":{"input_tokens":7,"output_tokens":3}}}`,
	})

	for _, want := range []string{
		"event: message_start", "event: content_block_start",
		"event: content_block_delta", "event: content_block_stop",
		"event: message_delta", "event: message_stop",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in stream:\n%s", want, out)
		}
	}
	if !strings.Contains(out, `"text":"Hello"`) || !strings.Contains(out, `"text":" world"`) {
		t.Errorf("missing text deltas:\n%s", out)
	}
	if strings.LastIndex(out, "event: message_stop") < strings.LastIndex(out, "event: content_block_stop") {
		t.Error("message_stop must come after content_block_stop")
	}
}

// A tool call stream must emit a tool_use block with input_json_delta chunks.
func TestResponsesStream_ToolCallSequence(t *testing.T) {
	out := runResponsesStream([]string{
		`{"type":"response.output_item.added","item":{"type":"function_call","name":"get_weather","call_id":"call_3"}}`,
		`{"type":"response.function_call_arguments.delta","delta":"{\"city\":"}`,
		`{"type":"response.function_call_arguments.delta","delta":"\"Beijing\"}"}`,
		`{"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_3"}}`,
		`{"type":"response.completed","response":{"usage":{"input_tokens":50,"output_tokens":20}}}`,
	})

	if !strings.Contains(out, `"type":"tool_use"`) {
		t.Errorf("expected a tool_use content block:\n%s", out)
	}
	if !strings.Contains(out, `"id":"call_3"`) || !strings.Contains(out, `"name":"get_weather"`) {
		t.Errorf("expected the tool identity to be carried through:\n%s", out)
	}
	if !strings.Contains(out, "input_json_delta") {
		t.Errorf("expected input_json_delta chunks:\n%s", out)
	}
	if !strings.Contains(out, `"stop_reason":"tool_use"`) {
		t.Errorf("expected stop_reason=tool_use:\n%s", out)
	}
}

// Interleaved text and a tool call must occupy distinct block indices, otherwise
// clients merge them into one block.
func TestResponsesStream_TextThenToolUsesDistinctIndices(t *testing.T) {
	out := runResponsesStream([]string{
		`{"type":"response.output_text.delta","delta":"thinking"}`,
		`{"type":"response.output_item.added","item":{"type":"function_call","name":"f","call_id":"c1"}}`,
		`{"type":"response.function_call_arguments.delta","delta":"{}"}`,
		`{"type":"response.completed","response":{}}`,
	})
	if !strings.Contains(out, `"index":0`) || !strings.Contains(out, `"index":1`) {
		t.Errorf("expected separate block indices:\n%s", out)
	}
}

// finish() must be idempotent so a stream error after normal completion cannot
// emit a second message_stop.
func TestResponsesStream_FinishIsIdempotent(t *testing.T) {
	rec := httptest.NewRecorder()
	st := &responsesStreamState{w: rec, flusher: rec, msgID: "m", displayModel: "GPT-5.6 Sol"}
	st.finish("end_turn")
	st.finish("end_turn")
	if n := strings.Count(rec.Body.String(), "event: message_stop"); n != 1 {
		t.Errorf("message_stop emitted %d times, want 1", n)
	}
}

// An output cap during streaming must be reported as max_tokens.
func TestResponsesStream_IncompleteMapsToMaxTokens(t *testing.T) {
	out := runResponsesStream([]string{
		`{"type":"response.output_text.delta","delta":"partial"}`,
		`{"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"}}}`,
	})
	if !strings.Contains(out, `"stop_reason":"max_tokens"`) {
		t.Errorf("expected stop_reason=max_tokens:\n%s", out)
	}
}

func runGPTResponsesHandlerStream(t *testing.T, sse string) (string, store.RequestOutcome) {
	t.Helper()
	joycode.DefaultRegistry.Seed([]joycode.ModelInfo{solModel()})
	t.Cleanup(func() { joycode.DefaultRegistry.Seed(joycode.StaticModels()) })
	client := joycode.NewClient("pt-test", "user-test")
	client.SetHTTPClient(&http.Client{Transport: rtFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(sse))}, nil
	})})
	handler := NewHandler(client, nil)
	mux := http.NewServeMux()
	handler.RegisterRoutes(mux)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"GPT-5.6 Sol","stream":true,"max_tokens":64,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Content-Type", "application/json")
	req = store.InitTokenUsage(req)
	req = store.InitRequestOutcome(req)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	return rec.Body.String(), store.GetRequestOutcome(req)
}

func TestResponsesHandlerStream_PrematureEOFSurfacesError(t *testing.T) {
	out, outcome := runGPTResponsesHandlerStream(t,
		`data: {"type":"response.output_text.delta","delta":"partial"}`+"\n\n")
	if !strings.Contains(out, `"type":"error"`) || !strings.Contains(out, "结束标记前断开") {
		t.Errorf("missing Anthropic stream error:\n%s", out)
	}
	if strings.Contains(out, "event: message_stop") || strings.Contains(out, `"stop_reason":"end_turn"`) {
		t.Errorf("premature EOF must not fake message_stop/end_turn:\n%s", out)
	}
	if outcome.StatusCode != http.StatusBadGateway || !strings.Contains(outcome.Error, "stream_truncated") {
		t.Errorf("outcome = %+v", outcome)
	}
}

func TestResponsesHandlerStream_FailedEventSurfacesError(t *testing.T) {
	out, outcome := runGPTResponsesHandlerStream(t,
		`data: {"type":"response.failed","response":{"error":{"message":"model unavailable"}}}`+"\n\n")
	if !strings.Contains(out, `"type":"error"`) || !strings.Contains(out, "model unavailable") {
		t.Errorf("missing failed event error:\n%s", out)
	}
	if strings.Contains(out, "event: message_stop") {
		t.Errorf("failed event must not emit message_stop:\n%s", out)
	}
	if outcome.StatusCode != http.StatusBadGateway || !strings.Contains(outcome.Error, "upstream_error") {
		t.Errorf("outcome = %+v", outcome)
	}
}

func TestTranslateAnthropicToResponses_ParallelToolChoice(t *testing.T) {
	for _, tc := range []struct {
		raw  string
		want bool
	}{
		{`{"type":"auto","disable_parallel_tool_use":true}`, false},
		{`{"type":"any","disable_parallel_tool_use":false}`, true},
	} {
		req := &MessageRequest{
			Messages:   []MessageParam{{Role: "user", Content: json.RawMessage(`"hi"`)}},
			Tools:      []Tool{{Name: "f", InputSchema: json.RawMessage(`{"type":"object"}`)}},
			ToolChoice: json.RawMessage(tc.raw),
		}
		body := TranslateAnthropicToResponses(req, solModel())
		if got, ok := body["parallel_tool_calls"].(bool); !ok || got != tc.want {
			t.Errorf("tool choice %s: parallel_tool_calls=%v", tc.raw, body["parallel_tool_calls"])
		}
	}
	req := &MessageRequest{
		Messages: []MessageParam{{Role: "user", Content: json.RawMessage(`"hi"`)}},
		Tools:    []Tool{{Name: "f"}}, ToolChoice: json.RawMessage(`{"type":"auto"}`),
	}
	if _, exists := TranslateAnthropicToResponses(req, solModel())["parallel_tool_calls"]; exists {
		t.Error("parallel_tool_calls must be omitted when the client did not specify it")
	}
}

func TestResponsesStream_InterleavedParallelTools(t *testing.T) {
	out := runResponsesStream([]string{
		`{"type":"response.output_item.added","item":{"id":"item_a","type":"function_call","name":"alpha","call_id":"call_a"}}`,
		`{"type":"response.output_item.added","item":{"id":"item_b","type":"function_call","name":"beta","call_id":"call_b"}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"item_a","delta":"{\"a\":"}`,
		`{"type":"response.function_call_arguments.delta","item_id":"item_b","delta":"{\"b\":2}"}`,
		`{"type":"response.function_call_arguments.delta","call_id":"call_a","delta":"1}"}`,
		`{"type":"response.output_item.done","item":{"id":"item_b","type":"function_call","call_id":"call_b"}}`,
		`{"type":"response.output_item.done","item":{"id":"item_a","type":"function_call","call_id":"call_a"}}`,
		`{"type":"response.completed","response":{}}`,
	})
	if !strings.Contains(out, `"index":0`) || !strings.Contains(out, `"index":1`) {
		t.Fatalf("missing distinct tool indices:\n%s", out)
	}
	if !strings.Contains(out, `"id":"call_a"`) || !strings.Contains(out, `"id":"call_b"`) {
		t.Fatalf("missing tool identities:\n%s", out)
	}
	if strings.Count(out, `"type":"content_block_stop"`) != 2 {
		t.Errorf("each tool must stop exactly once:\n%s", out)
	}
	// A's two deltas must stay on index 0; B's delta stays on index 1.
	if !strings.Contains(out, `"index":0,"delta":{"type":"input_json_delta","partial_json":"{\"a\":"}`) ||
		!strings.Contains(out, `"index":0,"delta":{"type":"input_json_delta","partial_json":"1}"}`) ||
		!strings.Contains(out, `"index":1,"delta":{"type":"input_json_delta","partial_json":"{\"b\":2}"}`) {
		t.Errorf("tool argument deltas were routed incorrectly:\n%s", out)
	}
}

func TestTranslateResponsesToAnthropic_RejectsFailedAndUnknownIncomplete(t *testing.T) {
	cases := []map[string]interface{}{
		{"status": "failed", "error": nil, "output": []interface{}{}},
		{"status": "failed", "error": map[string]interface{}{"message": "model unavailable"}},
		{"status": "incomplete", "incomplete_details": map[string]interface{}{"reason": "content_filter"}, "output": []interface{}{map[string]interface{}{"type": "message", "content": []interface{}{map[string]interface{}{"type": "output_text", "text": "partial"}}}}},
	}
	for _, raw := range cases {
		if _, err := TranslateResponsesToAnthropic(raw, "GPT-5.6 Sol"); err == nil {
			t.Errorf("expected error for %#v", raw)
		}
	}
}

func TestTranslateAnthropicToResponses_Effort(t *testing.T) {
	for requested, want := range map[string]string{"none": "none", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max", "extra": "xhigh"} {
		req := &MessageRequest{Messages: []MessageParam{{Role: "user", Content: json.RawMessage(`"hi"`)}}, OutputConfig: &OutputConfig{Effort: requested}}
		body := TranslateAnthropicToResponses(req, solModel())
		reasoning, ok := body["reasoning"].(map[string]interface{})
		if !ok || reasoning["effort"] != want {
			t.Errorf("effort %s => %#v", requested, body["reasoning"])
		}
	}
	req := &MessageRequest{Messages: []MessageParam{{Role: "user", Content: json.RawMessage(`"hi"`)}}}
	if _, exists := TranslateAnthropicToResponses(req, solModel())["reasoning"]; exists {
		t.Fatal("reasoning emitted without effort")
	}
}
