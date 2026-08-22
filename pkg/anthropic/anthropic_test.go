package anthropic

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
)

// seedRegistry pins the model registry to a known list so resolution tests do
// not depend on what upstream currently offers.
func seedRegistry(t *testing.T) {
	t.Helper()
	joycode.DefaultRegistry.Seed([]joycode.ModelInfo{
		{Label: "JoyAI-Code-1.5", ChatAPIModel: "JoyAI-Code-1.5"},
		{Label: "GLM-5.1", ChatAPIModel: "GLM-5.1"},
		{Label: "Kimi-K2.6", ChatAPIModel: "Kimi-K2.6"},
		{Label: "GPT-5.6 Sol", ChatAPIModel: "GPT-5.6 Sol",
			ExtJSON: map[string]interface{}{"adapterType": joycode.AdapterOpenAIResponse}},
		{Label: "Claude-Opus-4.8", ChatAPIModel: "Claude-Opus-4.8-hq", SortOrder: 51,
			ExtJSON: map[string]interface{}{"adapterType": joycode.AdapterAnthropic}},
		{Label: "Claude-Opus-4.7", ChatAPIModel: "Claude-Opus-4.7-hq", SortOrder: 50,
			ExtJSON: map[string]interface{}{"adapterType": joycode.AdapterAnthropic}},
	})
}

func TestResolveModel(t *testing.T) {
	seedRegistry(t)
	tests := []struct {
		name           string
		model          string
		accountDefault string
		systemDefault  string
		expected       string
	}{
		{"known model passes through", "GLM-5.1", "", "", "GLM-5.1"},
		{"unknown model falls back to default", "some-unknown-model", "", "", joycode.DefaultModel},
		{"account default overrides for unknown model", "nope", "Kimi-K2.6", "GLM-5.1", "Kimi-K2.6"},
		{"system default used when no account default", "unknown-model", "", "GLM-5.1", "GLM-5.1"},
		// Resolution must yield chatApiModel, not the label: upstream accepts
		// the label but silently serves it with a different, weaker model.
		{"claude label resolves to -hq api name", "Claude-Opus-4.7", "", "", "Claude-Opus-4.7-hq"},
		// Model names containing spaces must survive dash/space normalisation.
		{"space and dash forms both resolve", "gpt-5.6-sol", "", "", "GPT-5.6 Sol"},
		{"exact spaced form resolves", "GPT-5.6 Sol", "", "", "GPT-5.6 Sol"},
	}
	for _, tt := range tests {
		got := resolveModel(tt.model, tt.accountDefault, tt.systemDefault)
		if got != tt.expected {
			t.Errorf("%s: resolveModel(%q, %q, %q) = %q, want %q",
				tt.name, tt.model, tt.accountDefault, tt.systemDefault, got, tt.expected)
		}
	}
}

// The native Anthropic path must send chatApiModel and must not be pinned to a
// hardcoded version, so newly released Claude models are reachable.
func TestResolveNativeAnthropicModel(t *testing.T) {
	seedRegistry(t)
	if got := resolveNativeAnthropicModel("Claude-Opus-4.7", "", ""); got != "Claude-Opus-4.7-hq" {
		t.Errorf("expected Claude-Opus-4.7-hq, got %q", got)
	}
	if got := resolveNativeAnthropicModel("Claude-Opus-4.8", "", ""); got != "Claude-Opus-4.8-hq" {
		t.Errorf("expected Claude-Opus-4.8-hq, got %q", got)
	}
	// An unknown Claude alias should land on the newest Anthropic model rather
	// than a hardcoded one.
	if got := resolveNativeAnthropicModel("claude-sonnet-4-5-20250929", "", ""); got != "Claude-Opus-4.8-hq" {
		t.Errorf("expected newest Claude (Claude-Opus-4.8-hq), got %q", got)
	}
}

func TestTranslateRequest(t *testing.T) {
	temp := 0.7
	req := &MessageRequest{
		Model:     "claude-sonnet-4-20250514",
		MaxTokens: 1024,
		Messages: []MessageParam{
			{Role: "user", Content: json.RawMessage(`"Hello"`)},
		},
		Temperature: &temp,
	}
	body := TranslateRequest(req, "", "")

	if body["model"] != joycode.DefaultModel {
		t.Errorf("model = %v, want %s", body["model"], joycode.DefaultModel)
	}
	if body["max_tokens"] != 1024 {
		t.Errorf("max_tokens = %v, want 1024", body["max_tokens"])
	}
	if body["temperature"] != 0.7 {
		t.Errorf("temperature = %v, want 0.7", body["temperature"])
	}
	msgs, ok := body["messages"].([]map[string]interface{})
	if !ok {
		t.Fatalf("messages type = %T, want []map[string]interface{}", body["messages"])
	}
	if len(msgs) != 1 || msgs[0]["role"] != "user" {
		t.Errorf("messages = %v, unexpected", msgs)
	}
}

func TestTranslateRequestWithSystem(t *testing.T) {
	req := &MessageRequest{
		Model:     "claude-sonnet-4",
		MaxTokens: 2048,
		System:    json.RawMessage(`"You are a helpful assistant"`),
		Messages: []MessageParam{
			{Role: "user", Content: json.RawMessage(`"Hi"`)},
		},
	}
	body := TranslateRequest(req, "", "")
	msgs := body["messages"].([]map[string]interface{})
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages (system + user), got %d", len(msgs))
	}
	if msgs[0]["role"] != "system" {
		t.Errorf("first message role = %q, want system", msgs[0]["role"])
	}
}

// TestBuildMessages_DanglingToolUseDowngraded: a tool_use with no matching
// tool_result anywhere in the retained conversation would produce an assistant
// message with tool_calls but no tool response — a protocol violation OpenAI
// rejects. Such calls must be downgraded to a text note instead.
func TestBuildMessages_DanglingToolUseDowngraded(t *testing.T) {
	req := &MessageRequest{
		Model:     "claude-sonnet-4",
		MaxTokens: 1024,
		Messages: []MessageParam{
			{Role: "user", Content: json.RawMessage(`"start"`)},
			// Assistant calls a tool but the user interrupted before the result.
			{Role: "assistant", Content: json.RawMessage(`[{"type":"tool_use","id":"call_a3","name":"Bash","input":{"command":"ls"}}]`)},
			{Role: "user", Content: json.RawMessage(`"wait, stop that"`)},
		},
	}
	body := TranslateRequest(req, "", "")
	msgs := body["messages"].([]map[string]interface{})

	// No message may carry tool_calls (the call a3 has no result).
	for _, m := range msgs {
		if tc, ok := m["tool_calls"]; ok && tc != nil {
			t.Fatalf("dangling tool_use must be downgraded, got tool_calls in %v", m)
		}
	}
	// The assistant turn should now be a text note mentioning the omitted call.
	var assistantText string
	for _, m := range msgs {
		if m["role"] == "assistant" {
			if s, ok := m["content"].(string); ok {
				assistantText = s
			}
		}
	}
	if !strings.Contains(assistantText, "Bash") {
		t.Errorf("expected a text note about the omitted tool call, got %q", assistantText)
	}
}

// TestBuildMessages_PairedToolUseKept: a tool_use WITH a matching tool_result
// must still produce a real tool_calls entry and a tool message (regression
// guard — the dangling downgrade must not eat legitimate exchanges).
func TestBuildMessages_PairedToolUseKept(t *testing.T) {
	req := &MessageRequest{
		Model:     "claude-sonnet-4",
		MaxTokens: 1024,
		Messages: []MessageParam{
			{Role: "user", Content: json.RawMessage(`"list files"`)},
			{Role: "assistant", Content: json.RawMessage(`[{"type":"tool_use","id":"call_ok","name":"Bash","input":{"command":"ls"}}]`)},
			{Role: "user", Content: json.RawMessage(`[{"type":"tool_result","tool_use_id":"call_ok","content":"a.go b.go"}]`)},
		},
	}
	body := TranslateRequest(req, "", "")
	msgs := body["messages"].([]map[string]interface{})

	var sawToolCall, sawToolMsg bool
	for _, m := range msgs {
		if m["role"] == "assistant" {
			if tc, ok := m["tool_calls"].([]interface{}); ok && len(tc) == 1 {
				sawToolCall = true
			}
		}
		if m["role"] == "tool" && m["tool_call_id"] == "call_ok" {
			sawToolMsg = true
		}
	}
	if !sawToolCall || !sawToolMsg {
		t.Errorf("paired exchange must be kept: toolCall=%v toolMsg=%v\n%v", sawToolCall, sawToolMsg, msgs)
	}
}


func TestTranslateResponse(t *testing.T) {
	jcResp := map[string]interface{}{
		"choices": []interface{}{
			map[string]interface{}{
				"message": map[string]interface{}{
					"content": "Hi! How can I help?",
				},
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     float64(10),
			"completion_tokens": float64(5),
		},
	}
	resp := TranslateResponse(jcResp, "claude-sonnet-4-20250514")

	if resp.Type != "message" {
		t.Errorf("Type = %q, want message", resp.Type)
	}
	if resp.Role != "assistant" {
		t.Errorf("Role = %q, want assistant", resp.Role)
	}
	if resp.Model != "claude-sonnet-4-20250514" {
		t.Errorf("Model = %q, want claude-sonnet-4-20250514", resp.Model)
	}
	if resp.StopReason == nil || *resp.StopReason != "end_turn" {
		t.Errorf("StopReason = %v, want end_turn", resp.StopReason)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != "text" {
		t.Errorf("Content = %v, unexpected", resp.Content)
	}
	if resp.Content[0].Text != "Hi! How can I help?" {
		t.Errorf("Text = %q, want 'Hi! How can I help?'", resp.Content[0].Text)
	}
	if resp.Usage.InputTokens != 10 {
		t.Errorf("InputTokens = %d, want 10", resp.Usage.InputTokens)
	}
	if resp.Usage.OutputTokens != 5 {
		t.Errorf("OutputTokens = %d, want 5", resp.Usage.OutputTokens)
	}
	if !strings.HasPrefix(resp.ID, "msg_") {
		t.Errorf("ID = %q, want msg_ prefix", resp.ID)
	}
}

func TestParseStreamDelta(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{`data: {"choices":[{"delta":{"content":"Hello"}}]}`, "Hello"},
		{`data: [DONE]`, ""},
		{`data: {"choices":[]}`, ""},
		{``, ""},
		{`data: {"choices":[{"delta":{"content":" world"}}]}`, " world"},
	}
	for _, tt := range tests {
		got := ParseStreamDelta(tt.input)
		if got != tt.expected {
			t.Errorf("ParseStreamDelta(%q) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}

func TestParseContent(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{`"plain text"`, "plain text"},
		{`[{"type":"text","text":"hello"},{"type":"text","text":"world"}]`, "hello\nworld"},
		{`[{"type":"image","text":"skip me"}]`, ""},
		{``, ""},
	}
	for _, tt := range tests {
		got := parseContent(json.RawMessage(tt.input))
		if got != tt.expected {
			t.Errorf("parseContent(%q) = %q, want %q", tt.input, got, tt.expected)
		}
	}
}

func TestHandlerMessagesEndpoint(t *testing.T) {
	tests := []struct {
		name       string
		method     string
		body       string
		wantStatus int
	}{
		{"OPTIONS preflight", "OPTIONS", "", 200},
		{"GET rejected", "GET", "", 405},
		{"invalid JSON", "POST", `{bad}`, 400},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var body strings.Reader
			if tt.body != "" {
				body = *strings.NewReader(tt.body)
			}
			req := httptest.NewRequest(tt.method, "/v1/messages", &body)
			if tt.method == "POST" {
				req.Header.Set("Content-Type", "application/json")
			}
			w := httptest.NewRecorder()
			h := &Handler{Client: nil}
			h.handleMessages(w, req)
			if w.Code != tt.wantStatus {
				t.Errorf("status = %d, want %d; body = %s", w.Code, tt.wantStatus, w.Body.String())
			}
		})
	}
}

func TestAnthropicResponseFormat(t *testing.T) {
	jcResp := map[string]interface{}{
		"choices": []interface{}{
			map[string]interface{}{
				"message": map[string]interface{}{
					"content": "Test response",
				},
			},
		},
		"usage": map[string]interface{}{
			"prompt_tokens":     float64(100),
			"completion_tokens": float64(50),
		},
	}
	resp := TranslateResponse(jcResp, "claude-sonnet-4-20250514")

	// Verify JSON serialization matches Anthropic format
	b, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}

	var parsed map[string]interface{}
	json.Unmarshal(b, &parsed)

	// Check all required Anthropic response fields
	required := []string{"id", "type", "role", "content", "model", "stop_reason", "usage"}
	for _, field := range required {
		if _, ok := parsed[field]; !ok {
			t.Errorf("missing required field: %s in response: %s", field, string(b))
		}
	}

	if parsed["type"] != "message" {
		t.Errorf("type = %v, want message", parsed["type"])
	}
	if parsed["role"] != "assistant" {
		t.Errorf("role = %v, want assistant", parsed["role"])
	}
}

func TestCountTokensEndpoint(t *testing.T) {
	h := NewHandler(nil, nil)
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)
	body := `{"model":"GPT-5.6 Sol","system":"system prompt","messages":[{"role":"user","content":"hello world"}],"tools":[{"name":"lookup","description":"lookup values","input_schema":{"type":"object","properties":{"q":{"type":"string"}}}}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(body))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var result map[string]int
	if err := json.Unmarshal(rec.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if result["input_tokens"] <= 0 {
		t.Fatalf("input_tokens=%d", result["input_tokens"])
	}
}

func TestEstimateInputTokensIncludesToolsAndImages(t *testing.T) {
	base := &MessageRequest{Messages: []MessageParam{{Role: "user", Content: json.RawMessage(`"hello"`)}}}
	baseCount := EstimateInputTokens(base)
	withTools := &MessageRequest{Messages: base.Messages, Tools: []Tool{{Name: "f", Description: "long tool description", InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`)}}}
	if EstimateInputTokens(withTools) <= baseCount {
		t.Fatal("tools did not increase estimate")
	}
	withImage := &MessageRequest{Messages: []MessageParam{{Role: "user", Content: json.RawMessage(`[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + strings.Repeat("A", 10000) + `"}}]`)}}}
	count := EstimateInputTokens(withImage)
	if count < 1000 || count > 5000 {
		t.Fatalf("image estimate=%d", count)
	}
}
