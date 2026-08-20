package openai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

// collectSSE parses all data: payloads from an SSE body into raw events.
func collectSSE(body string) []map[string]interface{} {
	events := []map[string]interface{}{}
	for _, block := range strings.Split(body, "\n\n") {
		for _, line := range strings.Split(block, "\n") {
			line = strings.TrimSpace(line)
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			payload := strings.TrimPrefix(line, "data: ")
			if payload == "[DONE]" || !strings.HasPrefix(payload, "{") {
				continue
			}
			var ev map[string]interface{}
			if err := json.Unmarshal([]byte(payload), &ev); err == nil {
				events = append(events, ev)
			}
		}
	}
	return events
}

// TestChatFallbackStream_OutputIndexesAreUnique reproduces the Codex
// "stops after the first tool round" bug: the reasoning item and the
// function_call item were both assigned output_index 1, so Codex could not
// reconcile the streamed function_call and silently dropped it.
func TestChatFallbackStream_OutputIndexesAreUnique(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{kimiModel()})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		chunks := []string{
			`{"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"thinking"}}]}`,
			`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"shell_0","type":"function","function":{"name":"shell","arguments":""}}]}}]}`,
			`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"command\":[\"ls\"]}"}}]}}]}`,
			`{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		}
		for _, c := range chunks {
			io.WriteString(w, "data: "+c+"\n\n")
			flusher.Flush()
		}
		io.WriteString(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer backend.Close()

	server := NewServer(newMockClient(backend), nil)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"Kimi-K3","input":"list files","stream":true}`))
	req = store.InitTokenUsage(req)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}

	seen := map[float64]string{}
	for _, ev := range collectSSE(rec.Body.String()) {
		evType, _ := ev["type"].(string)
		if evType != "response.output_item.added" {
			continue
		}
		idx, ok := ev["output_index"].(float64)
		if !ok {
			t.Fatalf("output_item.added without numeric output_index: %v", ev)
		}
		item, _ := ev["item"].(map[string]interface{})
		itemType, _ := item["type"].(string)
		if prev, dup := seen[idx]; dup {
			t.Fatalf("output_index %v used by both %q and %q", idx, prev, itemType)
		}
		seen[idx] = itemType
	}
	if len(seen) != 2 {
		t.Fatalf("expected reasoning + function_call items, got %v", seen)
	}

	// The terminal response.completed must embed both items, in index order.
	var completed map[string]interface{}
	for _, ev := range collectSSE(rec.Body.String()) {
		if ev["type"] == "response.completed" {
			completed, _ = ev["response"].(map[string]interface{})
		}
	}
	if completed == nil {
		t.Fatalf("no response.completed in stream:\n%s", rec.Body.String())
	}
	output, _ := completed["output"].([]interface{})
	if len(output) != 2 {
		t.Fatalf("completed output items = %v", output)
	}
	first, _ := output[0].(map[string]interface{})
	second, _ := output[1].(map[string]interface{})
	if first["type"] != "reasoning" || second["type"] != "function_call" {
		t.Errorf("completed output order = %v, %v", first["type"], second["type"])
	}
	if args, _ := second["arguments"].(string); args != `{"command":["ls"]}` {
		t.Errorf("function_call arguments = %q", args)
	}
}

// TestChatFallbackStream_TrailingUsageChunk reproduces lost token stats: with
// stream_options.include_usage the upstream sends usage in a separate chunk
// after the finish chunk. Completing at the finish chunk dropped it, so the
// response.completed event reported zero tokens.
func TestChatFallbackStream_TrailingUsageChunk(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{kimiModel()})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		chunks := []string{
			`{"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"}}]}`,
			`{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":null}`,
			`{"id":"c1","choices":[],"usage":{"prompt_tokens":17,"completion_tokens":3,"total_tokens":20}}`,
		}
		for _, c := range chunks {
			io.WriteString(w, "data: "+c+"\n\n")
			flusher.Flush()
		}
		io.WriteString(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer backend.Close()

	server := NewServer(newMockClient(backend), nil)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"Kimi-K3","input":"hello","stream":true}`))
	req = store.InitTokenUsage(req)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}

	var completed map[string]interface{}
	for _, ev := range collectSSE(rec.Body.String()) {
		if ev["type"] == "response.completed" {
			completed, _ = ev["response"].(map[string]interface{})
		}
	}
	if completed == nil {
		t.Fatalf("no response.completed in stream:\n%s", rec.Body.String())
	}
	usage, _ := completed["usage"].(map[string]interface{})
	if in, _ := usage["input_tokens"].(float64); in != 17 {
		t.Errorf("completed input_tokens = %v", usage["input_tokens"])
	}
	if out, _ := usage["output_tokens"].(float64); out != 3 {
		t.Errorf("completed output_tokens = %v", usage["output_tokens"])
	}
	if inTk, outTk := store.GetTokenUsage(req); inTk != 17 || outTk != 3 {
		t.Errorf("stored usage = %d/%d", inTk, outTk)
	}
}

// TestTranslateResponsesReqToChat_MergesAssistantTurn covers the
// multi-round tool loop: Codex replays the previous turn as separate items
// (assistant message, function_call, function_call_output). The chat
// translation must keep an assistant turn's text and tool_calls on a single
// message — consecutive assistant messages break Kimi and stall the loop.
func TestTranslateResponsesReqToChat_MergesAssistantTurn(t *testing.T) {
	body := map[string]interface{}{
		"model": "Kimi-K3",
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "list files"},
			map[string]interface{}{
				"type": "message", "role": "assistant",
				"content": []interface{}{
					map[string]interface{}{"type": "output_text", "text": "Let me check."},
				},
			},
			map[string]interface{}{
				"type": "function_call", "call_id": "shell_0", "name": "shell",
				"arguments": `{"command":["ls"]}`,
			},
			map[string]interface{}{
				"type": "function_call", "call_id": "shell_1", "name": "shell",
				"arguments": `{"command":["pwd"]}`,
			},
			map[string]interface{}{
				"type": "function_call_output", "call_id": "shell_0", "output": "file.txt",
			},
			map[string]interface{}{
				"type": "function_call_output", "call_id": "shell_1", "output": "/tmp",
			},
		},
	}
	chat, err := TranslateResponsesReqToChat(body, "Kimi-K3")
	if err != nil {
		t.Fatal(err)
	}
	messages, _ := chat["messages"].([]map[string]interface{})
	if len(messages) != 4 {
		t.Fatalf("messages = %v", messages)
	}
	assistant := messages[1]
	if assistant["role"] != "assistant" {
		t.Fatalf("messages[1].role = %v", assistant["role"])
	}
	if assistant["content"] != "Let me check." {
		t.Errorf("assistant content = %v", assistant["content"])
	}
	calls, _ := assistant["tool_calls"].([]map[string]interface{})
	if len(calls) != 2 {
		t.Fatalf("assistant tool_calls = %v", assistant["tool_calls"])
	}
	if calls[0]["id"] != "shell_0" || calls[1]["id"] != "shell_1" {
		t.Errorf("tool_call ids = %v, %v", calls[0]["id"], calls[1]["id"])
	}
	for i, m := range messages[2:] {
		if m["role"] != "tool" {
			t.Errorf("messages[%d].role = %v, want tool", i+2, m["role"])
		}
	}

	// function_call first, text after: same merge must apply.
	body["input"] = []interface{}{
		map[string]interface{}{
			"type": "function_call", "call_id": "shell_0", "name": "shell",
			"arguments": `{"command":["ls"]}`,
		},
		map[string]interface{}{
			"type": "message", "role": "assistant",
			"content": []interface{}{
				map[string]interface{}{"type": "output_text", "text": "done"},
			},
		},
	}
	chat, err = TranslateResponsesReqToChat(body, "Kimi-K3")
	if err != nil {
		t.Fatal(err)
	}
	messages, _ = chat["messages"].([]map[string]interface{})
	if len(messages) != 1 {
		t.Fatalf("messages = %v", messages)
	}
	if messages[0]["content"] != "done" {
		t.Errorf("merged content = %v", messages[0]["content"])
	}
}

func TestTranslateResponsesReqToChat_DropsToolChoiceWithOnlyBuiltinTools(t *testing.T) {
	body := map[string]interface{}{
		"input":       "hello",
		"tool_choice": "required",
		"tools": []interface{}{
			map[string]interface{}{"type": "namespace", "name": "shell"},
			map[string]interface{}{"type": "web_search", "name": "search"},
		},
	}
	chat, err := TranslateResponsesReqToChat(body, "Kimi-K3")
	if err != nil {
		t.Fatal(err)
	}
	if _, exists := chat["tools"]; exists {
		t.Errorf("builtin tools should be omitted: %v", chat["tools"])
	}
	if _, exists := chat["tool_choice"]; exists {
		t.Errorf("tool_choice without function tools should be omitted: %v", chat["tool_choice"])
	}
}

// TestChatFallbackStream_TruncatedToolArgumentsFallBackToEmptyObject covers a
// finish_reason=length truncation that lands mid-arguments: the emitted
// function_call must carry valid JSON ("{}") rather than partial JSON that
// Codex would fail to parse.
func TestChatFallbackStream_TruncatedToolArgumentsFallBackToEmptyObject(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{kimiModel()})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		chunks := []string{
			`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":"{\"command\":[\"ls"}}]}}]}`,
			`{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}`,
		}
		for _, c := range chunks {
			io.WriteString(w, "data: "+c+"\n\n")
			flusher.Flush()
		}
		io.WriteString(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer backend.Close()

	server := NewServer(newMockClient(backend), nil)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"Kimi-K3","input":"list files","stream":true}`))
	req = store.InitTokenUsage(req)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}

	var incomplete map[string]interface{}
	for _, ev := range collectSSE(rec.Body.String()) {
		if ev["type"] == "response.incomplete" {
			incomplete, _ = ev["response"].(map[string]interface{})
		}
	}
	if incomplete == nil {
		t.Fatalf("expected response.incomplete for length finish, got:\n%s", rec.Body.String())
	}
	output, _ := incomplete["output"].([]interface{})
	if len(output) != 1 {
		t.Fatalf("output items = %v", output)
	}
	fc, _ := output[0].(map[string]interface{})
	args, _ := fc["arguments"].(string)
	if args != "{}" {
		t.Errorf("truncated arguments must degrade to {}, got %q", args)
	}
	if !json.Valid([]byte(args)) {
		t.Errorf("arguments are not valid JSON: %q", args)
	}
}

func TestChatFallbackStream_ObjectToolArguments(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{kimiModel()})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		chunks := []string{
			`{"id":"c1","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"Write","arguments":{"file_path":"/tmp/a.go"}}}]}}]}`,
			`{"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		}
		for _, c := range chunks {
			io.WriteString(w, "data: "+c+"\n\n")
			flusher.Flush()
		}
		io.WriteString(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer backend.Close()

	server := NewServer(newMockClient(backend), nil)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"Kimi-K3","input":"write file","stream":true}`))
	req = store.InitTokenUsage(req)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var completed map[string]interface{}
	for _, ev := range collectSSE(rec.Body.String()) {
		if ev["type"] == "response.completed" {
			completed, _ = ev["response"].(map[string]interface{})
		}
	}
	if completed == nil {
		t.Fatalf("no response.completed in stream:\n%s", rec.Body.String())
	}
	output, _ := completed["output"].([]interface{})
	if len(output) != 1 {
		t.Fatalf("output items = %v", output)
	}
	item, _ := output[0].(map[string]interface{})
	if got := item["arguments"]; got != `{"file_path":"/tmp/a.go"}` {
		t.Fatalf("arguments = %v", got)
	}
}

// TestChatFallbackNonStream_RetriesTransientFailures verifies the bridged
// non-stream path retries transient upstream errors (5xx) instead of failing
// the Codex request on the first attempt.
func TestChatFallbackNonStream_RetriesTransientFailures(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{kimiModel()})
	var calls int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		attempt := atomic.AddInt32(&calls, 1)
		if attempt < 3 {
			w.WriteHeader(http.StatusBadGateway)
			io.WriteString(w, `{"error":"upstream temporarily unavailable"}`)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"id":"c1","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1}}`)
	}))
	defer backend.Close()

	server := NewServer(newMockClient(backend), nil)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"Kimi-K3","input":"hi"}`))
	req = store.InitTokenUsage(req)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s (attempts=%d)", rec.Code, rec.Body.String(), calls)
	}
	if got := atomic.LoadInt32(&calls); got != 3 {
		t.Errorf("expected 3 upstream attempts, got %d", got)
	}
}

// TestChatFallbackNonStream_NoRetryOnBadRequest verifies deterministic 400
// errors are surfaced immediately without burning retry attempts.
func TestChatFallbackNonStream_NoRetryOnBadRequest(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{kimiModel()})
	var calls int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&calls, 1)
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"error":"invalid request"}`)
	}))
	defer backend.Close()

	server := NewServer(newMockClient(backend), nil)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"Kimi-K3","input":"hi"}`))
	req = store.InitTokenUsage(req)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if got := atomic.LoadInt32(&calls); got != 1 {
		t.Errorf("400 must not be retried, got %d attempts", got)
	}
}
