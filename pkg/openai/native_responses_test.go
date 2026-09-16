package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

func seedNativeResponsesRegistry(t *testing.T, models []joycode.ModelInfo) {
	t.Helper()
	joycode.DefaultRegistry.Seed(models)
	t.Cleanup(func() { joycode.DefaultRegistry.Seed(joycode.StaticModels()) })
}

func gptAstraModel() joycode.ModelInfo {
	return joycode.ModelInfo{
		Label: "GPT-6 Astra", ChatAPIModel: "gpt-6-astra",
		ExtJSON: map[string]interface{}{"adapterType": joycode.AdapterOpenAIResponse},
	}
}

type astraRoundTripFunc func(*http.Request) (*http.Response, error)

func (f astraRoundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestNativeResponses_GPT6AstraPreservesToolsAndReasoning(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{gptAstraModel()})
	var upstreamBody map[string]interface{}
	client := joycode.NewClient("test-key", "test-user")
	client.SetTransport(astraRoundTripFunc(func(r *http.Request) (*http.Response, error) {
		decoder := json.NewDecoder(r.Body)
		decoder.UseNumber()
		if err := decoder.Decode(&upstreamBody); err != nil {
			t.Fatal(err)
		}
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"application/json"}},
			Body: io.NopCloser(strings.NewReader(
				`{"id":"resp_astra","object":"response","status":"completed","output":[],"usage":{"input_tokens":7,"output_tokens":2,"total_tokens":9}}`)),
			Request: r,
		}, nil
	}))

	server := NewServer(client, nil)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"GPT-6 Astra","input":"hello","reasoning":{"effort":"low"},"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}]}`))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if upstreamBody["model"] != "gpt-6-astra" {
		t.Errorf("model=%v, want gpt-6-astra", upstreamBody["model"])
	}
	reasoning, ok := upstreamBody["reasoning"].(map[string]interface{})
	if !ok || reasoning["effort"] != "low" {
		t.Errorf("reasoning=%v, want preserved low effort", upstreamBody["reasoning"])
	}
	tools, ok := upstreamBody["tools"].([]interface{})
	if !ok || len(tools) != 1 {
		t.Errorf("tools=%v, want one preserved function tool", upstreamBody["tools"])
	}
}

func TestNativeResponses_NonStream(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{gptSolModel()})
	var upstreamBody map[string]interface{}
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		decoder := json.NewDecoder(r.Body)
		decoder.UseNumber()
		if err := decoder.Decode(&upstreamBody); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"resp_1","object":"response","status":"completed","output":[],"usage":{"input_tokens":11,"output_tokens":5,"total_tokens":16}}`))
	}))
	defer backend.Close()

	server := NewServer(newMockClient(backend), nil)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)
	frontend := httptest.NewServer(mux)
	defer frontend.Close()

	for _, path := range []string{"/v1/responses", "/responses"} {
		resp, err := http.Post(frontend.URL+path, "application/json", strings.NewReader(
			`{"model":"gpt-5.6-sol","input":"hello","max_output_tokens":2000}`))
		if err != nil {
			t.Fatal(err)
		}
		data, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("%s status=%d body=%s", path, resp.StatusCode, data)
		}
		var output map[string]interface{}
		json.Unmarshal(data, &output)
		if output["object"] != "response" || output["id"] != "resp_1" {
			t.Errorf("%s output=%v", path, output)
		}
		if upstreamBody["model"] != "GPT-5.6 Sol" {
			t.Errorf("model=%v", upstreamBody["model"])
		}
		if intFromJSON(upstreamBody["max_output_tokens"]) != 64000 {
			t.Errorf("max_output_tokens=%v, want model maximum", upstreamBody["max_output_tokens"])
		}
	}
}

func TestNativeResponses_RejectsAnthropicModel(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{
		gptSolModel(),
		{Label: "Claude Sonnet", ChatAPIModel: "claude-sonnet",
			ExtJSON: map[string]interface{}{"adapterType": joycode.AdapterAnthropic}},
	})
	server := NewServer(&joycode.Client{}, nil)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"Claude Sonnet","input":"hello"}`))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "unsupported_model") {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
}

func TestNativeResponses_StreamPassThrough(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{gptSolModel()})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		io.WriteString(w, `data: {"type":"response.created","response":{"id":"resp_1"}}`+"\n\n")
		flusher.Flush()
		io.WriteString(w, `data: {"type":"response.output_text.delta","delta":"hi"}`+"\n\n")
		flusher.Flush()
		io.WriteString(w, `data: {"type":"response.completed","response":{"usage":{"input_tokens":3,"output_tokens":1}}}`+"\n\n")
		flusher.Flush()
	}))
	defer backend.Close()

	server := NewServer(newMockClient(backend), nil)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"gpt-5.6-sol","input":"hello","stream":true}`))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	out := rec.Body.String()
	if rec.Code != http.StatusOK || !strings.Contains(out, `"type":"response.output_text.delta"`) ||
		!strings.Contains(out, `"type":"response.completed"`) {
		t.Fatalf("status=%d body=%s", rec.Code, out)
	}
}

func collectNativeReads(input string) []nativeResponsesRead {
	var reads []nativeResponsesRead
	for item := range readNativeResponsesSSE(context.Background(), strings.NewReader(input)) {
		reads = append(reads, item)
	}
	return reads
}

func TestReadNativeResponsesSSE_UsageAndFraming(t *testing.T) {
	cases := []struct {
		name     string
		input    string
		wantIn   int
		wantOut  int
		terminal bool
	}{
		{"standard nested", `data: {"type":"response.completed","response":{"usage":{"input_tokens":11,"output_tokens":5}}}` + "\n\n", 11, 5, true},
		{"no space top level", `data:{"type":"response.completed","usage":{"input_tokens":12,"output_tokens":6}}` + "\n\n", 12, 6, true},
		{"multiple spaces", `data:   {"type":"response.incomplete","response":{"usage":{"input_tokens":13,"output_tokens":7}}}` + "\n\n", 13, 7, true},
		{"zero usage", `data: {"type":"response.failed","response":{"usage":{"input_tokens":0,"output_tokens":0}}}` + "\n\n", 0, 0, true},
		{"EOF without blank", `data: {"type":"response.completed","response":{"usage":{"input_tokens":14,"output_tokens":8}}}`, 14, 8, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reads := collectNativeReads(tc.input)
			if len(reads) != 1 {
				t.Fatalf("reads = %#v", reads)
			}
			got := reads[0]
			if got.Err != nil || !got.HasUsage || got.InputTokens != tc.wantIn || got.OutputTokens != tc.wantOut || got.Terminal != tc.terminal {
				t.Errorf("read = %+v, want usage %d/%d terminal=%v", got, tc.wantIn, tc.wantOut, tc.terminal)
			}
		})
	}
}

func TestReadNativeResponsesSSE_MultilineDataAndMetadata(t *testing.T) {
	input := ": keepalive\n" +
		"event: response.completed\n" +
		"id: evt_1\n" +
		`data: {"type":"response.completed",` + "\n" +
		`data: "response":{"usage":{"input_tokens":21,"output_tokens":9}}}` + "\n\n"
	reads := collectNativeReads(input)
	if len(reads) != 1 || reads[0].Err != nil || !reads[0].Terminal || reads[0].InputTokens != 21 || reads[0].OutputTokens != 9 {
		t.Fatalf("reads = %#v", reads)
	}
}

func TestNativeResponses_StreamStoresUsage(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{gptSolModel()})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, `data: {"type":"response.completed","response":{"usage":{"input_tokens":31,"output_tokens":10}}}`+"\n\n")
		w.(http.Flusher).Flush()
	}))
	defer backend.Close()

	server := NewServer(newMockClient(backend), nil)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"gpt-5.6-sol","input":"hello","stream":true}`))
	req = store.InitTokenUsage(req)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	inTk, outTk := store.GetTokenUsage(req)
	if inTk != 31 || outTk != 10 {
		t.Fatalf("usage = %d/%d, want 31/10; body=%s", inTk, outTk, rec.Body.String())
	}
}

func TestNativeResponses_PrematureEOFSurfacesError(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{gptSolModel()})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, `data: {"type":"response.created","response":{"id":"resp_1"}}`+"\n\n")
		io.WriteString(w, `data: {"type":"response.output_text.delta","delta":"partial"}`+"\n\n")
		w.(http.Flusher).Flush()
		// Clean EOF without response.completed/incomplete/failed.
	}))
	defer backend.Close()

	server := NewServer(newMockClient(backend), nil)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"gpt-5.6-sol","input":"hello","stream":true}`))
	req = store.InitTokenUsage(req)
	req = store.InitRequestOutcome(req)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	out := rec.Body.String()
	if !strings.Contains(out, `"code":"stream_truncated"`) {
		t.Errorf("missing stream_truncated error:\n%s", out)
	}
	if strings.Contains(out, `"type":"response.completed"`) || strings.Contains(out, "data: [DONE]") {
		t.Errorf("premature EOF must not fake completion:\n%s", out)
	}
	outcome := store.GetRequestOutcome(req)
	if outcome.StatusCode != http.StatusBadGateway || !strings.Contains(outcome.Error, "stream_truncated") {
		t.Errorf("outcome = %+v", outcome)
	}
}

func TestNativeResponses_FailedEventSurfacesError(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{gptSolModel()})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, `data: {"type":"response.failed","response":{"error":{"message":"model unavailable"},"usage":{"input_tokens":9,"output_tokens":0}}}`+"\n\n")
		w.(http.Flusher).Flush()
	}))
	defer backend.Close()

	server := NewServer(newMockClient(backend), nil)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)
	req := httptest.NewRequest(http.MethodPost, "/responses", strings.NewReader(
		`{"model":"gpt-5.6-sol","input":"hello","stream":true}`))
	req = store.InitTokenUsage(req)
	req = store.InitRequestOutcome(req)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	out := rec.Body.String()
	if !strings.Contains(out, "model unavailable") || !strings.Contains(out, `"code":"upstream_error"`) {
		t.Errorf("missing failed event error:\n%s", out)
	}
	if strings.Contains(out, `"type":"response.failed"`) {
		t.Errorf("failed event must be translated to error, not passed as success:\n%s", out)
	}
	inTk, outTk := store.GetTokenUsage(req)
	if inTk != 9 || outTk != 0 {
		t.Errorf("usage = %d/%d", inTk, outTk)
	}
	if outcome := store.GetRequestOutcome(req); outcome.StatusCode != http.StatusBadGateway {
		t.Errorf("outcome = %+v", outcome)
	}
}

func TestNativeResponses_RateLimitEventIsClassified(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{gptSolModel()})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, `data: {"type":"error","error":{"code":"rate_limit_exceeded","message":"rate limit exceeded"}}`+"\n\n")
		w.(http.Flusher).Flush()
	}))
	defer backend.Close()

	server := NewServer(newMockClient(backend), nil)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)
	req := httptest.NewRequest(http.MethodPost, "/responses", strings.NewReader(
		`{"model":"gpt-5.6-sol","input":"hello","stream":true}`))
	req = store.InitRequestOutcome(req)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if !strings.Contains(rec.Body.String(), `"code":"rate_limit_error"`) {
		t.Errorf("missing rate_limit_error:\n%s", rec.Body.String())
	}
	outcome := store.GetRequestOutcome(req)
	if outcome.StatusCode != http.StatusTooManyRequests || !strings.Contains(outcome.Error, "rate_limit_error") {
		t.Errorf("outcome = %+v", outcome)
	}
}

func TestNativeResponsesNonStream_FailedMarksOutcomeButPreservesResponse(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{gptSolModel()})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"resp_1","object":"response","status":"failed","error":null,"output":[],"usage":{"input_tokens":7,"output_tokens":0}}`))
	}))
	defer backend.Close()
	server := NewServer(newMockClient(backend), nil)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"gpt-5.6-sol","input":"hello"}`))
	req = store.InitTokenUsage(req)
	req = store.InitRequestOutcome(req)
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"status":"failed"`) {
		t.Fatalf("native response must be preserved: status=%d body=%s", rec.Code, rec.Body.String())
	}
	if outcome := store.GetRequestOutcome(req); outcome.StatusCode != http.StatusBadGateway || !strings.Contains(outcome.Error, "upstream_error") {
		t.Errorf("outcome = %+v", outcome)
	}
	if inTk, outTk := store.GetTokenUsage(req); inTk != 7 || outTk != 0 {
		t.Errorf("usage = %d/%d", inTk, outTk)
	}
}

func kimiModel() joycode.ModelInfo {
	return joycode.ModelInfo{Label: "Kimi-K3", ChatAPIModel: "Kimi-K3"}
}

// TestNativeResponses_ChatModelFallbackNonStream verifies that a Responses
// request aimed at a chat-completions model is bridged to the upstream chat
// endpoint and wrapped back into a Responses object.
func TestNativeResponses_ChatModelFallbackNonStream(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{kimiModel()})
	var upstreamBody map[string]interface{}
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasSuffix(r.URL.Path, "/chat/completions") {
			t.Errorf("unexpected upstream path %s", r.URL.Path)
		}
		decoder := json.NewDecoder(r.Body)
		decoder.UseNumber()
		if err := decoder.Decode(&upstreamBody); err != nil {
			t.Fatal(err)
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl_1","object":"chat.completion","choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"hi there"}}],"usage":{"prompt_tokens":12,"completion_tokens":4}}`))
	}))
	defer backend.Close()

	server := NewServer(newMockClient(backend), nil)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"Kimi-K3","input":"hello","instructions":"be terse","max_output_tokens":8192}`))
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var out map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatal(err)
	}
	if out["object"] != "response" || out["status"] != "completed" {
		t.Fatalf("output = %s", rec.Body.String())
	}
	items, _ := out["output"].([]interface{})
	if len(items) != 1 {
		t.Fatalf("output items = %s", rec.Body.String())
	}
	item, _ := items[0].(map[string]interface{})
	if item["type"] != "message" || item["role"] != "assistant" {
		t.Fatalf("item = %v", item)
	}
	usage, _ := out["usage"].(map[string]interface{})
	if intFromJSON(usage["input_tokens"]) != 12 || intFromJSON(usage["output_tokens"]) != 4 {
		t.Fatalf("usage = %v", usage)
	}

	// Verify the translated chat request.
	if upstreamBody["model"] != "Kimi-K3" {
		t.Errorf("model = %v", upstreamBody["model"])
	}
	if intFromJSON(upstreamBody["max_tokens"]) != 8192 {
		t.Errorf("max_tokens = %v", upstreamBody["max_tokens"])
	}
	messages, _ := upstreamBody["messages"].([]interface{})
	if len(messages) != 2 {
		t.Fatalf("messages = %v", upstreamBody["messages"])
	}
	first, _ := messages[0].(map[string]interface{})
	if first["role"] != "system" || first["content"] != "be terse" {
		t.Errorf("first message = %v", first)
	}
	second, _ := messages[1].(map[string]interface{})
	if second["role"] != "user" || second["content"] != "hello" {
		t.Errorf("second message = %v", second)
	}
}

// TestNativeResponses_ChatModelFallbackStream verifies chat-completions
// streaming chunks are re-emitted as Responses events.
func TestNativeResponses_ChatModelFallbackStream(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{kimiModel()})
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		io.WriteString(w, `data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":"he"}}]}`+"\n\n")
		flusher.Flush()
		io.WriteString(w, `data: {"id":"c1","choices":[{"index":0,"delta":{"content":"llo"}}]}`+"\n\n")
		flusher.Flush()
		io.WriteString(w, `data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":2}}`+"\n\n")
		flusher.Flush()
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
	out := rec.Body.String()
	for _, want := range []string{
		`"type":"response.created"`,
		`"type":"response.output_item.added"`,
		`"type":"response.output_text.delta"`,
		`"delta":"he"`, `"delta":"llo"`,
		`"type":"response.completed"`,
		`"input_tokens":9`, `"output_tokens":2`,
	} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in stream:\n%s", want, out)
		}
	}
	if inTk, outTk := store.GetTokenUsage(req); inTk != 9 || outTk != 2 {
		t.Errorf("usage = %d/%d", inTk, outTk)
	}
}

// TestTranslateResponsesReqToChat covers the request-direction translator in
// isolation.
func TestTranslateResponsesReqToChat(t *testing.T) {
	body := map[string]interface{}{
		"model":             "Kimi-K3",
		"instructions":      "be nice",
		"max_output_tokens": 128,
		"stream":            true,
		"input": []interface{}{
			map[string]interface{}{"type": "message", "role": "user", "content": "hi"},
			map[string]interface{}{
				"type": "function_call", "call_id": "call_1", "name": "get_weather",
				"arguments": `{"city":"BJ"}`,
			},
			map[string]interface{}{
				"type": "function_call_output", "call_id": "call_1",
				"output": "sunny",
			},
		},
		"tools": []interface{}{
			map[string]interface{}{
				"type": "function", "name": "get_weather",
				"description": "Get weather", "parameters": map[string]interface{}{"type": "object"},
			},
			map[string]interface{}{
				"type": "namespace", "name": "shell", "namespace": map[string]interface{}{"name": "shell"},
			},
		},
		"tool_choice": "auto",
	}
	chat, err := TranslateResponsesReqToChat(body, "Kimi-K3")
	if err != nil {
		t.Fatal(err)
	}
	if chat["model"] != "Kimi-K3" || chat["stream"] != true {
		t.Errorf("model/stream = %v %v", chat["model"], chat["stream"])
	}
	if intFromJSON(chat["max_tokens"]) != 128 {
		t.Errorf("max_tokens = %v", chat["max_tokens"])
	}
	streamOpts, _ := chat["stream_options"].(map[string]interface{})
	if streamOpts["include_usage"] != true {
		t.Errorf("stream_options = %v", streamOpts)
	}
	messages, _ := chat["messages"].([]map[string]interface{})
	if len(messages) != 4 {
		t.Fatalf("messages = %v", chat["messages"])
	}
	system := messages[0]
	if system["role"] != "system" {
		t.Errorf("messages[0] = %v", system)
	}
	assistant := messages[2]
	if assistant["role"] != "assistant" {
		t.Fatalf("messages[2] = %v", assistant)
	}
	toolCalls, _ := assistant["tool_calls"].([]map[string]interface{})
	if len(toolCalls) != 1 || toolCalls[0]["id"] != "call_1" {
		t.Errorf("tool_calls = %v", toolCalls)
	}
	toolMsg := messages[3]
	if toolMsg["role"] != "tool" || toolMsg["tool_call_id"] != "call_1" || toolMsg["content"] != "sunny" {
		t.Errorf("tool message = %v", toolMsg)
	}
	tools, _ := chat["tools"].([]map[string]interface{})
	if len(tools) != 1 || tools[0]["type"] != "function" {
		t.Fatalf("tools = %v", tools)
	}
	fn, _ := tools[0]["function"].(map[string]interface{})
	if fn["name"] != "get_weather" {
		t.Errorf("function = %v", fn)
	}
}
