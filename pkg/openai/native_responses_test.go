package openai

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
)

func seedNativeResponsesRegistry(t *testing.T, models []joycode.ModelInfo) {
	t.Helper()
	joycode.DefaultRegistry.Seed(models)
	t.Cleanup(func() { joycode.DefaultRegistry.Seed(joycode.StaticModels()) })
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

func TestNativeResponses_RequiresResponsesModel(t *testing.T) {
	seedNativeResponsesRegistry(t, []joycode.ModelInfo{
		gptSolModel(),
		{Label: "JoyAI-Code-1.5", ChatAPIModel: "JoyAI-Code-1.5"},
	})
	server := NewServer(&joycode.Client{}, nil)
	mux := http.NewServeMux()
	server.RegisterRoutes(mux)
	req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(
		`{"model":"JoyAI-Code-1.5","input":"hello"}`))
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
