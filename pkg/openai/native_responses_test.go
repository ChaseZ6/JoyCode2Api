package openai

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

func collectNativeReads(input string) []nativeResponsesRead {
	var reads []nativeResponsesRead
	for item := range readNativeResponsesSSE(strings.NewReader(input)) {
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
