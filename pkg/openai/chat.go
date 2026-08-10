package openai

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/common"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
	if !requirePOST(w, r) {
		return
	}
	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		slog.Error("decode chat request", "error", err)
		writeError(w, 400, fmt.Sprintf("请求体解析失败: %s。请检查请求是否完整，或尝试开启新对话减少上下文长度。", err.Error()))
		return
	}
	systemDefault := ""
	configuredMaxTokens := 0
	if s.store != nil {
		systemDefault = s.store.GetSetting("default_model")
		configuredMaxTokens = s.store.GetIntSetting("default_max_tokens", 0)
	}
	info := ResolveModelInfo(req.Model, store.GetAccountDefaultModel(r), systemDefault)
	store.SetModel(r, info.DisplayName())

	// Resolve max_tokens against the model's own cap once the model is known.
	requestedMaxTokens := req.MaxTokens
	req.MaxTokens = info.EffectiveMaxTokens(requestedMaxTokens, configuredMaxTokens)
	slog.Info("chat request", "model", req.Model, "resolved", info.APIName(),
		"adapter", info.Adapter(), "stream", req.Stream,
		"max_tokens", req.MaxTokens, "max_tokens_requested", requestedMaxTokens)

	client := s.getClient(r)

	// Dispatch on the model's upstream adapter: sending a model to the wrong
	// endpoint fails with the opaque upstream error 1032.
	switch info.Adapter() {
	case joycode.AdapterOpenAIResponse:
		if req.Stream {
			s.handleResponsesStream(w, r, client, &req, info)
		} else {
			s.handleResponsesNonStream(w, r, client, &req, info)
		}
		return
	case joycode.AdapterAnthropic:
		// Claude models are served only by the native Anthropic endpoint, which
		// this OpenAI-compatible handler does not speak. Fail loudly rather than
		// letting upstream return an opaque 1032.
		writeError(w, 400, fmt.Sprintf(
			"模型 %s 需要 Anthropic 原生接口，请改用 /v1/messages 端点，或在此端点选择非 Claude 模型。",
			info.DisplayName()))
		return
	}

	jcBody := TranslateRequest(&req, info.APIName())
	if req.Stream {
		s.handleStreamChat(w, r, client, jcBody, info.DisplayName())
	} else {
		s.handleNonStreamChat(w, r, client, jcBody, info.DisplayName())
	}
}

func (s *Server) handleNonStreamChat(w http.ResponseWriter, r *http.Request, client *joycode.Client, jcBody map[string]interface{}, model string) {
	resp, err := client.Post("/api/saas/openai/v1/chat/completions", jcBody)
	if err != nil {
		slog.Error("chat non-stream upstream error", "model", model, "error", err)
		msg := err.Error()
		code := 500
		if isTimeoutError(msg) {
			code = 504
			msg = "上游服务响应超时，请稍后重试。原始错误: " + msg
		}
		writeError(w, code, msg)
		return
	}
	if usage, ok := resp["usage"].(map[string]interface{}); ok {
		inTk, _ := usage["prompt_tokens"].(float64)
		outTk, _ := usage["completion_tokens"].(float64)
		store.SetTokenUsage(r, int(inTk), int(outTk))
	}
	writeJSON(w, 200, TranslateResponse(resp, model))
}

func (s *Server) handleStreamChat(w http.ResponseWriter, r *http.Request, client *joycode.Client, jcBody map[string]interface{}, model string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		slog.Error("streaming not supported by response writer")
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "close")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.WriteHeader(200)

	// Start a heartbeat goroutine. The upstream JoyCode API buffers the entire
	// response before sending anything (TTFB can be 10–30s for reasoning models).
	// Without keepalive, downstream clients (Claude Code, OpenAI clients) may
	// time out or show "no response" during this gap. SSE comment lines (": ...")
	// are part of the spec and ignored by all compliant clients.
	stopHeartbeat := make(chan struct{})
	heartbeatDone := make(chan struct{})
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		defer close(heartbeatDone)
		for {
			select {
			case <-stopHeartbeat:
				return
			case <-ticker.C:
				if _, err := w.Write([]byte(": keepalive\n\n")); err != nil {
					return
				}
				flusher.Flush()
			}
		}
	}()

	streamStart := time.Now()
	resp, err := client.PostStream("/api/saas/openai/v1/chat/completions", jcBody)
	if err != nil {
		close(stopHeartbeat)
		<-heartbeatDone
		slog.Error("chat stream upstream error", "model", model, "error", err)
		msg := err.Error()
		if isTimeoutError(msg) {
			msg = "上游服务响应超时，请稍后重试。原始错误: " + msg
		}
		fmt.Fprintf(w, "data: {\"error\":{\"message\":\"%s\"}}\n\n", msg)
		flusher.Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}
	defer resp.Body.Close()
	close(stopHeartbeat)
	<-heartbeatDone
	slog.Info("stream: connected to upstream", "model", model, "ttfb_ms", time.Since(streamStart).Milliseconds())

	// Pipe JoyCode SSE response line-by-line — already OpenAI-compatible format.
	// Using bufio.Scanner (not raw Read) ensures each SSE event is forwarded
	// as soon as it arrives, without buffering multiple events into one write.
	// Also extract usage tokens from the final chunk for dashboard stats.
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var inTk, outTk int
	sawDone := false
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			// Empty line separates SSE events; forward as-is.
			w.Write([]byte("\n"))
			flusher.Flush()
			continue
		}
		if strings.Contains(line, "[DONE]") {
			sawDone = true
		}
		w.Write([]byte(line))
		w.Write([]byte("\n"))
		flusher.Flush()
		// Extract usage from data lines (the final chunk carries usage stats)
		if strings.HasPrefix(line, "data: ") && !strings.Contains(line, "[DONE]") {
			var chunk struct {
				Usage *struct {
					PromptTokens     int `json:"prompt_tokens"`
					CompletionTokens int `json:"completion_tokens"`
				} `json:"usage"`
			}
			if json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk) == nil && chunk.Usage != nil {
				inTk = chunk.Usage.PromptTokens
				outTk = chunk.Usage.CompletionTokens
			}
		}
	}
	if err := scanner.Err(); err != nil {
		slog.Error("chat stream read error", "model", model, "error", err)
	}
	// Ensure the stream terminates with [DONE]. Some upstream responses omit it
	// (e.g. premature close, certain error paths), leaving clients like
	// CherryStudio hanging in "generating" state. Always send a final [DONE].
	if !sawDone {
		slog.Warn("stream ended without [DONE], sending terminator", "model", model)
		w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}
	if inTk > 0 || outTk > 0 {
		store.SetTokenUsage(r, inTk, outTk)
	}
}

func isTimeoutError(msg string) bool {
	return common.IsTimeoutError(errors.New(msg))
}
