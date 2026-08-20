package openai

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
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
	if err := ValidateChatRequest(&req); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", "invalid_request", "")
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
	requestedMaxTokens := req.RequestedMaxTokens()
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
	resp, err := client.PostContext(r.Context(), "/api/saas/openai/v1/chat/completions", jcBody)
	if err != nil {
		slog.Error("chat non-stream upstream error", "model", model, "error", err)
		status := upstreamHTTPStatus(err)
		errorType := "api_error"
		code := "upstream_error"
		if status == http.StatusBadRequest {
			errorType = "invalid_request_error"
			code = "invalid_request"
		}
		writeOpenAIError(w, status, err.Error(), errorType, code, "")
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
		writeError(w, http.StatusInternalServerError, "streaming unsupported by server")
		return
	}

	// Wait briefly for an upstream rejection before committing HTTP 200. For slow
	// model handshakes, connectUpstreamStream starts SSE heartbeats after 5s.
	streamStart := time.Now()
	resp, committed := connectUpstreamStream(w, flusher, r, func() (*http.Response, error) {
		return client.PostStreamContext(r.Context(), "/api/saas/openai/v1/chat/completions", jcBody)
	})
	if resp == nil {
		return
	}
	defer resp.Body.Close()
	if !committed {
		prepareSSE(w)
		w.WriteHeader(http.StatusOK)
	}
	slog.Info("stream: connected to upstream", "model", model, "ttfb_ms", time.Since(streamStart).Milliseconds())

	reads := readChatSSE(r.Context(), resp.Body)
	heartbeat := time.NewTicker(5 * time.Second)
	defer heartbeat.Stop()
	var inTk, outTk int
	sawFinish := false
	for {
		select {
		case <-r.Context().Done():
			return
		case <-heartbeat.C:
			writeSSEHeartbeat(w, flusher)
		case item, open := <-reads:
			if !open {
				if sawFinish {
					writeSSEDone(w, flusher)
					if inTk > 0 || outTk > 0 {
						store.SetTokenUsage(r, inTk, outTk)
					}
					return
				}
				writeSSEError(w, flusher, "upstream stream ended before finish_reason", "api_error", "stream_truncated")
				return
			}
			if item.Err != nil {
				writeSSEError(w, flusher, item.Err.Error(), "api_error", "stream_read_error")
				return
			}
			if item.Done {
				if inTk > 0 || outTk > 0 {
					store.SetTokenUsage(r, inTk, outTk)
				}
				writeSSEDone(w, flusher)
				return
			}
			if item.BusinessErr != nil {
				writeSSEError(w, flusher, item.BusinessErr.Error(), "api_error", "upstream_error")
				return
			}
			if item.FinishReason != "" {
				sawFinish = true
			}
			if item.InputTokens > 0 || item.OutputTokens > 0 {
				inTk, outTk = item.InputTokens, item.OutputTokens
			}
			fmt.Fprintf(w, "data: %s\n\n", item.Data)
			flusher.Flush()
		}
	}
}

func isTimeoutError(msg string) bool {
	return common.IsTimeoutError(errors.New(msg))
}
