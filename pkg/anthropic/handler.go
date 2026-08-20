package anthropic

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/common"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

const chatEndpoint = "/api/saas/openai/v1/chat/completions"
const anthropicEndpoint = "/api/saas/anthropic/v1/messages"

// ClientResolver returns the appropriate joycode.Client for a request.
type ClientResolver func(r *http.Request) *joycode.Client

// Handler serves the Anthropic Messages API.
type Handler struct {
	Client   *joycode.Client
	Resolver ClientResolver
	store    *store.Store
}

// NewHandler creates a new Anthropic API handler.
func NewHandler(c *joycode.Client, s *store.Store) *Handler {
	return &Handler{Client: c, store: s}
}

func (h *Handler) getClient(r *http.Request) *joycode.Client {
	if h.Resolver != nil {
		return h.Resolver(r)
	}
	return h.Client
}

// RegisterRoutes registers the Anthropic Messages API endpoint.
func (h *Handler) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/v1/messages", func(w http.ResponseWriter, r *http.Request) {
		common.ApplyLocalCORS(w, r, "POST, OPTIONS", "Content-Type, Authorization, X-API-Key, Anthropic-Version, Anthropic-Beta")
		h.handleMessages(w, r)
	})
	mux.HandleFunc("/v1/messages/count_tokens", func(w http.ResponseWriter, r *http.Request) {
		common.ApplyLocalCORS(w, r, "POST, OPTIONS", "Content-Type, Authorization, X-API-Key, Anthropic-Version, Anthropic-Beta")
		h.handleCountTokens(w, r)
	})
}

func (h *Handler) handleCountTokens(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodOptions {
		w.WriteHeader(http.StatusOK)
		return
	}
	if r.Method != http.MethodPost {
		writeAnthropicError(w, http.StatusMethodNotAllowed, "method not allowed")
		return
	}
	var req MessageRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeAnthropicRequestError(w, "请求体解析失败: "+err.Error())
		return
	}
	if len(req.Messages) == 0 {
		writeAnthropicRequestError(w, "messages must not be empty")
		return
	}
	writeAnthropicJSON(w, http.StatusOK, map[string]int{"input_tokens": EstimateInputTokens(&req)})
}

func (h *Handler) handleMessages(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodOptions {
		w.Header().Set("Access-Control-Allow-Methods", "POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization, X-API-Key, Anthropic-Version, Anthropic-Beta")
		w.WriteHeader(200)
		return
	}
	if r.Method != http.MethodPost {
		writeAnthropicError(w, 405, "method not allowed")
		return
	}

	var req MessageRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		reqLog(r).Error("decode anthropic request", "error", err)
		writeAnthropicError(w, 400, fmt.Sprintf("请求体解析失败: %s。请检查请求是否完整，或尝试开启新对话减少上下文长度。", err.Error()))
		return
	}
	accountDefault := store.GetAccountDefaultModel(r)
	systemDefault := ""
	configuredMaxTokens := 0
	if h.store != nil {
		systemDefault = h.store.GetSetting("default_model")
		configuredMaxTokens = h.store.GetIntSetting("default_max_tokens", 0)
	}
	info := ResolveModelInfo(req.Model, accountDefault, systemDefault)
	store.SetModel(r, info.DisplayName())

	// Resolve max_tokens against the model's own cap, which requires the model to
	// be known first. Upstream advertises 64000 for every visible model.
	requestedMaxTokens := req.MaxTokens
	req.MaxTokens = info.EffectiveMaxTokens(requestedMaxTokens, configuredMaxTokens)

	reqLog(r).Info("anthropic request", "model", req.Model, "resolved", info.APIName(),
		"adapter", info.Adapter(), "stream", req.Stream,
		"max_tokens", req.MaxTokens, "max_tokens_requested", requestedMaxTokens,
		"messages", len(req.Messages), "tools", len(req.Tools))

	client := h.getClient(r)

	// Models with the openai-response adapter are served only by the upstream
	// Responses API; both chat/completions and the native Anthropic endpoint
	// reject them with the opaque error 1032.
	if info.Adapter() == joycode.AdapterOpenAIResponse {
		if req.OutputConfig != nil && req.OutputConfig.Effort != "" {
			if _, ok := normalizeResponsesEffort(req.OutputConfig.Effort); !ok {
				writeAnthropicRequestError(w, "output_config.effort must be one of none, low, medium, high, xhigh, max, or extra")
				return
			}
		}
		if req.Stream {
			flusher, ok := w.(http.Flusher)
			if !ok {
				writeAnthropicError(w, 500, "streaming unsupported by server")
				return
			}
			h.handleResponsesStream(w, r, &req, client, flusher, info)
		} else {
			h.handleResponsesNonStream(w, r, &req, client, info)
		}
		return
	}

	if req.Stream {
		h.handleStream(w, r, &req, client)
	} else {
		h.handleNonStream(w, r, &req, client)
	}
}

// useNativeAnthropic reports whether a request should go to the upstream native
// Anthropic endpoint. It dispatches on the resolved model's adapter type rather
// than a name prefix, and keeps the enable_claude gate.
func (h *Handler) useNativeAnthropic(r *http.Request, req *MessageRequest, systemDefault string) bool {
	if !ClaudeNativeEnabled(h.store) {
		return false
	}
	info := ResolveModelInfo(req.Model, store.GetAccountDefaultModel(r), systemDefault)
	if info.Adapter() == joycode.AdapterAnthropic {
		return true
	}
	// The client asked for a Claude model by a name the registry does not know
	// (e.g. an upstream alias like claude-sonnet-4-5-20250929). Honour the
	// intent and use the native path.
	return IsNativeAnthropicModel(req.Model) && info.Adapter() != joycode.AdapterOpenAIResponse
}

func (h *Handler) handleNonStream(w http.ResponseWriter, r *http.Request, req *MessageRequest, client *joycode.Client) {
	systemDefault := ""
	if h.store != nil {
		systemDefault = h.store.GetSetting("default_model")
	}
	if h.useNativeAnthropic(r, req, systemDefault) {
		h.handleNativeAnthropicNonStream(w, r, req, client, systemDefault)
		return
	}
	// Preemptive truncation: estimate tokens and truncate before sending
	if rounds := PreemptiveTruncate(req); rounds < 0 {
		writeAnthropicRequestError(w, "上下文过长，自动截断后仍超出限制，请使用 /compact 或开启新对话。")
		return
	} else if rounds > 0 {
		slog.Warn("preemptive truncation applied (non-stream)", "rounds", rounds)
	}

	jcBody := TranslateRequest(req, store.GetAccountDefaultModel(r), systemDefault)
	logRequestDetails(r, "translated request (non-stream)", jcBody)
	maxRetries := 3
	if h.store != nil {
		maxRetries = h.store.GetIntSetting("max_retries", 3)
	}
	var jcResp map[string]interface{}
	var lastErr error

	for attempt := 1; attempt <= maxRetries; attempt++ {
		jcResp, lastErr = client.PostContext(r.Context(), chatEndpoint, jcBody)
		if lastErr != nil {
			if isContextLimitError(lastErr.Error()) {
				// Progressive truncation on context limit. Route through
				// PreemptiveTruncate so the output-token budget (max_tokens)
				// is accounted for the same way as in preflight truncation.
				if rounds := PreemptiveTruncate(req); rounds > 0 {
					jcBody = TranslateRequest(req, store.GetAccountDefaultModel(r), systemDefault)
					reqLog(r).Warn("retrying with truncated messages (non-stream)", "attempt", attempt)
					continue
				}
				reqLog(r).Warn("context limit exceeded, cannot truncate further")
				writeAnthropicRequestError(w, "上下文长度超出模型限制，且无法进一步截断。请压缩对话历史或开启新对话。原始错误: "+lastErr.Error())
				return
			}
			reqLog(r).Error("non-stream retry error", "attempt", attempt, "max", maxRetries, "error", lastErr)
			if attempt < maxRetries {
				waitForRetry(r.Context(), time.Duration(attempt)*500*time.Millisecond)
			}
			continue
		}
		break
	}

	if lastErr != nil {
		errMsg := lastErr.Error()
		if isContextLimitError(errMsg) {
			writeAnthropicRequestError(w, "上下文长度超出模型限制。请压缩对话历史或开启新对话。原始错误: "+errMsg)
			return
		}
		if isTimeoutError(lastErr) {
			reqLog(r).Error("upstream timeout after retries", "error", lastErr)
			writeAnthropicError(w, 504, "上游服务响应超时，请稍后重试。如果问题持续，请尝试减少上下文长度或开启新对话。原始错误: "+errMsg)
			return
		}
		if strings.Contains(errMsg, "content_filter") || strings.Contains(errMsg, "SENSITIVE_CONTENT") {
			reqLog(r).Warn("upstream content_filter (non-stream), returning detailed error")
			writeContentFilterError(w, errMsg)
			return
		}
		writeAnthropicError(w, 500, errMsg)
		return
	}
	resp := TranslateResponse(jcResp, req.Model)
	// Check for content_filter in non-stream response
	if choices, ok := jcResp["choices"].([]interface{}); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]interface{}); ok {
			if fr, ok := choice["finish_reason"].(string); ok && fr == "content_filter" {
				reqLog(r).Warn("content_filter in non-stream response")
				raw, _ := json.Marshal(choice)
				writeContentFilterError(w, string(raw))
				return
			}
		}
	}
	if usage, ok := jcResp["usage"].(map[string]interface{}); ok {
		inTk, _ := usage["prompt_tokens"].(float64)
		outTk, _ := usage["completion_tokens"].(float64)
		store.SetTokenUsage(r, int(inTk), int(outTk))
	}
	writeAnthropicJSON(w, 200, resp)
}

// prependReader replays a buffered first line before reading from the underlying source.
type prependReader struct {
	first  []byte
	offset int
	source io.Reader
	body   io.ReadCloser
}

func (r *prependReader) Read(p []byte) (int, error) {
	if r.offset < len(r.first) {
		n := copy(p, r.first[r.offset:])
		r.offset += n
		return n, nil
	}
	return r.source.Read(p)
}

func (r *prependReader) Close() error {
	return r.body.Close()
}

func (h *Handler) handleStream(w http.ResponseWriter, r *http.Request, req *MessageRequest, client *joycode.Client) {
	systemDefault := ""
	if h.store != nil {
		systemDefault = h.store.GetSetting("default_model")
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeAnthropicError(w, 500, "streaming not supported")
		return
	}
	if h.useNativeAnthropic(r, req, systemDefault) {
		h.handleNativeAnthropicStream(w, r, req, client, flusher, systemDefault)
		return
	}

	jcBody := TranslateRequest(req, store.GetAccountDefaultModel(r), systemDefault)
	jcBody["stream"] = true
	logRequestDetails(r, "translated request (stream)", jcBody)

	// Preemptive truncation: estimate tokens and truncate before sending
	if rounds := PreemptiveTruncate(req); rounds < 0 {
		writeAnthropicRequestError(w, "上下文过长，自动截断后仍超出限制，请使用 /compact 或开启新对话。")
		return
	} else if rounds > 0 {
		reqLog(r).Warn("preemptive truncation applied (stream)", "rounds", rounds)
	}

	jcBody = TranslateRequest(req, store.GetAccountDefaultModel(r), systemDefault)
	jcBody["stream"] = true

	// Commit SSE headers + message_start early so we can send heartbeat ping
	// events while waiting for the upstream to respond. The JoyCode upstream
	// buffers the entire response (TTFB 10–30s for reasoning models); without
	// keepalive, Claude Code and other clients may time out during this gap.
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(200)

	msgID := NewMessageID()
	model := req.Model
	totalOutput := 0

	FormatSSE(w, "message_start", sseMessageStart{
		Type: "message_start",
		Message: MessageResponse{
			ID: msgID, Type: "message", Role: "assistant",
			Model: model, Content: []ContentBlock{}, Usage: Usage{},
		},
	})
	FormatSSE(w, "ping", ssePing{Type: "ping"})
	flusher.Flush()

	// Heartbeat: send periodic ping events while upstream is silent.
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
				FormatSSE(w, "ping", ssePing{Type: "ping"})
				flusher.Flush()
			}
		}
	}()

	// Connect with retry, progressive auto-truncate on context limit
	resp, err := h.connectStreamWithRetry(r, jcBody, client)
	for truncRound := 0; err != nil && isContextLimitError(err.Error()) && truncRound < maxTruncationRounds; truncRound++ {
		reqLog(r).Warn("stream context limit, truncating", "round", truncRound+1)
		if rounds := PreemptiveTruncate(req); rounds <= 0 {
			break
		}
		jcBody = TranslateRequest(req, store.GetAccountDefaultModel(r), systemDefault)
		jcBody["stream"] = true
		resp, err = h.connectStreamWithRetry(r, jcBody, client)
	}
	close(stopHeartbeat)
	<-heartbeatDone
	if err != nil {
		errMsg := err.Error()
		if isContextLimitError(errMsg) {
			reqLog(r).Warn("context limit exceeded (stream), cannot proceed even after progressive truncation")
			writeStreamError(w, flusher, "上下文长度超出模型限制，已尝试自动截断但仍无法满足。请压缩对话历史或开启新对话。原始错误: "+errMsg)
			return
		}
		if isTimeoutError(err) {
			reqLog(r).Error("upstream timeout (stream) after retries", "error", err)
			writeStreamError(w, flusher, "上游服务响应超时，请稍后重试。如果问题持续，请尝试减少上下文长度或开启新对话。原始错误: "+errMsg)
			return
		}
		if strings.Contains(errMsg, "content_filter") || strings.Contains(errMsg, "SENSITIVE_CONTENT") {
			reqLog(r).Warn("upstream content_filter (stream), returning detailed error")
			writeStreamError(w, flusher, errMsg)
			return
		}
		reqLog(r).Error("stream failed after retries", "error", errMsg)
		writeStreamError(w, flusher, errMsg)
		return
	}
	defer resp.Body.Close()

	type toolCallAccum struct {
		ID        string
		Name      string
		Arguments string
	}
	toolCalls := make(map[int]*toolCallAccum)
	currentBlockIndex := 0
	// openBlock tracks the single currently-open content block (-1 = none).
	// Anthropic requires blocks to be sequential: a block must be stopped
	// before the next one starts, and deltas must target the open block.
	// Text blocks are streamed in real time; tool blocks are buffered and
	// emitted sequentially at finish time, which guarantees protocol
	// compliance even when upstream interleaves or parallelises tool deltas.
	openBlock := -1
	openIsText := false
	closeOpenBlock := func() {
		if openBlock < 0 {
			return
		}
		FormatSSE(w, "content_block_stop", sseContentBlockStop{
			Type: "content_block_stop", Index: openBlock,
		})
		openBlock = -1
		currentBlockIndex++
	}
	// Tool blocks are emitted sequentially at finish time in sorted tool
	// index order: each block gets a content_block_start, one input_json_delta
	// holding the full accumulated arguments JSON, and a content_block_stop.
	emitToolBlock := func(tc *toolCallAccum) {
		closeOpenBlock()
		openIsText = false
		openBlock = currentBlockIndex
		tcID := tc.ID
		if tcID == "" {
			tcID = "toolu_" + newID()
			tc.ID = tcID
		}
		FormatSSE(w, "content_block_start", sseContentBlockStart{
			Type:  "content_block_start",
			Index: currentBlockIndex,
			ContentBlock: ContentBlock{
				Type:  "tool_use",
				ID:    tcID,
				Name:  tc.Name,
				Input: json.RawMessage("{}"),
			},
		})
		args := tc.Arguments
		if args == "" || !json.Valid([]byte(args)) {
			// Upstream truncated the arguments mid-stream (e.g. finish_reason
			// length): degrade to an empty object so clients don't try to
			// parse partial JSON.
			args = "{}"
		}
		FormatSSE(w, "content_block_delta", sseContentBlockDelta{
			Type:  "content_block_delta",
			Index: currentBlockIndex,
			Delta: deltaText{Type: "input_json_delta", PartialJSON: args},
		})
		closeOpenBlock()
	}
	openTextBlock := func() {
		openIsText = true
		openBlock = currentBlockIndex
		FormatSSE(w, "content_block_start", sseContentBlockStart{
			Type:         "content_block_start",
			Index:        currentBlockIndex,
			ContentBlock: ContentBlock{Type: "text", Text: ""},
		})
		flusher.Flush()
	}

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	chunkCount := 0
	var streamInTk, streamOutTk int
	finishReasonSeen := false

	for scanner.Scan() {
		line := scanner.Text()
		chunkCount++
		chunk := ParseStreamChunk(line)
		if chunk == nil || len(chunk.Choices) == 0 {
			if chunk != nil && chunk.Usage != nil {
				streamInTk = chunk.Usage.PromptTokens
				streamOutTk = chunk.Usage.CompletionTokens
			}
			continue
		}
		choice := chunk.Choices[0]
		if chunk.Usage != nil {
			streamInTk = chunk.Usage.PromptTokens
			streamOutTk = chunk.Usage.CompletionTokens
		}

		for _, tc := range choice.Delta.ToolCalls {
			idx := tc.Index
			if _, exists := toolCalls[idx]; !exists {
				toolCalls[idx] = &toolCallAccum{
					ID:   tc.ID,
					Name: tc.Function.Name,
				}
			}
			if tc.ID != "" {
				toolCalls[idx].ID = tc.ID
			}
			if tc.Function.Name != "" {
				toolCalls[idx].Name = tc.Function.Name
			}
			toolCalls[idx].Arguments += string(tc.Function.Arguments)
			// Temporary diagnostics for the missing file_path investigation
			if strings.Contains(string(tc.Function.Arguments), "}{") {
				reqLog(r).Warn("tool arguments chunk contains json boundary", "index", idx, "name", toolCalls[idx].Name, "chunk", tc.Function.Arguments)
			}
		}

		text := choice.Delta.Content
		if text != "" {
			if openBlock < 0 || !openIsText {
				closeOpenBlock()
				openTextBlock()
			}
			totalOutput += len(text)
			FormatSSE(w, "content_block_delta", sseContentBlockDelta{
				Type:  "content_block_delta",
				Index: openBlock,
				Delta: deltaText{Type: "text_delta", Text: text},
			})
			flusher.Flush()
		}

		if choice.FinishReason != nil {
			fr := *choice.FinishReason
			finishReasonSeen = true
			reqLog(r).Info("stream completed", "chunks", chunkCount, "reason", fr, "tools", len(toolCalls))

			// Ensure at least one content block exists — Anthropic SDK requires it
			if openBlock < 0 && currentBlockIndex == 0 && len(toolCalls) == 0 {
				openTextBlock()
			}

			// Emit buffered tool blocks sequentially in sorted tool index order.
			if len(toolCalls) > 0 {
				toolIndexes := make([]int, 0, len(toolCalls))
				for idx := range toolCalls {
					toolIndexes = append(toolIndexes, idx)
				}
				sort.Ints(toolIndexes)
				for _, idx := range toolIndexes {
					emitToolBlock(toolCalls[idx])
				}
			} else {
				closeOpenBlock()
			}

			if fr == "content_filter" {
				// Don't disguise a filter-truncated turn as a clean end_turn
				// (issue #2). Blocks are already closed above; surface an error.
				reqLog(r).Warn("upstream content_filter mid-stream, surfacing as error")
				writeStreamError(w, flusher, "上游内容安全策略拦截了本次回复，输出可能不完整。结束原因: content_filter")
			} else {
				stopReason := "end_turn"
				switch fr {
				case "tool_calls":
					stopReason = "tool_use"
				case "length":
					stopReason = "max_tokens"
				case "stop":
					stopReason = "end_turn"
				}
				// Some chat-completions gateways (including Kimi variants) emit
				// finish_reason=stop even when tool deltas were present. Anthropic
				// clients must receive tool_use in that case or they treat the turn
				// as final and stop without executing the tool.
				if len(toolCalls) > 0 && fr != "length" {
					stopReason = "tool_use"
				}
				FormatSSE(w, "message_delta", sseMessageDelta{
					Type:  "message_delta",
					Delta: deltaStop{StopReason: stopReason},
					Usage: struct {
						OutputTokens int `json:"output_tokens"`
					}{OutputTokens: totalOutput / 4},
				})
				FormatSSE(w, "message_stop", sseMessageStop{Type: "message_stop"})
				flusher.Flush()
			}
		}
	}

	// If the loop ended without ever seeing an upstream finish_reason, the stream
	// was truncated — a read error (scanner.Err) or a clean EOF mid-generation.
	// Don't fake a clean completion (issue #2): close any open content blocks for
	// well-formed SSE, then surface an error event so the client shows a failure
	// (and can retry) instead of a silent truncated "success". If finishReasonSeen
	// is true the loop already emitted message_stop, so a trailing read error after
	// a complete answer is ignored.
	if !finishReasonSeen {
		closeOpenBlock()
		if err := scanner.Err(); err != nil {
			reqLog(r).Error("stream ended abnormally before finish_reason", "error", err, "chunks", chunkCount)
			store.SetRequestError(r, http.StatusBadGateway, "stream_read_error: "+err.Error())
			writeStreamError(w, flusher, "读取上游流式响应失败，本次回复不完整，请重试。原始错误: "+err.Error())
		} else {
			reqLog(r).Warn("stream closed before finish_reason (premature upstream close)", "chunks", chunkCount)
			store.SetRequestError(r, http.StatusBadGateway, "stream_truncated: upstream closed before finish_reason")
			writeStreamError(w, flusher, "上游在返回结束标记前断开了流式响应，本次回复可能不完整，请重试。")
		}
	}
	if streamInTk > 0 || streamOutTk > 0 {
		store.SetTokenUsage(r, streamInTk, streamOutTk)
	}
}

func (h *Handler) handleNativeAnthropicStream(w http.ResponseWriter, r *http.Request, req *MessageRequest, client *joycode.Client, flusher http.Flusher, systemDefault string) {
	body := TranslateAnthropicRequest(req, store.GetAccountDefaultModel(r), systemDefault)
	logRequestDetails(r, "translated native anthropic request (stream)", body)

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(200)

	// Send an immediate keepalive so the client knows the connection is alive
	// right away, before we even hit the upstream.
	w.Write([]byte(": keepalive\n\n")) //nolint:errcheck
	flusher.Flush()

	// Background heartbeat: 1-second interval to prevent client timeout during
	// round-1 collection, web search execution, and round-2 connection.
	// SSE comment lines (": ...") are ignored by the Anthropic SDK.
	stopHB := make(chan struct{})
	hbDone := make(chan struct{})
	go func() {
		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()
		defer close(hbDone)
		for {
			select {
			case <-stopHB:
				return
			case <-ticker.C:
				w.Write([]byte(": keepalive\n\n")) //nolint:errcheck
				flusher.Flush()
			}
		}
	}()

	resp, err := h.connectNativeAnthropicStreamWithRetry(r, body, client)
	if err != nil {
		close(stopHB)
		<-hbDone
		reqLog(r).Error("native anthropic stream failed after retries", "error", err)
		writeStreamError(w, flusher, err.Error())
		return
	}

	// Collect round-1 while heartbeat keeps the client alive.
	round1, collectErr := collectNativeAnthropicStream(resp.Body)
	resp.Body.Close()

	if collectErr == nil && round1.StopReason == "tool_use" {
		searchCalls := extractWebSearchCalls(round1.Content)
		if len(searchCalls) > 0 {
			reqLog(r).Info("web_search interceptor: intercepting tool_use", "calls", len(searchCalls))
			// Heartbeat is still running during search + round-2 connect.
			toolResults := executeWebSearchCalls(searchCalls, client)
			body2 := appendWebSearchResults(body, round1.Content, toolResults)
			resp2, err2 := h.connectNativeAnthropicStreamWithRetry(r, body2, client)
			close(stopHB)
			<-hbDone
			if err2 != nil {
				reqLog(r).Error("web_search interceptor: round 2 failed", "error", err2)
				writeStreamError(w, flusher, err2.Error())
				return
			}
			reqLog(r).Info("web_search interceptor: piping round 2 response")
			// Fully pipe round-2: client gets a complete self-consistent SSE stream
			// with the upstream's own message_start (correct id + usage).
			h.pipeNativeAnthropicStream(w, r, resp2, flusher)
			return
		}
	}

	close(stopHB)
	<-hbDone

	if collectErr != nil {
		reqLog(r).Error("native anthropic stream collect error", "error", collectErr)
		writeStreamError(w, flusher, "读取上游流式响应失败: "+collectErr.Error())
		return
	}

	// Transparent pass-through: replay buffered round-1 lines to the client.
	var inTk, outTk int
	inTk, outTk = round1.InputTk, round1.OutputTk
	sawTerminal := false
	pendingEvent := ""
	for _, line := range round1.RawLines {
		payload := unwrapNativeAnthropicSSE(line)
		if payload == "" {
			continue
		}
		if strings.HasPrefix(payload, "event: ") {
			pendingEvent = strings.TrimSpace(strings.TrimPrefix(payload, "event: "))
			continue
		}
		if strings.HasPrefix(payload, "data: ") {
			raw := strings.TrimSpace(strings.TrimPrefix(payload, "data: "))
			if raw == "[DONE]" {
				fmt.Fprintln(w, "data: [DONE]")
				fmt.Fprintln(w)
				sawTerminal = true
				flusher.Flush()
				pendingEvent = ""
				continue
			}
			if raw != "" && strings.HasPrefix(raw, "{") {
				eventName := pendingEvent
				if eventName == "" {
					eventName = nativeAnthropicEventType(raw)
				}
				if eventName == "message_stop" || eventName == "error" {
					sawTerminal = true
				}
				if eventName != "" {
					fmt.Fprintf(w, "event: %s\n", eventName)
				}
				fmt.Fprintf(w, "data: %s\n\n", raw)
				flusher.Flush()
				pendingEvent = ""
				continue
			}
			fmt.Fprintf(w, "data: %s\n\n", raw)
			flusher.Flush()
		}
		pendingEvent = ""
	}
	if !sawTerminal {
		reqLog(r).Warn("native anthropic stream closed before message_stop (premature upstream close)")
		writeStreamError(w, flusher, "上游在返回结束标记前断开了流式响应，本次回复可能不完整，请重试。")
	}
	if inTk > 0 || outTk > 0 {
		store.SetTokenUsage(r, inTk, outTk)
	}
}

// pipeNativeAnthropicStream fully pipes an upstream SSE response to the client.
// Used for the second round of the web_search agentic loop. All events including
// message_start are forwarded unchanged so the client receives a complete,
// self-consistent Anthropic SSE stream with correct IDs and usage fields.
//
// The upstream wraps every line with an extra "data: " prefix:
//
//	data: event: content_block_start
//	data: data: {"type":"content_block_start",...}
//
// We strip the outer "data: " then handle the inner "event: " / "data: " normally.
func (h *Handler) pipeNativeAnthropicStream(w http.ResponseWriter, r *http.Request, resp *http.Response, flusher http.Flusher) {
	defer resp.Body.Close()
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var inTk, outTk int
	pendingEvent := ""
	sawTerminal := false
	for scanner.Scan() {
		raw := scanner.Text()
		// Strip ALL leading "data: " prefixes (upstream double-wraps each line).
		line := raw
		for strings.HasPrefix(line, "data: ") {
			line = strings.TrimPrefix(line, "data: ")
		}
		line = strings.TrimSpace(line)

		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "event: ") {
			pendingEvent = strings.TrimSpace(strings.TrimPrefix(line, "event: "))
			continue
		}
		if line == "[DONE]" {
			fmt.Fprintln(w, "data: [DONE]")
			fmt.Fprintln(w)
			flusher.Flush()
			sawTerminal = true
			continue
		}
		if !strings.HasPrefix(line, "{") {
			continue
		}
		eventName := pendingEvent
		if eventName == "" {
			eventName = nativeAnthropicEventType(line)
		}
		if eventName == "message_stop" || eventName == "error" {
			sawTerminal = true
		}
		if eventName != "" {
			fmt.Fprintf(w, "event: %s\n", eventName)
		}
		fmt.Fprintf(w, "data: %s\n\n", line)
		updateNativeAnthropicUsage(line, &inTk, &outTk)
		pendingEvent = ""
		flusher.Flush()
	}
	if !sawTerminal {
		if err := scanner.Err(); err != nil {
			reqLog(r).Error("web_search round2 stream ended abnormally", "error", err)
			writeStreamError(w, flusher, "读取上游流式响应失败，本次回复不完整，请重试。原始错误: "+err.Error())
		} else {
			reqLog(r).Warn("web_search round2 stream closed before message_stop")
			writeStreamError(w, flusher, "上游在返回结束标记前断开了流式响应，本次回复可能不完整，请重试。")
		}
	} else if err := scanner.Err(); err != nil {
		reqLog(r).Error("web_search round2 stream scanner error", "error", err)
	}
	reqLog(r).Info("pipe round2 done", "sawTerminal", sawTerminal)
	if inTk > 0 || outTk > 0 {
		store.SetTokenUsage(r, inTk, outTk)
	}
}

func (h *Handler) handleNativeAnthropicNonStream(w http.ResponseWriter, r *http.Request, req *MessageRequest, client *joycode.Client, systemDefault string) {
	body := TranslateAnthropicRequest(req, store.GetAccountDefaultModel(r), systemDefault)
	logRequestDetails(r, "translated native anthropic request (non-stream)", body)

	resp, err := h.connectNativeAnthropicStreamWithRetry(r, body, client)
	if err != nil {
		reqLog(r).Error("native anthropic non-stream failed after retries", "error", err)
		writeAnthropicError(w, 500, err.Error())
		return
	}

	// --- Round 1: collect the full upstream response ---
	round1, collectErr := collectNativeAnthropicStream(resp.Body)
	resp.Body.Close()
	if collectErr != nil {
		reqLog(r).Error("native anthropic non-stream collect error", "error", collectErr)
		writeAnthropicError(w, 500, "读取上游响应失败: "+collectErr.Error())
		return
	}

	// Check for upstream error embedded in round1 raw lines
	for _, line := range round1.RawLines {
		p := unwrapNativeAnthropicSSE(line)
		if strings.HasPrefix(p, "data: ") {
			p = strings.TrimSpace(strings.TrimPrefix(p, "data: "))
		}
		if p != "" && p != "[DONE]" && strings.HasPrefix(p, "{") && isUpstreamError(p) {
			writeAnthropicError(w, 500, p)
			return
		}
	}

	// --- Web search interception ---
	if round1.StopReason == "tool_use" {
		searchCalls := extractWebSearchCalls(round1.Content)
		if len(searchCalls) > 0 {
			reqLog(r).Info("web_search interceptor (non-stream): intercepting tool_use", "calls", len(searchCalls))
			toolResults := executeWebSearchCalls(searchCalls, client)
			body2 := appendWebSearchResults(body, round1.Content, toolResults)

			resp2, err2 := h.connectNativeAnthropicStreamWithRetry(r, body2, client)
			if err2 != nil {
				reqLog(r).Error("web_search interceptor (non-stream): round 2 failed", "error", err2)
				writeAnthropicError(w, 500, err2.Error())
				return
			}
			round2, collectErr2 := collectNativeAnthropicStream(resp2.Body)
			resp2.Body.Close()
			if collectErr2 != nil {
				reqLog(r).Error("web_search interceptor (non-stream): collect round 2 error", "error", collectErr2)
				writeAnthropicError(w, 500, "读取上游响应失败: "+collectErr2.Error())
				return
			}
			content := round2.Content
			if len(content) == 0 {
				content = []ContentBlock{{Type: "text", Text: ""}}
			}
			stopReason := round2.StopReason
			inTk := round1.InputTk + round2.InputTk
			outTk := round1.OutputTk + round2.OutputTk
			if inTk > 0 || outTk > 0 {
				store.SetTokenUsage(r, inTk, outTk)
			}
			writeAnthropicJSON(w, 200, &MessageResponse{
				ID:         NewMessageID(),
				Type:       "message",
				Role:       "assistant",
				Content:    content,
				Model:      req.Model,
				StopReason: &stopReason,
				Usage:      Usage{InputTokens: inTk, OutputTokens: outTk},
			})
			return
		}
	}

	// --- Normal path: return round1 result directly ---
	content := round1.Content
	if len(content) == 0 {
		content = []ContentBlock{{Type: "text", Text: ""}}
	}
	stopReason := round1.StopReason
	if round1.InputTk > 0 || round1.OutputTk > 0 {
		store.SetTokenUsage(r, round1.InputTk, round1.OutputTk)
	}
	writeAnthropicJSON(w, 200, &MessageResponse{
		ID:         NewMessageID(),
		Type:       "message",
		Role:       "assistant",
		Content:    content,
		Model:      req.Model,
		StopReason: &stopReason,
		Usage:      Usage{InputTokens: round1.InputTk, OutputTokens: round1.OutputTk},
	})
}

func (h *Handler) connectNativeAnthropicStreamWithRetry(r *http.Request, body map[string]interface{}, client *joycode.Client) (*http.Response, error) {
	maxRetries := 3
	if h.store != nil {
		maxRetries = h.store.GetIntSetting("max_retries", 3)
	}
	var lastErr error
	for attempt := 1; attempt <= maxRetries; attempt++ {
		resp, err := client.PostAnthropicStreamContext(r.Context(), anthropicEndpoint, body)
		if err != nil {
			lastErr = err
			reqLog(r).Error("native anthropic stream connect error", "attempt", attempt, "max", maxRetries, "error", err)
			if attempt < maxRetries {
				waitForRetry(r.Context(), time.Duration(attempt)*500*time.Millisecond)
			}
			continue
		}
		br := bufio.NewReaderSize(resp.Body, 64*1024)
		firstLine, err := br.ReadString('\n')
		if err != nil {
			resp.Body.Close()
			lastErr = fmt.Errorf("read first line: %w", err)
			if attempt < maxRetries {
				waitForRetry(r.Context(), time.Duration(attempt)*500*time.Millisecond)
			}
			continue
		}
		if err := nativeAnthropicLineError(firstLine); err != nil {
			resp.Body.Close()
			lastErr = err
			if attempt < maxRetries {
				waitForRetry(r.Context(), time.Duration(attempt)*500*time.Millisecond)
			}
			continue
		}
		resp.Body = &prependReader{first: []byte(firstLine), source: br, body: resp.Body}
		return resp, nil
	}
	return nil, lastErr
}

func unwrapNativeAnthropicSSE(line string) string {
	trimmed := strings.TrimSpace(line)
	for {
		next := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
		if next == trimmed {
			return trimmed
		}
		trimmed = next
	}
}

func nativeAnthropicEventType(payload string) string {
	var event struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal([]byte(payload), &event); err != nil {
		return ""
	}
	return event.Type
}

func nativeAnthropicLineError(line string) error {
	payload := unwrapNativeAnthropicSSE(line)
	if payload == "" || payload == "[DONE]" || !strings.HasPrefix(payload, "{") {
		return nil
	}
	if isUpstreamError(payload) {
		return fmt.Errorf("%s", payload)
	}
	return nil
}

func updateNativeAnthropicUsage(payload string, inputTokens, outputTokens *int) {
	if payload == "" || payload == "[DONE]" || !strings.HasPrefix(payload, "{") {
		return
	}
	var event struct {
		Type    string `json:"type"`
		Message struct {
			Usage struct {
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
			} `json:"usage"`
		} `json:"message"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal([]byte(payload), &event); err != nil {
		return
	}
	if event.Message.Usage.InputTokens > 0 {
		*inputTokens = event.Message.Usage.InputTokens
	}
	if event.Message.Usage.OutputTokens > 0 {
		*outputTokens = event.Message.Usage.OutputTokens
	}
	if event.Usage.InputTokens > 0 {
		*inputTokens = event.Usage.InputTokens
	}
	if event.Usage.OutputTokens > 0 {
		*outputTokens = event.Usage.OutputTokens
	}
}

// connectStreamWithRetry attempts to connect to upstream with retries.
// Peeks at the first SSE line to detect errors before returning the response.
func (h *Handler) connectStreamWithRetry(r *http.Request, jcBody map[string]interface{}, client *joycode.Client) (*http.Response, error) {
	maxRetries := 3
	if h.store != nil {
		maxRetries = h.store.GetIntSetting("max_retries", 3)
	}
	var lastErr error

	for attempt := 1; attempt <= maxRetries; attempt++ {
		resp, err := client.PostStreamContext(r.Context(), chatEndpoint, jcBody)
		if err != nil {
			lastErr = err
			reqLog(r).Error("stream connect error", "attempt", attempt, "max", maxRetries, "error", err)
			if attempt < maxRetries {
				waitForRetry(r.Context(), time.Duration(attempt)*500*time.Millisecond)
			}
			continue
		}

		br := bufio.NewReaderSize(resp.Body, 64*1024)
		firstLine, err := br.ReadString('\n')
		if err != nil {
			resp.Body.Close()
			lastErr = fmt.Errorf("read first line: %w", err)
			reqLog(r).Error("stream read first line", "attempt", attempt, "max", maxRetries, "error", lastErr)
			if attempt < maxRetries {
				waitForRetry(r.Context(), time.Duration(attempt)*500*time.Millisecond)
			}
			continue
		}

		trimmed := strings.TrimSpace(firstLine)
		dataContent := strings.TrimPrefix(trimmed, "data: ")
		if isUpstreamError(dataContent) {
			resp.Body.Close()
			lastErr = fmt.Errorf("upstream error: %s", truncate(dataContent, 500))
			logUpstreamError(r, attempt, maxRetries, dataContent)
			if isContextLimitError(dataContent) {
				return nil, lastErr
			}
			// SENSITIVE_CONTENT errors are deterministic — retrying is pointless
			if strings.Contains(dataContent, "SENSITIVE_CONTENT") {
				return nil, lastErr
			}
			if attempt < maxRetries {
				waitForRetry(r.Context(), time.Duration(attempt)*500*time.Millisecond)
			}
			continue
		}

		// Check first line for content_filter (content + finish_reason in same chunk)
		if filtered, chunkData := extractContentFilterInfo(dataContent); filtered {
			resp.Body.Close()
			lastErr = fmt.Errorf("%s", truncate(chunkData, 500))
			reqLog(r).Warn("content_filter detected in first chunk, not retrying", "chunk", truncate(chunkData, 300))
			return nil, lastErr
		}

		// Peek second line to detect content_filter with separate finish_reason
		replayLines := firstLine
		secondLine, sErr := br.ReadString('\n')
		if sErr == nil {
			replayLines += secondLine
			trimmedSecond := strings.TrimSpace(secondLine)
			dataSecond := strings.TrimPrefix(trimmedSecond, "data: ")
			if filtered, chunkData := extractContentFilterInfo(dataSecond); filtered {
				resp.Body.Close()
				lastErr = fmt.Errorf("%s", truncate(chunkData, 500))
				reqLog(r).Warn("content_filter detected in second chunk, not retrying", "chunk", truncate(chunkData, 300))
				return nil, lastErr
			}
		}

		// Wrap body to replay buffered lines for the scanner
		originalBody := resp.Body
		resp.Body = &prependReader{
			first:  []byte(replayLines),
			source: br,
			body:   originalBody,
		}
		reqLog(r).Info("stream connected", "attempt", attempt)
		return resp, nil
	}
	return nil, fmt.Errorf("stream failed after %d attempts: %w", maxRetries, lastErr)
}

// isTimeoutError checks if the error is caused by an upstream timeout.
func isTimeoutError(err error) bool {
	return common.IsTimeoutError(err)
}

// isContextLimitError checks if the upstream error indicates context length exceeded.
func isContextLimitError(body string) bool {
	lower := strings.ToLower(body)
	return strings.Contains(lower, "context length") ||
		strings.Contains(lower, "context window") ||
		strings.Contains(lower, "token limit") ||
		strings.Contains(lower, "tokens exceeded") ||
		strings.Contains(lower, "input length") ||
		strings.Contains(lower, "model_context_window_exceeded") ||
		strings.Contains(lower, "prompt length") ||
		strings.Contains(lower, "max_input_tokens")
}

// extractContentFilterInfo checks if a SSE data line contains content_filter finish_reason.
// Returns whether content_filter was detected and the raw data line for error reporting.
func extractContentFilterInfo(line string) (bool, string) {
	if line == "" || line == "[DONE]" {
		return false, ""
	}
	var parsed struct {
		Choices []struct {
			FinishReason *string         `json:"finish_reason"`
			Message      json.RawMessage `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(line), &parsed); err != nil {
		return false, ""
	}
	for _, c := range parsed.Choices {
		if c.FinishReason != nil && *c.FinishReason == "content_filter" {
			return true, line
		}
	}
	return false, ""
}

func isUpstreamError(line string) bool {
	if line == "" || line == "[DONE]" {
		return false
	}
	var parsed struct {
		Choices []interface{} `json:"choices"`
		Error   interface{}   `json:"error"`
		Code    interface{}   `json:"code"`
		Status  string        `json:"status"`
		Msg     string        `json:"msg"`
	}
	if err := json.Unmarshal([]byte(line), &parsed); err != nil {
		return false
	}
	if len(parsed.Choices) > 0 {
		return false
	}
	return parsed.Error != nil || parsed.Code != nil || parsed.Status != "" || parsed.Msg != ""
}

func truncate(s string, maxLen int) string {
	return common.Truncate(s, maxLen)
}

func writeAnthropicJSON(w http.ResponseWriter, code int, v interface{}) {
	b, err := json.Marshal(v)
	if err != nil {
		slog.Error("writeAnthropicJSON: marshal failed", "error", err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write(b)
}

// writeStreamError emits an Anthropic `error` SSE event mid-stream. Used when an
// already-started stream is truncated, so the client surfaces a failure (and can
// retry) instead of treating a partial response as a clean completion (issue #2).
func writeStreamError(w http.ResponseWriter, flusher http.Flusher, msg string) {
	FormatSSE(w, "error", map[string]interface{}{
		"type":  "error",
		"error": map[string]string{"type": "api_error", "message": msg},
	})
	flusher.Flush()
}

func writeAnthropicError(w http.ResponseWriter, code int, msg string) {
	writeAnthropicJSON(w, code, map[string]interface{}{
		"type":  "error",
		"error": map[string]string{"type": "api_error", "message": msg},
	})
}

// writeContentFilterError forwards the upstream content_filter response verbatim.
// Uses invalid_request_error type so Claude Code treats it as non-retryable.
func writeContentFilterError(w http.ResponseWriter, msgs ...string) {
	msg := "content_filter"
	if len(msgs) > 0 && msgs[0] != "" {
		msg = msgs[0]
	}
	writeAnthropicJSON(w, 400, map[string]interface{}{
		"type":  "error",
		"error": map[string]string{"type": "invalid_request_error", "message": msg},
	})
}

func waitForRetry(ctx context.Context, delay time.Duration) bool {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-timer.C:
		return true
	case <-ctx.Done():
		return false
	}
}

func writeAnthropicRequestError(w http.ResponseWriter, msg string) {
	writeAnthropicJSON(w, 400, map[string]interface{}{
		"type":  "error",
		"error": map[string]string{"type": "invalid_request_error", "message": msg},
	})
}
