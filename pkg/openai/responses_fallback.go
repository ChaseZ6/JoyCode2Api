package openai

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

// This file bridges the OpenAI Responses API (used by Codex / CC Switch) to the
// upstream Chat Completions endpoint, for models whose adapterType is
// "chat-completions" (empty). Codex only speaks Responses; without this bridge
// every chat-completions model (Kimi, GLM, Doubao, ...) is unusable from Codex.

// handleResponsesViaChatNonStream serves a Responses request by calling the
// upstream chat-completions endpoint and re-wrapping the answer as a
// Responses object.
func (s *Server) handleResponsesViaChatNonStream(w http.ResponseWriter, r *http.Request, client *joycode.Client, chatBody map[string]interface{}, displayName string) {
	maxRetries := 3
	if s.store != nil {
		maxRetries = s.store.GetIntSetting("max_retries", 3)
	}
	if maxRetries < 1 {
		maxRetries = 1
	}
	var resp map[string]interface{}
	var err error
	for attempt := 1; attempt <= maxRetries; attempt++ {
		if requestCanceled(r, nil) {
			return
		}
		resp, err = client.PostContext(r.Context(), "/api/saas/openai/v1/chat/completions", chatBody)
		if err == nil {
			break
		}
		if requestCanceled(r, err) {
			slog.Info("responses-to-chat request canceled", "attempt", attempt)
			return
		}
		// Deterministic client-side failures (bad request, content filter) will
		// fail identically on retry — only transient errors are worth another
		// attempt.
		if status := upstreamHTTPStatus(err); status == http.StatusBadRequest {
			break
		}
		// A gateway timeout has already consumed the upstream wait budget; a
		// second long request only increases the chance that the caller cancels.
		if isTimeoutError(err.Error()) {
			break
		}
		slog.Error("responses->chat non-stream upstream error", "model", displayName, "attempt", attempt, "max", maxRetries, "error", err)
		if attempt < maxRetries {
			select {
			case <-r.Context().Done():
				return
			case <-time.After(time.Duration(attempt) * 500 * time.Millisecond):
			}
		}
	}
	if err != nil {
		if requestCanceled(r, err) {
			return
		}
		status := upstreamHTTPStatus(err)
		copyRetryAfter(w, err)
		errorType := "api_error"
		code := "upstream_error"
		if status == http.StatusBadRequest {
			errorType = "invalid_request_error"
			code = "invalid_request"
		}
		writeOpenAIError(w, status, err.Error(), errorType, code, "")
		return
	}
	inTk, outTk := 0, 0
	if usage, ok := resp["usage"].(map[string]interface{}); ok {
		inTk = numberToInt(usage["prompt_tokens"])
		outTk = numberToInt(usage["completion_tokens"])
		store.SetTokenUsage(r, inTk, outTk)
	}
	out, err := chatCompletionToResponses(resp, displayName, inTk, outTk)
	if err != nil {
		store.SetRequestError(r, http.StatusBadGateway, "chat->responses wrap failed: "+err.Error())
		writeOpenAIError(w, http.StatusBadGateway, "解析上游响应失败: "+err.Error(), "api_error", "upstream_error", "")
		return
	}
	writeJSON(w, http.StatusOK, out)
}

// chatCompletionToResponses converts an upstream chat.completion object into
// an OpenAI Responses object.
func chatCompletionToResponses(resp map[string]interface{}, model string, inTk, outTk int) (map[string]interface{}, error) {
	output := []map[string]interface{}{}
	finish := ""
	if choices, ok := resp["choices"].([]interface{}); ok && len(choices) > 0 {
		choice, _ := choices[0].(map[string]interface{})
		finish, _ = choice["finish_reason"].(string)
		message, _ := choice["message"].(map[string]interface{})
		if message != nil {
			output = append(output, chatMessageToResponsesItems(message)...)
		}
	}
	if len(output) == 0 {
		output = append(output, map[string]interface{}{
			"id": "msg_" + newShortID(), "type": "message", "role": "assistant",
			"status": "completed", "content": []map[string]interface{}{},
		})
	}

	status := "completed"
	var incomplete interface{}
	switch finish {
	case "length":
		status = "incomplete"
		incomplete = map[string]interface{}{"reason": "max_output_tokens"}
	case "content_filter":
		status = "incomplete"
		incomplete = map[string]interface{}{"reason": "content_filter"}
	}

	out := map[string]interface{}{
		"id":         "resp_" + newShortID(),
		"object":     "response",
		"created_at": time.Now().Unix(),
		"status":     status,
		"model":      model,
		"output":     output,
		"usage": map[string]interface{}{
			"input_tokens":  inTk,
			"output_tokens": outTk,
			"total_tokens":  inTk + outTk,
		},
	}
	if incomplete != nil {
		out["incomplete_details"] = incomplete
	}
	return out, nil
}

// chatMessageToResponsesItems splits a chat assistant message into Responses
// output items: reasoning -> reasoning item, text -> message item,
// tool_calls -> function_call items.
func chatMessageToResponsesItems(message map[string]interface{}) []map[string]interface{} {
	items := []map[string]interface{}{}
	if reasoning, ok := message["reasoning_content"].(string); ok && reasoning != "" {
		items = append(items, map[string]interface{}{
			"id": "rs_" + newShortID(), "type": "reasoning",
			"summary": []map[string]interface{}{{"type": "summary_text", "text": reasoning}},
		})
	}
	content, _ := message["content"].(string)
	if content != "" {
		items = append(items, map[string]interface{}{
			"id": "msg_" + newShortID(), "type": "message", "role": "assistant",
			"status":  "completed",
			"content": []map[string]interface{}{{"type": "output_text", "text": content}},
		})
	}
	if calls, ok := message["tool_calls"].([]interface{}); ok {
		for _, raw := range calls {
			call, ok := raw.(map[string]interface{})
			if !ok {
				continue
			}
			fn, _ := call["function"].(map[string]interface{})
			name, _ := fn["name"].(string)
			args, _ := joycode.NormalizeToolArguments(fn["arguments"])
			callID, _ := call["id"].(string)
			if callID == "" {
				callID = "call_" + newShortID()
			}
			if args == "" || !json.Valid([]byte(args)) {
				args = "{}"
			}
			items = append(items, map[string]interface{}{
				"id": "fc_" + newShortID(), "type": "function_call",
				"call_id": callID, "name": name, "arguments": args,
				"status": "completed",
			})
		}
	}
	return items
}

// handleResponsesViaChatStream serves a streaming Responses request by
// translating upstream chat.completion.chunk SSE events into Responses SSE
// events.
func (s *Server) handleResponsesViaChatStream(w http.ResponseWriter, r *http.Request, client *joycode.Client, chatBody map[string]interface{}, displayName string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming unsupported by server")
		return
	}
	resp, committed := connectUpstreamStream(w, flusher, r, func() (*http.Response, error) {
		return client.PostStreamContext(r.Context(), "/api/saas/openai/v1/chat/completions", chatBody)
	})
	if resp == nil {
		return
	}
	defer resp.Body.Close()
	if !committed {
		prepareSSE(w)
		w.WriteHeader(http.StatusOK)
	}

	state := newResponsesFromChatStreamState(w, flusher, r, displayName)
	state.emitCreated()

	reads := readChatSSE(r.Context(), resp.Body)
	heartbeat := time.NewTicker(5 * time.Second)
	defer heartbeat.Stop()
	// After [DONE] the upstream may send a trailing usage-only chunk (set via
	// stream_options.include_usage) before EOF. Wait a short grace window for
	// it instead of finalizing immediately on [DONE], otherwise the token
	// stats on response.completed are always 0. If no usage arrives within the
	// window, finalize with what we have.
	const doneGracePeriod = 3 * time.Second
	var doneTimer *time.Timer
	var doneTimerC <-chan time.Time
	for {
		select {
		case <-r.Context().Done():
			return
		case <-heartbeat.C:
			writeSSEHeartbeat(w, flusher)
		case <-doneTimerC:
			// Grace window after [DONE] expired without a usage chunk.
			state.complete()
			writeSSEDone(w, flusher)
			return
		case item, open := <-reads:
			if !open {
				if !state.terminal && !state.failed {
					if state.finishReason != "" {
						// finish_reason arrived but completion was deferred
						// (e.g. waiting for a trailing usage chunk).
						state.complete()
					} else {
						state.fail("upstream stream ended before a terminal event", "stream_truncated")
					}
				}
				writeSSEDone(w, flusher)
				return
			}
			if item.Err != nil {
				state.fail(item.Err.Error(), "stream_read_error")
				writeSSEDone(w, flusher)
				return
			}
			if item.BusinessErr != nil {
				state.fail(item.BusinessErr.Error(), "upstream_error")
				writeSSEDone(w, flusher)
				return
			}
			if item.Done {
				// [DONE] received: don't finalize yet, wait for a possible
				// trailing usage chunk within the grace window. If the upstream
				// closes the stream (EOF) first, the !open branch above
				// finalizes via state.complete().
				if doneTimerC == nil {
					doneTimer = time.NewTimer(doneGracePeriod)
					doneTimerC = doneTimer.C
				}
				continue
			}
			if item.InputTokens > 0 || item.OutputTokens > 0 {
				state.inputTokens, state.outputTokens = item.InputTokens, item.OutputTokens
				// Trailing usage chunk after [DONE]: finalize now.
				if doneTimerC != nil {
					doneTimer.Stop()
					state.complete()
					writeSSEDone(w, flusher)
					return
				}
			}
			if err := state.consumeChunk(item.Data, item.FinishReason); err != nil {
				state.fail(err.Error(), "upstream_protocol_error")
				writeSSEDone(w, flusher)
				return
			}
		}
	}
}

// responsesFromChatStreamState converts chat.completion.chunk deltas into
// Responses stream events and tracks usage for the final response.completed.
type responsesFromChatStreamState struct {
	w            http.ResponseWriter
	flusher      http.Flusher
	request      *http.Request
	id           string
	created      int64
	model        string
	inputTokens  int
	outputTokens int
	terminal     bool
	failed       bool
	textOpen     bool
	textIndex    int
	reasonOpen   bool
	reasonIndex  int
	nextOutput   int
	textItemID   string
	reasonItemID string
	tools        map[int]*responsesToolStream
	sawReasoning bool
	sawText      bool
	finishReason string
	reasonText   strings.Builder
	textBody     strings.Builder
}

type responsesToolStream struct {
	OutputIndex int
	ItemID      string
	CallID      string
	Name        string
	Announced   bool
	ArgsDone    bool
	Args        strings.Builder
}

func newResponsesFromChatStreamState(w http.ResponseWriter, flusher http.Flusher, r *http.Request, model string) *responsesFromChatStreamState {
	return &responsesFromChatStreamState{
		w: w, flusher: flusher, request: r,
		id: "resp_" + newShortID(), created: time.Now().Unix(), model: model,
		textItemID: "msg_" + newShortID(), reasonItemID: "rs_" + newShortID(),
		tools: make(map[int]*responsesToolStream),
	}
}

func (s *responsesFromChatStreamState) baseResponse(status string) map[string]interface{} {
	return map[string]interface{}{
		"id": s.id, "object": "response", "created_at": s.created,
		"status": status, "model": s.model,
	}
}

func (s *responsesFromChatStreamState) emit(eventType string, payload map[string]interface{}) {
	payload["type"] = eventType
	writeSSEData(s.w, s.flusher, payload)
}

func (s *responsesFromChatStreamState) emitCreated() {
	s.emit("response.created", map[string]interface{}{
		"response": s.baseResponse("in_progress"),
	})
	s.emit("response.in_progress", map[string]interface{}{
		"response": s.baseResponse("in_progress"),
	})
}

func (s *responsesFromChatStreamState) fail(message, code string) {
	if s.terminal || s.failed {
		return
	}
	s.failed = true
	writeSSEError(s.w, s.flusher, message, "api_error", code)
}

func (s *responsesFromChatStreamState) openText() {
	if s.textOpen {
		return
	}
	s.closeReasoning()
	s.textIndex = s.nextIndex()
	s.textOpen = true
	s.emit("response.output_item.added", map[string]interface{}{
		"output_index": s.textIndex,
		"item": map[string]interface{}{
			"id": s.textItemID, "type": "message", "role": "assistant",
			"status": "in_progress", "content": []interface{}{},
		},
	})
	s.emit("response.content_part.added", map[string]interface{}{
		"item_id": s.textItemID, "output_index": s.textIndex, "content_index": 0,
		"part": map[string]interface{}{"type": "output_text", "text": ""},
	})
}

func (s *responsesFromChatStreamState) closeText() {
	if !s.textOpen {
		return
	}
	s.textOpen = false
	s.emit("response.output_text.done", map[string]interface{}{
		"item_id": s.textItemID, "output_index": s.textIndex, "content_index": 0,
		"text": s.textBody.String(),
	})
	s.emit("response.content_part.done", map[string]interface{}{
		"item_id": s.textItemID, "output_index": s.textIndex, "content_index": 0,
		"part": map[string]interface{}{"type": "output_text", "text": s.textBody.String()},
	})
	s.emit("response.output_item.done", map[string]interface{}{
		"output_index": s.textIndex,
		"item": map[string]interface{}{
			"id": s.textItemID, "type": "message", "role": "assistant",
			"status":  "completed",
			"content": []map[string]interface{}{{"type": "output_text", "text": s.textBody.String()}},
		},
	})
}

func (s *responsesFromChatStreamState) openReasoning() {
	if s.reasonOpen {
		return
	}
	s.reasonIndex = s.nextIndex()
	s.reasonOpen = true
	s.emit("response.output_item.added", map[string]interface{}{
		"output_index": s.reasonIndex,
		"item": map[string]interface{}{
			"id": s.reasonItemID, "type": "reasoning",
			"summary": []interface{}{},
		},
	})
}

func (s *responsesFromChatStreamState) closeReasoning() {
	if !s.reasonOpen {
		return
	}
	s.reasonOpen = false
	s.emit("response.reasoning_summary_text.done", map[string]interface{}{
		"item_id": s.reasonItemID, "output_index": s.reasonIndex,
		"summary_index": 0, "text": s.reasonText.String(),
	})
	s.emit("response.reasoning_summary_part.done", map[string]interface{}{
		"item_id": s.reasonItemID, "output_index": s.reasonIndex,
		"summary_index": 0,
		"part":          map[string]interface{}{"type": "summary_text", "text": s.reasonText.String()},
	})
	s.emit("response.output_item.done", map[string]interface{}{
		"output_index": s.reasonIndex,
		"item": map[string]interface{}{
			"id": s.reasonItemID, "type": "reasoning",
			"summary": []map[string]interface{}{{"type": "summary_text", "text": s.reasonText.String()}},
		},
	})
}

// nextIndex allocates the next output_index. Indexes are monotonically
// increasing and never reused: Codex reconciles streamed items by
// output_index, and a collision (e.g. a reasoning item and a function_call
// both landing on 1) makes it drop the tool call, which surfaces as the
// agent stopping after the first tool round.
func (s *responsesFromChatStreamState) nextIndex() int {
	idx := s.nextOutput
	s.nextOutput++
	return idx
}

// consumeChunk handles one upstream chat.completion.chunk data payload.
func (s *responsesFromChatStreamState) consumeChunk(data string, finishReason string) error {
	var chunk struct {
		Choices []struct {
			Delta struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
				ToolCalls        []struct {
					Index    int    `json:"index"`
					ID       string `json:"id"`
					Type     string `json:"type"`
					Function struct {
						Name      string                `json:"name"`
						Arguments joycode.ToolArguments `json:"arguments"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"delta"`
			FinishReason *string `json:"finish_reason"`
		} `json:"choices"`
	}
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return fmt.Errorf("invalid chat stream chunk: %w", err)
	}
	if len(chunk.Choices) == 0 {
		return nil
	}
	choice := chunk.Choices[0]

	if choice.Delta.ReasoningContent != "" {
		s.openReasoning()
		s.sawReasoning = true
		s.reasonText.WriteString(choice.Delta.ReasoningContent)
		s.emit("response.reasoning_summary_text.delta", map[string]interface{}{
			"item_id": s.reasonItemID, "output_index": s.reasonIndex,
			"summary_index": 0, "delta": choice.Delta.ReasoningContent,
		})
	}
	if choice.Delta.Content != "" {
		s.openText()
		s.sawText = true
		s.textBody.WriteString(choice.Delta.Content)
		s.emit("response.output_text.delta", map[string]interface{}{
			"item_id": s.textItemID, "output_index": s.textIndex,
			"content_index": 0, "delta": choice.Delta.Content,
		})
	}
	for _, tc := range choice.Delta.ToolCalls {
		tool, ok := s.tools[tc.Index]
		if !ok {
			s.closeReasoning()
			s.closeText()
			tool = &responsesToolStream{
				OutputIndex: s.nextIndex(),
				ItemID:      "fc_" + newShortID(),
				CallID:      tc.ID,
				Name:        tc.Function.Name,
			}
			if tool.CallID == "" {
				tool.CallID = "call_" + newShortID()
			}
			s.tools[tc.Index] = tool
		}
		if tc.ID != "" {
			tool.CallID = tc.ID
		}
		if tc.Function.Name != "" {
			tool.Name = tc.Function.Name
		}
		if !tool.Announced && tool.Name != "" {
			tool.Announced = true
			s.emit("response.output_item.added", map[string]interface{}{
				"output_index": tool.OutputIndex,
				"item": map[string]interface{}{
					"id": tool.ItemID, "type": "function_call",
					"call_id": tool.CallID, "name": tool.Name,
					"arguments": "", "status": "in_progress",
				},
			})
		}
		if tc.Function.Arguments != "" {
			args := string(tc.Function.Arguments)
			tool.Args.WriteString(args)
			s.emit("response.function_call_arguments.delta", map[string]interface{}{
				"item_id": tool.ItemID, "output_index": tool.OutputIndex,
				"delta": args,
			})
		}
	}

	if choice.FinishReason != nil && *choice.FinishReason != "" {
		s.finishReason = *choice.FinishReason
	} else if finishReason != "" {
		s.finishReason = finishReason
	}
	// Completion is deferred until the stream ends ([DONE] or EOF) so a
	// trailing usage-only chunk — sent after the finish chunk when
	// stream_options.include_usage is set — is captured before the terminal
	// response.completed event is emitted with token stats.
	return nil
}

// complete closes any open items and emits the terminal response.completed /
// response.incomplete event. It is idempotent.
func (s *responsesFromChatStreamState) complete() {
	if s.terminal || s.failed {
		return
	}
	s.terminal = true
	s.closeReasoning()
	s.closeText()
	toolIndexes := make([]int, 0, len(s.tools))
	for idx := range s.tools {
		toolIndexes = append(toolIndexes, idx)
	}
	sort.Ints(toolIndexes)
	for _, idx := range toolIndexes {
		tool := s.tools[idx]
		if tool.Announced && !tool.ArgsDone {
			tool.ArgsDone = true
			args := tool.Args.String()
			if !json.Valid([]byte(args)) {
				// A truncated upstream (e.g. finish_reason=length mid-arguments)
				// leaves partial JSON. Codex parses arguments strictly, so
				// degrade to an empty object rather than emitting broken JSON.
				args = "{}"
			}
			s.emit("response.function_call_arguments.done", map[string]interface{}{
				"item_id": tool.ItemID, "output_index": tool.OutputIndex,
				"arguments": args,
			})
			s.emit("response.output_item.done", map[string]interface{}{
				"output_index": tool.OutputIndex,
				"item": map[string]interface{}{
					"id": tool.ItemID, "type": "function_call",
					"call_id": tool.CallID, "name": tool.Name,
					"arguments": args, "status": "completed",
				},
			})
		}
	}

	status := "completed"
	eventType := "response.completed"
	var incomplete interface{}
	switch s.finishReason {
	case "length":
		status = "incomplete"
		eventType = "response.incomplete"
		incomplete = map[string]interface{}{"reason": "max_output_tokens"}
	case "content_filter":
		status = "incomplete"
		eventType = "response.incomplete"
		incomplete = map[string]interface{}{"reason": "content_filter"}
	}
	final := s.baseResponse(status)
	final["usage"] = map[string]interface{}{
		"input_tokens":  s.inputTokens,
		"output_tokens": s.outputTokens,
		"total_tokens":  s.inputTokens + s.outputTokens,
	}
	if incomplete != nil {
		final["incomplete_details"] = incomplete
	}
	final["output"] = s.finalOutputItems()
	s.emit(eventType, map[string]interface{}{"response": final})
	if s.inputTokens > 0 || s.outputTokens > 0 {
		store.SetTokenUsage(s.request, s.inputTokens, s.outputTokens)
	}
}

// finalOutputItems assembles the complete output array for the terminal
// response.completed / response.incomplete event, in ascending output_index
// order. Codex reconciles its transcript from this array — an empty output
// wipes everything that was streamed.
func (s *responsesFromChatStreamState) finalOutputItems() []map[string]interface{} {
	type indexed struct {
		idx  int
		item map[string]interface{}
	}
	var parts []indexed
	if s.sawReasoning {
		parts = append(parts, indexed{s.reasonIndex, map[string]interface{}{
			"id": s.reasonItemID, "type": "reasoning",
			"summary": []map[string]interface{}{{"type": "summary_text", "text": s.reasonText.String()}},
		}})
	}
	if s.sawText {
		parts = append(parts, indexed{s.textIndex, map[string]interface{}{
			"id": s.textItemID, "type": "message", "role": "assistant",
			"status":  "completed",
			"content": []map[string]interface{}{{"type": "output_text", "text": s.textBody.String()}},
		}})
	}
	toolIndexes := make([]int, 0, len(s.tools))
	for idx := range s.tools {
		toolIndexes = append(toolIndexes, idx)
	}
	sort.Ints(toolIndexes)
	for _, idx := range toolIndexes {
		tool := s.tools[idx]
		if !tool.Announced {
			continue
		}
		args := tool.Args.String()
		if !json.Valid([]byte(args)) {
			args = "{}"
		}
		parts = append(parts, indexed{tool.OutputIndex, map[string]interface{}{
			"id": tool.ItemID, "type": "function_call",
			"call_id": tool.CallID, "name": tool.Name,
			"arguments": args, "status": "completed",
		}})
	}
	sort.Slice(parts, func(i, j int) bool { return parts[i].idx < parts[j].idx })
	out := make([]map[string]interface{}, 0, len(parts))
	for _, p := range parts {
		out = append(out, p.item)
	}
	return out
}

// TranslateResponsesReqToChat converts an OpenAI Responses request body into an
// OpenAI Chat Completions body. Fields that have no chat equivalent
// (instructions become a system message; input items collapse into messages)
// are mapped conservatively so any chat-completions model can serve the call.
func TranslateResponsesReqToChat(body map[string]interface{}, apiModel string) (map[string]interface{}, error) {
	messages, err := responsesInputToChatMessages(body["input"])
	if err != nil {
		return nil, err
	}
	if instructions, ok := body["instructions"].(string); ok && instructions != "" {
		messages = append([]map[string]interface{}{{"role": "system", "content": instructions}}, messages...)
	}
	if len(messages) == 0 {
		return nil, fmt.Errorf("input is required")
	}

	out := map[string]interface{}{
		"model":    apiModel,
		"messages": messages,
	}
	if v := intFromJSON(body["max_output_tokens"]); v > 0 {
		out["max_tokens"] = v
	}
	if v, ok := body["temperature"].(json.Number); ok {
		f, _ := v.Float64()
		out["temperature"] = f
	} else if v, ok := body["temperature"].(float64); ok {
		out["temperature"] = v
	}
	if v, ok := body["top_p"].(json.Number); ok {
		f, _ := v.Float64()
		out["top_p"] = f
	} else if v, ok := body["top_p"].(float64); ok {
		out["top_p"] = v
	}
	tools, err := responsesToolsToChat(body["tools"])
	if err != nil {
		return nil, err
	}
	if len(tools) > 0 {
		out["tools"] = tools
		if choice, err := responsesToolChoiceToChat(body["tool_choice"]); err != nil {
			return nil, err
		} else if choice != nil {
			out["tool_choice"] = choice
		}
	}
	if raw, ok := body["text"].(map[string]interface{}); ok {
		if format, ok := raw["format"].(map[string]interface{}); ok {
			if rf, err := responsesTextFormatToChat(format); err != nil {
				return nil, err
			} else if rf != nil {
				out["response_format"] = rf
			}
		}
	}
	if v, ok := body["parallel_tool_calls"].(bool); ok {
		out["parallel_tool_calls"] = v
	}
	if stream, _ := body["stream"].(bool); stream {
		out["stream"] = true
		out["stream_options"] = map[string]interface{}{"include_usage": true}
	}
	return out, nil
}

// responsesInputToChatMessages accepts either a plain string (single user
// message) or the Responses input item array.
func responsesInputToChatMessages(input interface{}) ([]map[string]interface{}, error) {
	if input == nil {
		return nil, nil
	}
	if s, ok := input.(string); ok {
		if s == "" {
			return nil, nil
		}
		return []map[string]interface{}{{"role": "user", "content": s}}, nil
	}
	items, ok := input.([]interface{})
	if !ok {
		// Unrecognised input shape: treat as empty rather than failing.
		return nil, nil
	}
	messages := make([]map[string]interface{}, 0, len(items))
	for i, raw := range items {
		item, ok := raw.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("input[%d] must be an object", i)
		}
		// Items may be wrapped as {"type":"message","role":...,"content":[...]}
		// or bare {"role":...,"content":...}.
		typeName, _ := item["type"].(string)
		switch typeName {
		case "", "message":
			msg, err := responsesMessageToChat(item, i)
			if err != nil {
				return nil, err
			}
			if msg != nil {
				// An assistant text message following assistant tool_calls
				// belongs to the same turn: merge it instead of emitting two
				// consecutive assistant messages, which chat-completions
				// upstreams (Kimi) reject or mis-handle.
				merged := false
				if msg["role"] == "assistant" && len(messages) > 0 {
					last := messages[len(messages)-1]
					if last["role"] == "assistant" {
						if calls, has := last["tool_calls"]; has && calls != nil {
							if content := msg["content"]; content != nil && content != "" {
								last["content"] = content
							}
							merged = true
						}
					}
				}
				if !merged {
					messages = append(messages, msg)
				}
			}
		case "function_call":
			callID, _ := item["call_id"].(string)
			name, _ := item["name"].(string)
			args, _ := item["arguments"].(string)
			if callID == "" || name == "" {
				return nil, fmt.Errorf("input[%d]: function_call requires call_id and name", i)
			}
			if args == "" {
				args = "{}"
			}
			toolCall := map[string]interface{}{
				"id": callID, "type": "function",
				"function": map[string]interface{}{"name": name, "arguments": args},
			}
			// Merge into the preceding assistant message when possible:
			// chat-completions requires an assistant turn's text and
			// tool_calls to live on the same message. Two consecutive
			// assistant messages confuse upstreams (Kimi rejects or
			// mis-orders them), which is why Codex stalled after the first
			// tool round.
			merged := false
			if len(messages) > 0 {
				last := messages[len(messages)-1]
				if last["role"] == "assistant" {
					calls, _ := last["tool_calls"].([]map[string]interface{})
					last["tool_calls"] = append(calls, toolCall)
					if last["content"] == "" {
						last["content"] = nil
					}
					merged = true
				}
			}
			if !merged {
				messages = append(messages, map[string]interface{}{
					"role": "assistant", "content": nil,
					"tool_calls": []map[string]interface{}{toolCall},
				})
			}
		case "function_call_output":
			callID, _ := item["call_id"].(string)
			if callID == "" {
				return nil, fmt.Errorf("input[%d]: function_call_output requires call_id", i)
			}
			output := responsesOutputText(item["output"])
			messages = append(messages, map[string]interface{}{
				"role": "tool", "tool_call_id": callID, "content": output,
			})
		case "reasoning":
			// Reasoning replay is dropped: chat-completions models do not accept it.
		default:
			// Skip Responses-only item types (web_search_call, file_search_call,
			// computer_call, etc.) that chat-completions cannot represent.
			continue
		}
	}
	return messages, nil
}

func responsesMessageToChat(item map[string]interface{}, idx int) (map[string]interface{}, error) {
	role, _ := item["role"].(string)
	if role == "" {
		role = "user"
	}
	switch role {
	case "system", "developer":
		role = "system"
	case "user", "assistant", "tool":
	default:
		// Unknown role: coerce to "user" rather than rejecting the request.
		role = "user"
	}
	content, err := responsesContentToChat(item["content"], role)
	if err != nil {
		return nil, fmt.Errorf("input[%d].content: %w", idx, err)
	}
	if content == nil {
		return nil, nil
	}
	return map[string]interface{}{"role": role, "content": content}, nil
}

func responsesContentToChat(content interface{}, role string) (interface{}, error) {
	if content == nil {
		return nil, nil
	}
	if s, ok := content.(string); ok {
		return s, nil
	}
	parts, ok := content.([]interface{})
	if !ok {
		return nil, fmt.Errorf("must be a string or content-part array")
	}
	out := make([]map[string]interface{}, 0, len(parts))
	for i, raw := range parts {
		part, ok := raw.(map[string]interface{})
		if !ok {
			return nil, fmt.Errorf("part %d must be an object", i)
		}
		typeName, _ := part["type"].(string)
		switch typeName {
		case "input_text", "output_text", "text":
			text, _ := part["text"].(string)
			out = append(out, map[string]interface{}{"type": "text", "text": text})
		case "input_image":
			if role == "assistant" {
				// Assistant image content has no chat equivalent; skip it.
				continue
			}
			url, _ := part["image_url"].(string)
			if url == "" {
				continue
			}
			image := map[string]interface{}{"type": "image_url",
				"image_url": map[string]interface{}{"url": url}}
			if detail, ok := part["detail"].(string); ok && detail != "" {
				image["image_url"].(map[string]interface{})["detail"] = detail
			}
			out = append(out, image)
		case "refusal":
			text, _ := part["refusal"].(string)
			out = append(out, map[string]interface{}{"type": "text", "text": text})
		default:
			// Skip content types with no chat-completions equivalent
			// (input_file, input_audio, etc.) rather than failing the request.
			continue
		}
	}
	// A single text part collapses to a plain string for maximum compatibility
	// with upstream chat-completions implementations.
	if len(out) == 1 && out[0]["type"] == "text" {
		return out[0]["text"], nil
	}
	return out, nil
}

func responsesOutputText(output interface{}) string {
	switch v := output.(type) {
	case string:
		return v
	case []interface{}:
		var sb []byte
		for _, raw := range v {
			if part, ok := raw.(map[string]interface{}); ok {
				if text, ok := part["text"].(string); ok {
					sb = append(sb, text...)
				}
			}
		}
		return string(sb)
	}
	return ""
}

// responsesToolsToChat flattens Responses tools ({"type":"function","name":...})
// into the chat shape ({"type":"function","function":{...}}).
func responsesToolsToChat(raw interface{}) ([]map[string]interface{}, error) {
	if raw == nil {
		return nil, nil
	}
	tools, ok := raw.([]interface{})
	if !ok {
		return nil, nil
	}
	out := make([]map[string]interface{}, 0, len(tools))
	for _, item := range tools {
		tool, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		typeName, _ := tool["type"].(string)
		if typeName != "function" {
			// Skip built-in tool types (namespace, web_search, local_shell, etc.)
			// that the chat-completions upstream cannot accept.
			continue
		}
		name, _ := tool["name"].(string)
		if name == "" {
			continue
		}
		fn := map[string]interface{}{"name": name}
		if desc, ok := tool["description"].(string); ok {
			fn["description"] = desc
		}
		if params, ok := tool["parameters"]; ok && params != nil {
			fn["parameters"] = params
		} else {
			fn["parameters"] = map[string]interface{}{"type": "object", "properties": map[string]interface{}{}}
		}
		if strict, ok := tool["strict"].(bool); ok {
			fn["strict"] = strict
		}
		out = append(out, map[string]interface{}{"type": "function", "function": fn})
	}
	return out, nil
}

func responsesToolChoiceToChat(raw interface{}) (interface{}, error) {
	if raw == nil {
		return nil, nil
	}
	if s, ok := raw.(string); ok {
		switch s {
		case "auto", "none", "required":
			return s, nil
		default:
			// Unknown string form: fall back to "auto" rather than failing.
			return "auto", nil
		}
	}
	obj, ok := raw.(map[string]interface{})
	if !ok {
		return nil, nil
	}
	typeName, _ := obj["type"].(string)
	if typeName != "function" {
		// Responses-only tool_choice shapes (allowed_tools, etc.) have no chat
		// equivalent — omit tool_choice rather than rejecting the request.
		return nil, nil
	}
	name, _ := obj["name"].(string)
	if name == "" {
		return nil, nil
	}
	return map[string]interface{}{"type": "function", "function": map[string]interface{}{"name": name}}, nil
}

func responsesTextFormatToChat(format map[string]interface{}) (map[string]interface{}, error) {
	typeName, _ := format["type"].(string)
	switch typeName {
	case "", "text":
		return nil, nil
	case "json_object":
		return map[string]interface{}{"type": "json_object"}, nil
	case "json_schema":
		name, _ := format["name"].(string)
		schema, ok := format["schema"]
		if name == "" || !ok || schema == nil {
			// Malformed json_schema: degrade to plain text rather than failing.
			return nil, nil
		}
		payload := map[string]interface{}{"name": name, "schema": schema}
		if desc, ok := format["description"].(string); ok && desc != "" {
			payload["description"] = desc
		}
		if strict, ok := format["strict"].(bool); ok {
			payload["strict"] = strict
		}
		return map[string]interface{}{"type": "json_schema", "json_schema": payload}, nil
	default:
		// Unknown format type: degrade to plain text rather than failing.
		return nil, nil
	}
}
