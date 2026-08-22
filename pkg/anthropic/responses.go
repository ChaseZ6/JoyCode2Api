package anthropic

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/common"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

// This file bridges the Anthropic Messages API to the upstream OpenAI Responses
// API, so models whose adapterType is "openai-response" (e.g. "GPT-5.6 Sol") can
// be driven by Anthropic clients such as Claude Code and Claude Desktop.
//
// Those models reject both /chat/completions and /anthropic/v1/messages; the
// Responses endpoint is the only one that serves them, and it uses a different
// request shape ("input" instead of "messages", "max_output_tokens" instead of
// "max_tokens") and a different SSE event vocabulary.

// TranslateAnthropicToResponses converts an Anthropic Messages request into an
// OpenAI Responses API body targeting the given model.
func TranslateAnthropicToResponses(req *MessageRequest, model joycode.ModelInfo) map[string]interface{} {
	body := map[string]interface{}{
		"model": model.APIName(),
		"input": buildResponsesInput(req),
	}
	if req.MaxTokens > 0 {
		body["max_output_tokens"] = req.MaxTokens
	}
	if sys := responsesInstructions(req.System); sys != "" {
		body["instructions"] = sys
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	if tools := convertToolsToResponses(req.Tools); len(tools) > 0 {
		body["tools"] = tools
		if tc, parallel := convertToolChoiceToResponses(req.ToolChoice); tc != nil {
			body["tool_choice"] = tc
			if parallel != nil {
				body["parallel_tool_calls"] = *parallel
			}
		}
	}
	if req.OutputConfig != nil && req.OutputConfig.Effort != "" {
		if effort, ok := normalizeResponsesEffort(req.OutputConfig.Effort); ok {
			body["reasoning"] = map[string]interface{}{"effort": effort}
		}
	}
	return body
}

func normalizeResponsesEffort(effort string) (string, bool) {
	effort = strings.ToLower(strings.TrimSpace(effort))
	if effort == "extra" {
		return "xhigh", true
	}
	switch effort {
	case "none", "low", "medium", "high", "xhigh", "max":
		return effort, true
	default:
		return "", false
	}
}

// responsesInstructions flattens an Anthropic system prompt (string or block
// array) into the Responses API "instructions" string.
func responsesInstructions(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &blocks); err == nil {
		var sb strings.Builder
		for _, b := range blocks {
			if b.Text == "" {
				continue
			}
			if sb.Len() > 0 {
				sb.WriteString("\n\n")
			}
			sb.WriteString(b.Text)
		}
		return sb.String()
	}
	return ""
}

// buildResponsesInput converts Anthropic messages into Responses input items.
// Tool calls and tool results become top-level function_call /
// function_call_output items rather than nested content blocks.
func buildResponsesInput(req *MessageRequest) []map[string]interface{} {
	// Collect every tool_result call_id up front so a tool_use whose result was
	// removed by truncation (or interrupted by the user) can be downgraded to
	// text instead of emitting a function_call with no function_call_output —
	// a protocol-incomplete sequence Responses implementations may reject.
	answeredCalls := map[string]bool{}
	for _, msg := range req.Messages {
		if msg.Role != "user" {
			continue
		}
		var blocks []contentBlock
		if json.Unmarshal(msg.Content, &blocks) != nil {
			continue
		}
		for _, b := range blocks {
			if b.Type == "tool_result" && b.ToolUseID != "" {
				answeredCalls[b.ToolUseID] = true
			}
		}
	}

	items := make([]map[string]interface{}, 0, len(req.Messages))
	for _, msg := range req.Messages {
		// Plain string content is the common case.
		var text string
		if err := json.Unmarshal(msg.Content, &text); err == nil {
			if text != "" {
				items = append(items, map[string]interface{}{
					"role": msg.Role, "content": text,
				})
			}
			continue
		}

		var blocks []contentBlock
		if err := json.Unmarshal(msg.Content, &blocks); err != nil {
			continue
		}

		var parts []map[string]interface{}
		for _, b := range blocks {
			switch b.Type {
			case "text":
				if b.Text != "" {
					parts = append(parts, map[string]interface{}{
						"type": responsesTextType(msg.Role), "text": b.Text,
					})
				}
			case "image":
				if url := imageBlockToDataURL(b.Source); url != "" {
					parts = append(parts, map[string]interface{}{
						"type": "input_image", "image_url": url,
					})
				}
			case "tool_use":
				args := "{}"
				if len(b.Input) > 0 {
					args = string(b.Input)
				}
				// A tool_use with no matching tool_result in the retained
				// conversation would emit a function_call with no output.
				// Downgrade it to a text note to keep the sequence valid.
				if !answeredCalls[b.ID] {
					parts = append(parts, map[string]interface{}{
						"type": responsesTextType(msg.Role),
						"text": "[previous tool call: " + b.Name + "(" + args + ") — result omitted]",
					})
					continue
				}
				// Flush accumulated text before the call to preserve ordering.
				if len(parts) > 0 {
					items = append(items, map[string]interface{}{
						"role": msg.Role, "content": parts,
					})
					parts = nil
				}
				items = append(items, map[string]interface{}{
					"type":      "function_call",
					"call_id":   b.ID,
					"name":      b.Name,
					"arguments": args,
				})
			case "tool_result":
				if len(parts) > 0 {
					items = append(items, map[string]interface{}{
						"role": msg.Role, "content": parts,
					})
					parts = nil
				}
				items = append(items, map[string]interface{}{
					"type":    "function_call_output",
					"call_id": b.ToolUseID,
					"output":  extractToolResultContent(b.Content),
				})
			}
		}
		if len(parts) > 0 {
			items = append(items, map[string]interface{}{
				"role": msg.Role, "content": parts,
			})
		}
	}
	return items
}

// responsesTextType picks the content-part type: assistant turns carry
// output_text, user/system turns carry input_text.
func responsesTextType(role string) string {
	if role == "assistant" {
		return "output_text"
	}
	return "input_text"
}

// convertToolsToResponses maps Anthropic tool definitions to Responses function
// tools. Anthropic server-side builtins (web_search etc.) have no Responses
// equivalent and are dropped.
func convertToolsToResponses(tools []Tool) []map[string]interface{} {
	out := make([]map[string]interface{}, 0, len(tools))
	for _, t := range tools {
		if t.Name == "" || isBuiltinTool(t) {
			continue
		}
		params := json.RawMessage(`{"type":"object","properties":{}}`)
		if len(t.InputSchema) > 0 {
			params = t.InputSchema
		}
		out = append(out, map[string]interface{}{
			"type":        "function",
			"name":        t.Name,
			"description": t.Description,
			"parameters":  params,
		})
	}
	return out
}

// convertToolChoiceToResponses maps Anthropic tool_choice to the Responses form.
func convertToolChoiceToResponses(raw json.RawMessage) (interface{}, *bool) {
	if len(raw) == 0 {
		return nil, nil
	}
	var tc struct {
		Type                   string `json:"type"`
		Name                   string `json:"name"`
		DisableParallelToolUse *bool  `json:"disable_parallel_tool_use"`
	}
	if err := json.Unmarshal(raw, &tc); err != nil {
		return nil, nil
	}
	var parallel *bool
	if tc.DisableParallelToolUse != nil {
		value := !*tc.DisableParallelToolUse
		parallel = &value
	}
	switch tc.Type {
	case "any":
		return "required", parallel
	case "auto":
		return "auto", parallel
	case "none":
		return "none", parallel
	case "tool":
		if tc.Name != "" {
			return map[string]interface{}{"type": "function", "name": tc.Name}, parallel
		}
	}
	return nil, parallel
}

// --- Non-streaming ---

// responsesObject is the subset of the Responses API response we consume.
type responsesObject struct {
	ID     string `json:"id"`
	Model  string `json:"model"`
	Status string `json:"status"`
	Output []struct {
		Type    string `json:"type"`
		Name    string `json:"name"`
		CallID  string `json:"call_id"`
		Args    string `json:"arguments"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	} `json:"output"`
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
	Error json.RawMessage `json:"error"`
}

// TranslateResponsesToAnthropic converts a Responses API object into an
// Anthropic MessageResponse.
func TranslateResponsesToAnthropic(raw map[string]interface{}, displayModel string) (*MessageResponse, error) {
	b, err := json.Marshal(raw)
	if err != nil {
		return nil, err
	}
	var ro responsesObject
	if err := json.Unmarshal(b, &ro); err != nil {
		return nil, err
	}
	if len(ro.Error) > 0 && string(ro.Error) != "null" {
		return nil, fmt.Errorf("upstream response failed: %s", responsesErrorMessage(ro.Error))
	}
	if ro.Status == "failed" {
		return nil, fmt.Errorf("upstream response failed without error details")
	}
	if ro.Status == "incomplete" {
		reason := ""
		if ro.IncompleteDetails != nil {
			reason = ro.IncompleteDetails.Reason
		}
		if reason != "max_output_tokens" {
			return nil, fmt.Errorf("upstream response incomplete: %s", reason)
		}
	}

	content := make([]ContentBlock, 0, len(ro.Output))
	stopReason := "end_turn"
	for _, item := range ro.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				if c.Type == "output_text" && c.Text != "" {
					content = append(content, ContentBlock{Type: "text", Text: c.Text})
				}
			}
		case "function_call":
			var input interface{}
			if item.Args != "" {
				if err := json.Unmarshal([]byte(item.Args), &input); err != nil {
					input = map[string]interface{}{}
				}
			} else {
				input = map[string]interface{}{}
			}
			id := item.CallID
			if id == "" {
				id = NewToolUseID()
			}
			content = append(content, ContentBlock{
				Type: "tool_use", ID: id, Name: item.Name, Input: input,
			})
			stopReason = "tool_use"
		}
	}
	if ro.IncompleteDetails != nil && ro.IncompleteDetails.Reason == "max_output_tokens" {
		stopReason = "max_tokens"
	}
	if len(content) == 0 {
		content = append(content, ContentBlock{Type: "text", Text: ""})
	}

	id := ro.ID
	if id == "" {
		id = NewMessageID()
	}
	return &MessageResponse{
		ID:         id,
		Type:       "message",
		Role:       "assistant",
		Model:      displayModel,
		Content:    content,
		StopReason: strPtr(stopReason),
		Usage: Usage{
			InputTokens:  ro.Usage.InputTokens,
			OutputTokens: ro.Usage.OutputTokens,
		},
	}, nil
}

func (h *Handler) handleResponsesNonStream(w http.ResponseWriter, r *http.Request, req *MessageRequest, client *joycode.Client, model joycode.ModelInfo) {
	body := TranslateAnthropicToResponses(req, model)
	logRequestDetails(r, "translated responses request (non-stream)", body)

	raw, err := client.PostResponsesContext(r.Context(), body)
	if err != nil {
		reqLog(r).Error("responses non-stream upstream error", "model", model.APIName(), "error", err)
		writeAnthropicError(w, 500, "上游 Responses 接口调用失败: "+err.Error())
		return
	}
	if errField, ok := raw["error"]; ok && errField != nil {
		if msg := responsesErrorMessage(errField); msg != "" {
			reqLog(r).Error("responses non-stream upstream reported error", "error", msg)
			writeAnthropicError(w, 502, "上游 Responses 接口返回错误: "+msg)
			return
		}
	}

	resp, err := TranslateResponsesToAnthropic(raw, model.DisplayName())
	if err != nil {
		if usage, ok := raw["usage"].(map[string]interface{}); ok {
			store.SetTokenUsage(r, intValue(usage["input_tokens"]), intValue(usage["output_tokens"]))
		}
		store.SetRequestError(r, http.StatusBadGateway, "upstream Responses status error: "+err.Error())
		reqLog(r).Error("translate responses object", "error", err)
		writeAnthropicError(w, 502, "解析上游 Responses 响应失败: "+err.Error())
		return
	}
	store.SetTokenUsage(r, resp.Usage.InputTokens, resp.Usage.OutputTokens)
	writeAnthropicJSON(w, 200, resp)
}

func intValue(v interface{}) int {
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	case json.Number:
		value, _ := n.Int64()
		return int(value)
	}
	return 0
}

// responsesErrorMessage extracts a human-readable message from an upstream error
// field, which may be a string or an object.
func responsesErrorMessage(v interface{}) string {
	switch e := v.(type) {
	case string:
		return e
	case map[string]interface{}:
		if m, ok := e["message"].(string); ok {
			return m
		}
		if m, ok := e["cause"].(string); ok {
			return m
		}
		b, _ := json.Marshal(e)
		return string(b)
	}
	return ""
}

// --- Streaming ---

// responsesStreamEvent is the subset of Responses SSE payloads we consume.
type responsesStreamEvent struct {
	Type      string      `json:"type"`
	Delta     string      `json:"delta"`
	ItemID    string      `json:"item_id"`
	CallID    string      `json:"call_id"`
	OutputIdx int         `json:"output_index"`
	Error     interface{} `json:"error"`
	Item      struct {
		ID     string `json:"id"`
		Type   string `json:"type"`
		Name   string `json:"name"`
		CallID string `json:"call_id"`
		Args   string `json:"arguments"`
	} `json:"item"`
	Response struct {
		ID    string      `json:"id"`
		Error interface{} `json:"error"`
		Usage *struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
		IncompleteDetails *struct {
			Reason string `json:"reason"`
		} `json:"incomplete_details"`
	} `json:"response"`
}

// handleResponsesStream drives a Responses-API model and re-emits its SSE stream
// in Anthropic Messages format.
func (h *Handler) handleResponsesStream(w http.ResponseWriter, r *http.Request, req *MessageRequest, client *joycode.Client, flusher http.Flusher, model joycode.ModelInfo) {
	body := TranslateAnthropicToResponses(req, model)
	body["stream"] = true
	logRequestDetails(r, "translated responses request (stream)", body)

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(200)

	// Heartbeat while waiting for the upstream to connect. The Responses
	// upstream can take tens of seconds (occasionally minutes) to return its
	// first byte; without keepalives, clients with a short idle-timeout
	// (notably Codex) assume the connection is dead and reconnect. SSE comment
	// lines are ignored by event parsers, so they cannot corrupt the stream.
	// Mirrors the heartbeat handleStream already runs for the chat path.
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
				fmt.Fprint(w, ": keepalive\n\n")
				flusher.Flush()
			}
		}
	}()

	resp, err := client.PostResponsesStreamContext(r.Context(), body)
	close(stopHeartbeat)
	<-heartbeatDone
	if err != nil {
		reqLog(r).Error("responses stream connect failed", "model", model.APIName(), "error", err)
		writeStreamError(w, flusher, "上游 Responses 接口调用失败: "+err.Error())
		return
	}
	defer resp.Body.Close()

	msgID := NewMessageID()
	st := &responsesStreamState{
		w: w, flusher: flusher, msgID: msgID, displayModel: model.DisplayName(),
		toolsByItem: make(map[string]*anthropicToolStream),
		toolsByCall: make(map[string]*anthropicToolStream),
	}
	st.emitMessageStart(req)

	for read := range common.ReadSSE(r.Context(), resp.Body) {
		if read.Err != nil {
			reqLog(r).Error("responses stream read error", "error", read.Err)
			st.closeText()
			st.closeAllTools()
			store.SetRequestError(r, http.StatusBadGateway, "stream_read_error: "+read.Err.Error())
			writeStreamError(w, flusher, "读取上游流式响应失败，本次回复可能不完整，请重试。原始错误: "+read.Err.Error())
			return
		}
		if read.Event.Done {
			break
		}
		data := common.UnwrapSSEData(read.Event.Data)
		if data == "" || !strings.HasPrefix(data, "{") {
			continue
		}
		var ev responsesStreamEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			continue
		}
		st.handleEvent(&ev)
		if st.streamError != "" {
			st.closeText()
			st.closeAllTools()
			store.SetRequestError(r, http.StatusBadGateway, "upstream_error: "+st.streamError)
			reqLog(r).Error("responses stream upstream failure", "type", st.terminal, "error", st.streamError)
			writeStreamError(w, flusher, st.streamError)
			return
		}
		if st.sawTerminal {
			st.finish("end_turn")
			if st.inTokens > 0 || st.outTokens > 0 {
				store.SetTokenUsage(r, st.inTokens, st.outTokens)
			}
			reqLog(r).Info("responses stream done", "model", model.APIName(), "blocks", st.nextIndex, "in", st.inTokens, "out", st.outTokens)
			return
		}
	}

	if !st.sawTerminal {
		st.closeText()
		st.closeAllTools()
		message := "上游 Responses 流在返回结束标记前断开，本次回复可能不完整，请重试。"
		store.SetRequestError(r, http.StatusBadGateway, "stream_truncated: upstream closed before terminal event")
		reqLog(r).Warn("responses stream closed before terminal event", "blocks", st.nextIndex)
		writeStreamError(w, flusher, message)
		return
	}

	st.finish("end_turn")
	if st.inTokens > 0 || st.outTokens > 0 {
		store.SetTokenUsage(r, st.inTokens, st.outTokens)
	}
	reqLog(r).Info("responses stream done", "model", model.APIName(), "blocks", st.nextIndex, "in", st.inTokens, "out", st.outTokens)
}

type anthropicToolStream struct {
	index  int
	callID string
	name   string
	open   bool
}

// responsesStreamState tracks the Anthropic-side block indices and open blocks
// while translating a Responses stream.
type responsesStreamState struct {
	w            http.ResponseWriter
	flusher      http.Flusher
	msgID        string
	displayModel string

	nextIndex   int
	textOpen    bool
	textIndex   int
	toolsByItem map[string]*anthropicToolStream
	toolsByCall map[string]*anthropicToolStream
	stopReason  string
	finished    bool
	sawTerminal bool
	terminal    string
	streamError string
	inTokens    int
	outTokens   int
}

func (s *responsesStreamState) send(event string, payload interface{}) {
	b, err := json.Marshal(payload)
	if err != nil {
		return
	}
	fmt.Fprintf(s.w, "event: %s\n", event)
	fmt.Fprintf(s.w, "data: %s\n\n", b)
	s.flusher.Flush()
}

func (s *responsesStreamState) emitMessageStart(req *MessageRequest) {
	s.send("message_start", sseMessageStart{
		Type: "message_start",
		Message: MessageResponse{
			ID: s.msgID, Type: "message", Role: "assistant",
			Model: s.displayModel, Content: []ContentBlock{},
			Usage: Usage{InputTokens: 0, OutputTokens: 0},
		},
	})
	s.send("ping", ssePing{Type: "ping"})
}

func (s *responsesStreamState) handleEvent(ev *responsesStreamEvent) {
	switch ev.Type {
	case "response.output_text.delta":
		if ev.Delta == "" {
			return
		}
		if !s.textOpen {
			s.textIndex = s.nextIndex
			s.nextIndex++
			s.textOpen = true
			s.send("content_block_start", sseContentBlockStart{
				Type: "content_block_start", Index: s.textIndex,
				ContentBlock: ContentBlock{Type: "text", Text: ""},
			})
		}
		s.send("content_block_delta", sseContentBlockDelta{
			Type: "content_block_delta", Index: s.textIndex,
			Delta: deltaText{Type: "text_delta", Text: ev.Delta},
		})

	case "response.output_item.added":
		if ev.Item.Type != "function_call" {
			return
		}
		// Anthropic requires blocks to be strictly sequential: a started block
		// must be stopped before the next begins. If a text block is still open
		// (model emitted text before the tool call — very common), close it
		// first so the tool block does not interleave with it.
		s.closeText()
		itemID := ev.Item.ID
		callID := ev.Item.CallID
		if callID == "" {
			callID = NewToolUseID()
		}
		tool := &anthropicToolStream{index: s.nextIndex, callID: callID, name: ev.Item.Name, open: true}
		s.nextIndex++
		if itemID != "" {
			s.toolsByItem[itemID] = tool
		}
		s.toolsByCall[callID] = tool
		s.stopReason = "tool_use"
		s.send("content_block_start", sseContentBlockStart{
			Type: "content_block_start", Index: tool.index,
			ContentBlock: ContentBlock{
				Type: "tool_use", ID: callID, Name: ev.Item.Name,
				Input: map[string]interface{}{},
			},
		})

	case "response.function_call_arguments.delta":
		if ev.Delta == "" {
			return
		}
		tool := s.findTool(ev.ItemID, ev.CallID)
		if tool == nil || !tool.open {
			s.streamError = "cannot route function arguments delta to a tool call"
			return
		}
		s.send("content_block_delta", sseContentBlockDelta{
			Type: "content_block_delta", Index: tool.index,
			Delta: deltaText{Type: "input_json_delta", PartialJSON: ev.Delta},
		})

	case "response.output_item.done":
		if ev.Item.Type == "function_call" {
			s.stopReason = "tool_use"
			tool := s.findTool(ev.Item.ID, ev.Item.CallID)
			if tool == nil {
				tool = s.findTool(ev.ItemID, ev.CallID)
			}
			if tool != nil {
				s.closeTool(tool)
			}
		}

	case "response.completed":
		if u := ev.Response.Usage; u != nil {
			s.inTokens = u.InputTokens
			s.outTokens = u.OutputTokens
		}
		s.sawTerminal = true
		s.terminal = ev.Type

	case "response.incomplete":
		if u := ev.Response.Usage; u != nil {
			s.inTokens = u.InputTokens
			s.outTokens = u.OutputTokens
		}
		s.terminal = ev.Type
		reason := ""
		if d := ev.Response.IncompleteDetails; d != nil {
			reason = d.Reason
		}
		if reason == "max_output_tokens" {
			s.stopReason = "max_tokens"
			s.sawTerminal = true
		} else {
			s.streamError = "upstream response incomplete: " + reason
		}

	case "response.failed", "error":
		if u := ev.Response.Usage; u != nil {
			s.inTokens = u.InputTokens
			s.outTokens = u.OutputTokens
		}
		s.terminal = ev.Type
		errValue := ev.Response.Error
		if ev.Type == "error" {
			errValue = ev.Error
		}
		s.streamError = responsesErrorMessage(errValue)
		if s.streamError == "" || s.streamError == "null" {
			s.streamError = "upstream response failed"
		}
	}
}

func (s *responsesStreamState) closeText() {
	if !s.textOpen {
		return
	}
	s.send("content_block_stop", sseContentBlockStop{
		Type: "content_block_stop", Index: s.textIndex,
	})
	s.textOpen = false
}

func (s *responsesStreamState) findTool(itemID, callID string) *anthropicToolStream {
	if itemID != "" {
		if tool := s.toolsByItem[itemID]; tool != nil {
			return tool
		}
	}
	if callID != "" {
		return s.toolsByCall[callID]
	}
	var only *anthropicToolStream
	for _, tool := range s.toolsByCall {
		if !tool.open {
			continue
		}
		if only != nil && only != tool {
			return nil
		}
		only = tool
	}
	return only
}

func (s *responsesStreamState) closeTool(tool *anthropicToolStream) {
	if tool == nil || !tool.open {
		return
	}
	s.send("content_block_stop", sseContentBlockStop{Type: "content_block_stop", Index: tool.index})
	tool.open = false
}

func (s *responsesStreamState) closeAllTools() {
	byIndex := make(map[int]*anthropicToolStream)
	for _, tool := range s.toolsByCall {
		if tool.open {
			byIndex[tool.index] = tool
		}
	}
	for index := 0; index < s.nextIndex; index++ {
		if tool := byIndex[index]; tool != nil {
			s.closeTool(tool)
		}
	}
}

// finish closes any open blocks and emits message_delta + message_stop exactly
// once.
func (s *responsesStreamState) finish(defaultReason string) {
	if s.finished {
		return
	}
	s.finished = true
	s.closeText()
	s.closeAllTools()

	reason := s.stopReason
	if reason == "" {
		reason = defaultReason
	}
	md := sseMessageDelta{
		Type:  "message_delta",
		Delta: deltaStop{StopReason: reason},
	}
	md.Usage.OutputTokens = s.outTokens
	s.send("message_delta", md)
	s.send("message_stop", sseMessageStop{Type: "message_stop"})
}
