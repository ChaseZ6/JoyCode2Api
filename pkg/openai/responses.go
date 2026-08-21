package openai

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

// This file bridges the OpenAI Chat Completions API to the upstream OpenAI
// Responses API, for models whose adapterType is "openai-response" (e.g.
// "GPT-5.6 Sol"). Those models reject /chat/completions with upstream error 1032.

// TranslateChatToResponses converts an OpenAI chat request into a Responses body.
func TranslateChatToResponses(req *ChatRequest, model joycode.ModelInfo) (map[string]interface{}, error) {
	input, err := chatMessagesToResponsesInput(req.Messages)
	if err != nil {
		return nil, err
	}
	body := map[string]interface{}{
		"model": model.APIName(),
		"input": input,
	}
	if req.MaxTokens > 0 {
		body["max_output_tokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	tools, err := chatToolsToResponses(req.Tools)
	if err != nil {
		return nil, err
	}
	if len(tools) > 0 {
		body["tools"] = tools
		if len(req.ToolChoice) > 0 {
			choice, err := chatToolChoiceToResponses(req.ToolChoice)
			if err != nil {
				return nil, err
			}
			body["tool_choice"] = choice
		}
	}
	if req.ReasoningEffort != "" {
		switch req.ReasoningEffort {
		case "low", "medium", "high":
			body["reasoning"] = map[string]interface{}{"effort": req.ReasoningEffort}
		default:
			return nil, fmt.Errorf("reasoning_effort must be low, medium, or high")
		}
	}
	if req.ParallelToolCalls != nil {
		body["parallel_tool_calls"] = *req.ParallelToolCalls
	}
	if len(req.ResponseFormat) > 0 {
		format, err := chatResponseFormatToResponses(req.ResponseFormat)
		if err != nil {
			return nil, err
		}
		body["text"] = map[string]interface{}{"format": format}
	}
	return body, nil
}

// chatMessage is the subset of an OpenAI message we need to remap.
type chatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCallID string          `json:"tool_call_id"`
	ToolCalls  []struct {
		ID       string `json:"id"`
		Function struct {
			Name string `json:"name"`
			Args string `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls"`
}

// chatMessagesToResponsesInput remaps chat messages to Responses input items.
// System messages become "developer" role; tool calls and results become
// top-level function_call / function_call_output items.
func chatMessagesToResponsesInput(raw json.RawMessage) ([]map[string]interface{}, error) {
	if len(raw) == 0 {
		return nil, fmt.Errorf("messages is required")
	}
	var msgs []chatMessage
	if err := json.Unmarshal(raw, &msgs); err != nil {
		return nil, fmt.Errorf("messages must be an array: %w", err)
	}
	if len(msgs) == 0 {
		return nil, fmt.Errorf("messages must not be empty")
	}
	items := make([]map[string]interface{}, 0, len(msgs))
	for i, m := range msgs {
		if m.Role == "tool" {
			if m.ToolCallID == "" {
				return nil, fmt.Errorf("messages[%d].tool_call_id is required", i)
			}
			output, err := contentToTextStrict(m.Content)
			if err != nil {
				return nil, fmt.Errorf("messages[%d].content: %w", i, err)
			}
			items = append(items, map[string]interface{}{
				"type": "function_call_output", "call_id": m.ToolCallID, "output": output,
			})
			continue
		}
		role := m.Role
		if role == "system" {
			role = "developer"
		}
		switch role {
		case "user", "developer", "assistant":
		default:
			return nil, fmt.Errorf("messages[%d].role %q is not supported", i, m.Role)
		}
		if len(m.Content) > 0 && string(m.Content) != "null" {
			content, err := contentToResponses(m.Content, role)
			if err != nil {
				return nil, fmt.Errorf("messages[%d].content: %w", i, err)
			}
			if content != nil {
				items = append(items, map[string]interface{}{"role": role, "content": content})
			}
		}
		for j, tc := range m.ToolCalls {
			if tc.ID == "" || tc.Function.Name == "" {
				return nil, fmt.Errorf("messages[%d].tool_calls[%d] requires id and function.name", i, j)
			}
			args := tc.Function.Args
			if args == "" {
				args = "{}"
			}
			items = append(items, map[string]interface{}{
				"type": "function_call", "call_id": tc.ID,
				"name": tc.Function.Name, "arguments": args,
			})
		}
	}
	return items, nil
}

func contentToResponses(raw json.RawMessage, role string) (interface{}, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, nil
	}
	var parts []json.RawMessage
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, fmt.Errorf("must be a string or content-part array")
	}
	out := make([]map[string]interface{}, 0, len(parts))
	for i, rawPart := range parts {
		var part map[string]interface{}
		if err := json.Unmarshal(rawPart, &part); err != nil {
			return nil, fmt.Errorf("part %d is invalid JSON", i)
		}
		typeName, _ := part["type"].(string)
		switch typeName {
		case "text", "input_text", "output_text":
			text, _ := part["text"].(string)
			kind := "input_text"
			if role == "assistant" {
				kind = "output_text"
			}
			out = append(out, map[string]interface{}{"type": kind, "text": text})
		case "image_url", "input_image":
			if role == "assistant" {
				return nil, fmt.Errorf("part %d: assistant image content is not supported", i)
			}
			url, detail := imagePart(part)
			if url == "" {
				return nil, fmt.Errorf("part %d: image_url.url is required", i)
			}
			image := map[string]interface{}{"type": "input_image", "image_url": url}
			if detail != "" {
				image["detail"] = detail
			}
			out = append(out, image)
		default:
			return nil, fmt.Errorf("part %d: unsupported content type %q", i, typeName)
		}
	}
	return out, nil
}

func imagePart(part map[string]interface{}) (url, detail string) {
	switch image := part["image_url"].(type) {
	case string:
		url = image
	case map[string]interface{}:
		url, _ = image["url"].(string)
		detail, _ = image["detail"].(string)
	}
	if url == "" {
		url, _ = part["image_url"].(string)
	}
	if detail == "" {
		detail, _ = part["detail"].(string)
	}
	return
}

func contentToTextStrict(raw json.RawMessage) (string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return "", nil
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return text, nil
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err != nil {
		return "", fmt.Errorf("must be a string or text-part array")
	}
	var sb strings.Builder
	for i, part := range parts {
		if part.Type != "text" && part.Type != "input_text" && part.Type != "output_text" {
			return "", fmt.Errorf("part %d: tool output type %q is not supported", i, part.Type)
		}
		if sb.Len() > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString(part.Text)
	}
	return sb.String(), nil
}

// chatToolsToResponses maps OpenAI chat tools to Responses function tools, which
// are flat rather than nested under "function".
func chatToolsToResponses(raw json.RawMessage) ([]map[string]interface{}, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var tools []struct {
		Type     string `json:"type"`
		Function struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Parameters  json.RawMessage `json:"parameters"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &tools); err != nil {
		return nil, fmt.Errorf("tools must be an array: %w", err)
	}
	out := make([]map[string]interface{}, 0, len(tools))
	for i, tool := range tools {
		if tool.Type != "" && tool.Type != "function" {
			return nil, fmt.Errorf("tools[%d].type %q is not supported", i, tool.Type)
		}
		if tool.Function.Name == "" {
			return nil, fmt.Errorf("tools[%d].function.name is required", i)
		}
		params := json.RawMessage(`{"type":"object","properties":{}}`)
		if len(tool.Function.Parameters) > 0 {
			params = tool.Function.Parameters
		}
		out = append(out, map[string]interface{}{
			"type": "function", "name": tool.Function.Name,
			"description": tool.Function.Description, "parameters": params,
		})
	}
	return out, nil
}

func chatToolChoiceToResponses(raw json.RawMessage) (interface{}, error) {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		switch text {
		case "auto", "none", "required":
			return text, nil
		default:
			return nil, fmt.Errorf("tool_choice %q is not supported", text)
		}
	}
	var choice struct {
		Type     string `json:"type"`
		Name     string `json:"name"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &choice); err != nil {
		return nil, fmt.Errorf("tool_choice is invalid: %w", err)
	}
	if choice.Type != "function" {
		return nil, fmt.Errorf("tool_choice.type %q is not supported", choice.Type)
	}
	name := choice.Name
	if name == "" {
		name = choice.Function.Name
	}
	if name == "" {
		return nil, fmt.Errorf("tool_choice.function.name is required")
	}
	return map[string]interface{}{"type": "function", "name": name}, nil
}

func chatResponseFormatToResponses(raw json.RawMessage) (map[string]interface{}, error) {
	var format struct {
		Type       string `json:"type"`
		JSONSchema struct {
			Name        string          `json:"name"`
			Description string          `json:"description"`
			Schema      json.RawMessage `json:"schema"`
			Strict      *bool           `json:"strict"`
		} `json:"json_schema"`
	}
	if err := json.Unmarshal(raw, &format); err != nil {
		return nil, fmt.Errorf("response_format is invalid: %w", err)
	}
	switch format.Type {
	case "text", "json_object":
		return map[string]interface{}{"type": format.Type}, nil
	case "json_schema":
		if format.JSONSchema.Name == "" || len(format.JSONSchema.Schema) == 0 {
			return nil, fmt.Errorf("response_format.json_schema requires name and schema")
		}
		out := map[string]interface{}{
			"type": "json_schema", "name": format.JSONSchema.Name,
			"schema": format.JSONSchema.Schema,
		}
		if format.JSONSchema.Description != "" {
			out["description"] = format.JSONSchema.Description
		}
		if format.JSONSchema.Strict != nil {
			out["strict"] = *format.JSONSchema.Strict
		}
		return out, nil
	default:
		return nil, fmt.Errorf("response_format.type %q is not supported", format.Type)
	}
}

// responsesObject is the subset of the Responses response we consume.
type responsesObject struct {
	ID     string      `json:"id"`
	Status string      `json:"status"`
	Error  interface{} `json:"error"`
	Output []struct {
		ID      string `json:"id"`
		Type    string `json:"type"`
		Name    string `json:"name"`
		CallID  string `json:"call_id"`
		Args    string `json:"arguments"`
		Summary []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"summary"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
	} `json:"output"`
	Usage struct {
		InputTokens       int `json:"input_tokens"`
		OutputTokens      int `json:"output_tokens"`
		TotalTokens       int `json:"total_tokens"`
		InputTokenDetails struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"input_tokens_details"`
		OutputTokenDetails struct {
			ReasoningTokens int `json:"reasoning_tokens"`
		} `json:"output_tokens_details"`
	} `json:"usage"`
	IncompleteDetails *struct {
		Reason string `json:"reason"`
	} `json:"incomplete_details"`
}

// TranslateResponsesToChat converts a Responses object into an OpenAI chat
// completion.
func TranslateResponsesToChat(raw map[string]interface{}, model string) (map[string]interface{}, int, int, error) {
	b, err := json.Marshal(raw)
	if err != nil {
		return nil, 0, 0, err
	}
	var ro responsesObject
	if err := json.Unmarshal(b, &ro); err != nil {
		return nil, 0, 0, err
	}

	if ro.Error != nil {
		return nil, 0, 0, fmt.Errorf("upstream response failed: %s", responseErrorMessage(ro.Error))
	}
	if ro.Status == "failed" {
		return nil, 0, 0, fmt.Errorf("upstream response failed without error details")
	}

	var text, reasoning strings.Builder
	toolCalls := []map[string]interface{}{}
	finish := "stop"
	for _, item := range ro.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				switch c.Type {
				case "output_text":
					text.WriteString(c.Text)
				case "refusal":
					finish = "content_filter"
					text.WriteString(c.Text)
				}
			}
		case "reasoning":
			for _, summary := range item.Summary {
				if summary.Text != "" {
					reasoning.WriteString(summary.Text)
				}
			}
		case "function_call":
			args := item.Args
			if args == "" {
				args = "{}"
			}
			callID := item.CallID
			if callID == "" {
				callID = item.ID
			}
			toolCalls = append(toolCalls, map[string]interface{}{
				"id": callID, "type": "function",
				"function": map[string]interface{}{
					"name": item.Name, "arguments": args,
				},
			})
			finish = "tool_calls"
		}
	}
	if ro.IncompleteDetails != nil {
		switch ro.IncompleteDetails.Reason {
		case "max_output_tokens":
			finish = "length"
		case "content_filter", "safety_filter":
			finish = "content_filter"
		default:
			return nil, 0, 0, fmt.Errorf("upstream response incomplete: %s", ro.IncompleteDetails.Reason)
		}
	}

	message := map[string]interface{}{"role": "assistant", "content": text.String()}
	if reasoning.Len() > 0 {
		message["reasoning_content"] = reasoning.String()
	}
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
	}
	id := ro.ID
	if id == "" {
		id = "chatcmpl-" + newShortID()
	}
	total := ro.Usage.TotalTokens
	if total == 0 {
		total = ro.Usage.InputTokens + ro.Usage.OutputTokens
	}
	usage := map[string]interface{}{
		"prompt_tokens": ro.Usage.InputTokens, "completion_tokens": ro.Usage.OutputTokens,
		"total_tokens": total,
	}
	if ro.Usage.InputTokenDetails.CachedTokens > 0 {
		usage["prompt_tokens_details"] = map[string]interface{}{
			"cached_tokens": ro.Usage.InputTokenDetails.CachedTokens,
		}
	}
	if ro.Usage.OutputTokenDetails.ReasoningTokens > 0 {
		usage["completion_tokens_details"] = map[string]interface{}{
			"reasoning_tokens": ro.Usage.OutputTokenDetails.ReasoningTokens,
		}
	}
	return map[string]interface{}{
		"id": id, "object": "chat.completion",
		"created": time.Now().Unix(), "model": model,
		"choices": []map[string]interface{}{{
			"index": 0, "message": message, "finish_reason": finish,
		}},
		"usage": usage,
	}, ro.Usage.InputTokens, ro.Usage.OutputTokens, nil
}

func responseErrorMessage(raw interface{}) string {
	switch value := raw.(type) {
	case string:
		return value
	case map[string]interface{}:
		for _, key := range []string{"message", "cause", "code"} {
			if text, ok := value[key].(string); ok && text != "" {
				return text
			}
		}
	}
	encoded, _ := json.Marshal(raw)
	return string(encoded)
}

func (s *Server) handleResponsesNonStream(w http.ResponseWriter, r *http.Request, client *joycode.Client, req *ChatRequest, model joycode.ModelInfo) {
	body, err := TranslateChatToResponses(req, model)
	if err != nil {
		writeOpenAIError(w, 400, err.Error(), "invalid_request_error", "invalid_request", "")
		return
	}
	raw, err := client.PostResponsesContext(r.Context(), body)
	if err != nil {
		if requestCanceled(r, err) {
			slog.Info("responses non-stream request canceled")
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
		writeOpenAIError(w, status, "上游 Responses 接口调用失败: "+err.Error(), errorType, code, "")
		return
	}
	out, inTk, outTk, err := TranslateResponsesToChat(raw, model.DisplayName())
	if err != nil {
		if usage, ok := raw["usage"].(map[string]interface{}); ok {
			store.SetTokenUsage(r, numberToInt(usage["input_tokens"]), numberToInt(usage["output_tokens"]))
		}
		store.SetRequestError(r, http.StatusBadGateway, "upstream Responses status error: "+err.Error())
		writeError(w, 502, "解析上游 Responses 响应失败: "+err.Error())
		return
	}
	store.SetTokenUsage(r, inTk, outTk)
	writeJSON(w, 200, out)
}

// responsesStreamEvent is the subset of Responses SSE payloads we consume.
type responsesStreamEvent struct {
	Type      string      `json:"type"`
	Delta     string      `json:"delta"`
	ItemID    string      `json:"item_id"`
	OutputIdx int         `json:"output_index"`
	Error     interface{} `json:"error"`
	Item      struct {
		ID     string `json:"id"`
		Type   string `json:"type"`
		Name   string `json:"name"`
		CallID string `json:"call_id"`
	} `json:"item"`
	Response struct {
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

func (s *Server) handleResponsesStream(w http.ResponseWriter, r *http.Request, client *joycode.Client, req *ChatRequest, model joycode.ModelInfo) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming unsupported by server")
		return
	}
	body, err := TranslateChatToResponses(req, model)
	if err != nil {
		writeOpenAIError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", "invalid_request", "")
		return
	}
	body["stream"] = true

	resp, committed := connectResponsesStream(w, flusher, r, client, body)
	if resp == nil {
		return
	}
	defer resp.Body.Close()

	if !committed {
		prepareSSE(w)
		w.WriteHeader(http.StatusOK)
	}
	state := newChatResponsesStreamState(w, flusher, r, model.DisplayName(), req.StreamOptions != nil && req.StreamOptions.IncludeUsage)
	state.emitStart()

	reads := readResponsesSSE(r.Context(), resp.Body)
	heartbeat := time.NewTicker(5 * time.Second)
	defer heartbeat.Stop()
	for {
		select {
		case <-r.Context().Done():
			return
		case <-heartbeat.C:
			writeSSEHeartbeat(w, flusher)
		case item, open := <-reads:
			if !open {
				if !state.terminal && !state.failed {
					state.fail("upstream stream ended before a terminal event", "stream_truncated")
				}
				return
			}
			if item.Err != nil {
				state.fail(item.Err.Error(), "stream_read_error")
				return
			}
			event := item.Event
			if event == nil {
				continue
			}
			if err := handleChatResponsesEvent(state, event); err != nil {
				state.fail(err.Error(), "upstream_protocol_error")
				return
			}
			if state.terminal || state.failed {
				return
			}
		}
	}
}

func handleChatResponsesEvent(state *chatResponsesStreamState, event *responsesStreamEvent) error {
	switch event.Type {
	case "response.created", "response.in_progress", "response.output_item.done", "response.output_text.done", "response.function_call_arguments.done":
		return nil
	case "response.output_text.delta":
		if event.Delta != "" {
			state.emit(map[string]interface{}{"content": event.Delta}, nil)
		}
	case "response.reasoning_summary_text.delta", "response.reasoning_summary.delta":
		if event.Delta != "" {
			state.emit(map[string]interface{}{"reasoning_content": event.Delta}, nil)
		}
	case "response.output_item.added":
		if event.Item.Type == "function_call" {
			return state.addTool(event.Item.ID, event.Item.CallID, event.Item.Name)
		}
	case "response.function_call_arguments.delta":
		return state.toolDelta(event.ItemID, event.Item.CallID, event.Delta)
	case "response.completed":
		state.captureUsage(event)
		state.complete("")
	case "response.incomplete":
		state.captureUsage(event)
		reason := ""
		if event.Response.IncompleteDetails != nil {
			reason = event.Response.IncompleteDetails.Reason
		}
		switch reason {
		case "max_output_tokens":
			state.complete("length")
		case "content_filter", "safety_filter":
			state.complete("content_filter")
		default:
			return fmt.Errorf("upstream response incomplete: %s", reason)
		}
	case "response.failed":
		message := responseErrorMessage(event.Response.Error)
		if message == "null" || message == "" {
			message = "upstream response failed"
		}
		return fmt.Errorf("%s", message)
	case "error":
		message := responseErrorMessage(event.Error)
		if message == "null" || message == "" {
			message = "upstream stream returned an error"
		}
		return fmt.Errorf("%s", message)
	default:
		// Unknown events are ignored for forward compatibility. Known event
		// families are handled above and validated where they carry state.
	}
	return nil
}
