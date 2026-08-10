package openai

import (
	"bufio"
	"encoding/json"
	"fmt"
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
func TranslateChatToResponses(req *ChatRequest, model joycode.ModelInfo) map[string]interface{} {
	body := map[string]interface{}{
		"model": model.APIName(),
		"input": chatMessagesToResponsesInput(req.Messages),
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
	if tools := chatToolsToResponses(req.Tools); len(tools) > 0 {
		body["tools"] = tools
		if len(req.ToolChoice) > 0 {
			body["tool_choice"] = json.RawMessage(req.ToolChoice)
		}
	}
	return body
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
func chatMessagesToResponsesInput(raw json.RawMessage) []map[string]interface{} {
	var msgs []chatMessage
	if err := json.Unmarshal(raw, &msgs); err != nil {
		return nil
	}
	items := make([]map[string]interface{}, 0, len(msgs))
	for _, m := range msgs {
		if m.Role == "tool" {
			items = append(items, map[string]interface{}{
				"type":    "function_call_output",
				"call_id": m.ToolCallID,
				"output":  contentToText(m.Content),
			})
			continue
		}
		if text := contentToText(m.Content); text != "" {
			role := m.Role
			if role == "system" {
				role = "developer"
			}
			items = append(items, map[string]interface{}{
				"role": role, "content": text,
			})
		}
		for _, tc := range m.ToolCalls {
			args := tc.Function.Args
			if args == "" {
				args = "{}"
			}
			items = append(items, map[string]interface{}{
				"type":      "function_call",
				"call_id":   tc.ID,
				"name":      tc.Function.Name,
				"arguments": args,
			})
		}
	}
	return items
}

// contentToText flattens a string-or-parts content field into plain text.
func contentToText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err == nil {
		var sb strings.Builder
		for _, p := range parts {
			if p.Text == "" {
				continue
			}
			if sb.Len() > 0 {
				sb.WriteString("\n")
			}
			sb.WriteString(p.Text)
		}
		return sb.String()
	}
	return ""
}

// chatToolsToResponses maps OpenAI chat tools to Responses function tools, which
// are flat rather than nested under "function".
func chatToolsToResponses(raw json.RawMessage) []map[string]interface{} {
	if len(raw) == 0 {
		return nil
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
		return nil
	}
	out := make([]map[string]interface{}, 0, len(tools))
	for _, t := range tools {
		if t.Function.Name == "" {
			continue
		}
		params := json.RawMessage(`{"type":"object","properties":{}}`)
		if len(t.Function.Parameters) > 0 {
			params = t.Function.Parameters
		}
		out = append(out, map[string]interface{}{
			"type":        "function",
			"name":        t.Function.Name,
			"description": t.Function.Description,
			"parameters":  params,
		})
	}
	return out
}

// responsesObject is the subset of the Responses response we consume.
type responsesObject struct {
	ID     string `json:"id"`
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
		TotalTokens  int `json:"total_tokens"`
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

	var text strings.Builder
	toolCalls := []map[string]interface{}{}
	finish := "stop"
	for _, item := range ro.Output {
		switch item.Type {
		case "message":
			for _, c := range item.Content {
				if c.Type == "output_text" {
					text.WriteString(c.Text)
				}
			}
		case "function_call":
			args := item.Args
			if args == "" {
				args = "{}"
			}
			toolCalls = append(toolCalls, map[string]interface{}{
				"id": item.CallID, "type": "function",
				"function": map[string]interface{}{
					"name": item.Name, "arguments": args,
				},
			})
			finish = "tool_calls"
		}
	}
	if ro.IncompleteDetails != nil && ro.IncompleteDetails.Reason == "max_output_tokens" {
		finish = "length"
	}

	message := map[string]interface{}{"role": "assistant", "content": text.String()}
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
	return map[string]interface{}{
		"id": id, "object": "chat.completion",
		"created": time.Now().Unix(), "model": model,
		"choices": []map[string]interface{}{{
			"index": 0, "message": message, "finish_reason": finish,
		}},
		"usage": map[string]interface{}{
			"prompt_tokens":     ro.Usage.InputTokens,
			"completion_tokens": ro.Usage.OutputTokens,
			"total_tokens":      total,
		},
	}, ro.Usage.InputTokens, ro.Usage.OutputTokens, nil
}

func (s *Server) handleResponsesNonStream(w http.ResponseWriter, r *http.Request, client *joycode.Client, req *ChatRequest, model joycode.ModelInfo) {
	body := TranslateChatToResponses(req, model)
	raw, err := client.PostResponses(body)
	if err != nil {
		writeError(w, 500, "上游 Responses 接口调用失败: "+err.Error())
		return
	}
	out, inTk, outTk, err := TranslateResponsesToChat(raw, model.DisplayName())
	if err != nil {
		writeError(w, 502, "解析上游 Responses 响应失败: "+err.Error())
		return
	}
	store.SetTokenUsage(r, inTk, outTk)
	writeJSON(w, 200, out)
}

// responsesStreamEvent is the subset of Responses SSE payloads we consume.
type responsesStreamEvent struct {
	Type  string `json:"type"`
	Delta string `json:"delta"`
	Item  struct {
		Type   string `json:"type"`
		Name   string `json:"name"`
		CallID string `json:"call_id"`
	} `json:"item"`
	Response struct {
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
		writeError(w, 500, "streaming unsupported by server")
		return
	}
	body := TranslateChatToResponses(req, model)
	body["stream"] = true

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.WriteHeader(200)

	resp, err := client.PostResponsesStream(body)
	if err != nil {
		fmt.Fprintf(w, "data: {\"error\":{\"message\":%q}}\n\n", err.Error())
		flusher.Flush()
		return
	}
	defer resp.Body.Close()

	id := "chatcmpl-" + newShortID()
	created := time.Now().Unix()
	displayModel := model.DisplayName()
	emit := func(delta map[string]interface{}, finish interface{}) {
		chunk := map[string]interface{}{
			"id": id, "object": "chat.completion.chunk",
			"created": created, "model": displayModel,
			"choices": []map[string]interface{}{{
				"index": 0, "delta": delta, "finish_reason": finish,
			}},
		}
		b, _ := json.Marshal(chunk)
		fmt.Fprintf(w, "data: %s\n\n", b)
		flusher.Flush()
	}

	emit(map[string]interface{}{"role": "assistant", "content": ""}, nil)

	finish := "stop"
	toolIdx := -1
	var inTk, outTk int
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		for strings.HasPrefix(line, "data: ") {
			line = strings.TrimPrefix(line, "data: ")
		}
		line = strings.TrimSpace(line)
		if line == "" || !strings.HasPrefix(line, "{") {
			continue
		}
		var ev responsesStreamEvent
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			continue
		}
		switch ev.Type {
		case "response.output_text.delta":
			if ev.Delta != "" {
				emit(map[string]interface{}{"content": ev.Delta}, nil)
			}
		case "response.output_item.added":
			if ev.Item.Type == "function_call" {
				toolIdx++
				finish = "tool_calls"
				emit(map[string]interface{}{"tool_calls": []map[string]interface{}{{
					"index": toolIdx, "id": ev.Item.CallID, "type": "function",
					"function": map[string]interface{}{"name": ev.Item.Name, "arguments": ""},
				}}}, nil)
			}
		case "response.function_call_arguments.delta":
			if toolIdx >= 0 && ev.Delta != "" {
				emit(map[string]interface{}{"tool_calls": []map[string]interface{}{{
					"index":    toolIdx,
					"function": map[string]interface{}{"arguments": ev.Delta},
				}}}, nil)
			}
		case "response.completed", "response.incomplete", "response.failed":
			if u := ev.Response.Usage; u != nil {
				inTk, outTk = u.InputTokens, u.OutputTokens
			}
			if d := ev.Response.IncompleteDetails; d != nil && d.Reason == "max_output_tokens" {
				finish = "length"
			}
		}
	}

	emit(map[string]interface{}{}, finish)
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
	if inTk > 0 || outTk > 0 {
		store.SetTokenUsage(r, inTk, outTk)
	}
}
