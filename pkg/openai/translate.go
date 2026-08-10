package openai

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
)

// ValidateChatRequest validates the container shapes shared by every upstream
// adapter. Adapter-specific semantic validation remains in its translator.
func ValidateChatRequest(req *ChatRequest) error {
	if len(req.Messages) == 0 {
		return fmt.Errorf("messages is required")
	}
	var messages []json.RawMessage
	if err := json.Unmarshal(req.Messages, &messages); err != nil {
		return fmt.Errorf("messages must be an array")
	}
	if len(messages) == 0 {
		return fmt.Errorf("messages must not be empty")
	}
	if len(req.Tools) > 0 {
		var tools []json.RawMessage
		if err := json.Unmarshal(req.Tools, &tools); err != nil {
			return fmt.Errorf("tools must be an array")
		}
	}
	for name, raw := range map[string]json.RawMessage{
		"tool_choice": req.ToolChoice, "response_format": req.ResponseFormat, "stop": req.Stop,
	} {
		if len(raw) > 0 && !json.Valid(raw) {
			return fmt.Errorf("%s must be valid JSON", name)
		}
	}
	return nil
}

// TranslateRequest converts an OpenAI ChatRequest to JoyCode API body.
//
// upstreamModel is the resolved upstream identifier (chatApiModel). It must be
// used rather than req.Model: the client-supplied name may be an alias, and for
// Claude models the display label is accepted upstream but silently served by a
// different model.
func TranslateRequest(req *ChatRequest, upstreamModel string) map[string]interface{} {
	if upstreamModel == "" {
		upstreamModel = req.Model
	}
	body := map[string]interface{}{
		"model":  upstreamModel,
		"stream": req.Stream,
	}
	if len(req.Messages) > 0 {
		var msgs []interface{}
		json.Unmarshal(req.Messages, &msgs)
		body["messages"] = msgs
	}
	if req.MaxTokens > 0 {
		body["max_tokens"] = req.MaxTokens
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	if len(req.Tools) > 0 {
		var tools []interface{}
		json.Unmarshal(req.Tools, &tools)
		body["tools"] = tools
	}
	if len(req.ToolChoice) > 0 {
		body["tool_choice"] = json.RawMessage(req.ToolChoice)
	}
	if len(req.Stop) > 0 {
		body["stop"] = json.RawMessage(req.Stop)
	}
	if len(req.Thinking) > 0 && (ReasoningModels[req.Model] || ReasoningModels[upstreamModel]) {
		body["thinking"] = json.RawMessage(req.Thinking)
	}
	if req.ReasoningEffort != "" {
		body["reasoning_effort"] = req.ReasoningEffort
	}
	if len(req.ResponseFormat) > 0 {
		body["response_format"] = json.RawMessage(req.ResponseFormat)
	}
	if req.ParallelToolCalls != nil {
		body["parallel_tool_calls"] = *req.ParallelToolCalls
	}
	if req.StreamOptions != nil {
		body["stream_options"] = req.StreamOptions
	}
	return body
}

// TranslateResponse converts a JoyCode API response to OpenAI format.
func TranslateResponse(jcResp map[string]interface{}, model string) map[string]interface{} {
	return map[string]interface{}{
		"id":                 fmt.Sprintf("chatcmpl-%s", newShortID()),
		"object":             "chat.completion",
		"created":            time.Now().Unix(),
		"model":              model,
		"choices":            jcResp["choices"],
		"usage":              jcResp["usage"],
		"system_fingerprint": fmt.Sprintf("fp_%s", newShortID()),
	}
}

// TranslateModels converts JoyCode models to OpenAI /v1/models format. The id is
// the display label, which is what clients send back as "model"; hidden internal
// models are omitted. Capabilities come from the upstream feature list so newly
// added models are described correctly without a code change.
func TranslateModels(jcModels []joycode.ModelInfo) map[string]interface{} {
	data := make([]map[string]interface{}, 0, len(jcModels))
	for _, m := range jcModels {
		if m.IsHiddenModel() {
			continue
		}
		mid := m.DisplayName()
		if mid == "" {
			continue
		}
		created := m.CreateTime / 1000
		if created <= 0 {
			created = 1700000000
		}
		entry := map[string]interface{}{
			"id": mid, "object": "model",
			"created": created, "owned_by": "joycode",
			"capabilities": modelCapabilities(m),
		}
		data = append(data, entry)
	}
	return map[string]interface{}{"object": "list", "data": data}
}

// modelCapabilities derives a capability record from live upstream metadata,
// preferring the static table only when upstream reports nothing useful.
func modelCapabilities(m joycode.ModelInfo) ModelCapability {
	caps := ModelCapability{
		MaxTokens: m.MaxOutputTokens(),
		Ctx:       m.ContextWindowTokens(),
	}
	if caps.Ctx <= 0 {
		caps.Ctx = 200000
	}
	for _, f := range m.Features {
		switch f {
		case "vision":
			caps.Vision = true
		case "reasoning", "thinking":
			caps.Reasoning = true
		}
	}
	if static, ok := ModelCapabilities[m.DisplayName()]; ok {
		if !caps.Vision {
			caps.Vision = static.Vision
		}
		if !caps.Reasoning {
			caps.Reasoning = static.Reasoning
		}
		// Upstream omitted the token limits; use the known-good static values
		// rather than the generic default.
		if m.RespMaxTokens <= 0 && static.MaxTokens > 0 {
			caps.MaxTokens = static.MaxTokens
		}
		if m.MaxTotalTokens <= 0 && static.Ctx > 0 {
			caps.Ctx = static.Ctx
		}
	}
	return caps
}

// TranslateStreamChunk converts a JoyCode SSE data line to OpenAI format.
func TranslateStreamChunk(data string, model string) string {
	if data == "[DONE]" {
		return "data: [DONE]\n\n"
	}
	var chunk map[string]interface{}
	if err := json.Unmarshal([]byte(data), &chunk); err != nil {
		return fmt.Sprintf("data: %s\n\n", data)
	}
	if _, ok := chunk["id"]; !ok {
		chunk["id"] = fmt.Sprintf("chatcmpl-%s", newShortID())
	}
	chunk["model"] = model
	chunk["object"] = "chat.completion.chunk"
	b, _ := json.Marshal(chunk)
	return fmt.Sprintf("data: %s\n\n", b)
}

func newShortID() string {
	return fmt.Sprintf("%d", time.Now().UnixNano()%1e12)
}

// ResolveModelInfo resolves a requested model against the live model registry,
// falling back to the account default, then the system default. Live resolution
// means models upstream adds or removes work without a code change.
func ResolveModelInfo(model string, accountDefault string, systemDefault string) joycode.ModelInfo {
	return joycode.DefaultRegistry.ResolveRequest(model, accountDefault, systemDefault)
}

// ResolveModel returns the upstream identifier for the request's model.
func ResolveModel(model string, accountDefault string, systemDefault string) string {
	return ResolveModelInfo(model, accountDefault, systemDefault).APIName()
}
