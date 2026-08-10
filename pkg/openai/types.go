package openai

import "encoding/json"

// ChatRequest is the OpenAI-compatible chat completion request.
type ChatRequest struct {
	Model               string          `json:"model"`
	Messages            json.RawMessage `json:"messages"`
	Stream              bool            `json:"stream,omitempty"`
	MaxTokens           int             `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	Tools               json.RawMessage `json:"tools,omitempty"`
	ToolChoice          json.RawMessage `json:"tool_choice,omitempty"`
	Stop                json.RawMessage `json:"stop,omitempty"`
	Thinking            json.RawMessage `json:"thinking,omitempty"`
	ReasoningEffort     string          `json:"reasoning_effort,omitempty"`
	ResponseFormat      json.RawMessage `json:"response_format,omitempty"`
	ParallelToolCalls   *bool           `json:"parallel_tool_calls,omitempty"`
	StreamOptions       *StreamOptions  `json:"stream_options,omitempty"`
}

// RequestedMaxTokens returns the modern output limit when present, otherwise
// the legacy max_tokens value. OpenAI recommends max_completion_tokens for
// reasoning models, but older clients still send max_tokens.
func (r *ChatRequest) RequestedMaxTokens() int {
	if r.MaxCompletionTokens != nil {
		return *r.MaxCompletionTokens
	}
	return r.MaxTokens
}

// StreamOptions controls Chat Completions stream metadata.
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage,omitempty"`
}

// ModelCapability describes a model's feature set.
type ModelCapability struct {
	Vision    bool `json:"vision"`
	Reasoning bool `json:"reasoning"`
	MaxTokens int  `json:"max_tokens"`
	Ctx       int  `json:"ctx"`
}

// ModelCapabilities maps model IDs to their capabilities.
var ModelCapabilities = map[string]ModelCapability{
	"JoyAI-Code":          {MaxTokens: 64000, Ctx: 200000},
	"Claude-Opus-4.7":     {MaxTokens: 32000, Ctx: 200000},
	"MiniMax-M2.7":        {Reasoning: true, MaxTokens: 16384, Ctx: 200000},
	"Kimi-K2.5":           {Vision: true, MaxTokens: 16384, Ctx: 200000},
	"Kimi-K2.6":           {Vision: true, Reasoning: true, MaxTokens: 16384, Ctx: 200000},
	"GLM-5.1":             {Reasoning: true, MaxTokens: 16384, Ctx: 200000},
	"GLM-5":               {MaxTokens: 8192, Ctx: 200000},
	"GLM-4.7":             {MaxTokens: 8192, Ctx: 200000},
	"Doubao-Seed-2.0-pro": {MaxTokens: 16384, Ctx: 200000},
}

// ReasoningModels supports thinking/reasoning control parameters.
var ReasoningModels = map[string]bool{
	"GLM-5.1": true, "Kimi-K2.6": true, "MiniMax-M2.7": true,
}
