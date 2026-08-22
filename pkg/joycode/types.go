package joycode

// Adapter types reported by upstream in ModelInfo.ExtJSON["adapterType"].
// The adapter decides which upstream endpoint can serve a model: sending a model
// to the wrong endpoint fails with the opaque upstream error "HTTP调用异常" (1032).
const (
	// AdapterChatCompletions is the implicit adapter for models with no
	// adapterType set: /api/saas/openai/v1/chat/completions.
	AdapterChatCompletions = ""
	// AdapterAnthropic models are served only by /api/saas/anthropic/v1/messages.
	AdapterAnthropic = "anthropic"
	// AdapterOpenAIResponse models are served only by the OpenAI Responses API
	// at /api/saas/openai/v1/responses.
	AdapterOpenAIResponse = "openai-response"
)

// ModelInfo describes a JoyCode AI model.
type ModelInfo struct {
	Label              string                 `json:"label"`
	ChatAPIModel       string                 `json:"chatApiModel"`
	MaxTotalTokens     int                    `json:"maxTotalTokens"`
	RespMaxTokens      int                    `json:"respMaxTokens"`
	Temperature        float64                `json:"temperature"`
	Features           []string               `json:"features"`
	SupportStream      bool                   `json:"supportStream"`
	VerificationStatus string                 `json:"verificationStatus"`
	ModelID            string                 `json:"modelId"`
	CreateTime         int64                  `json:"createTime"`
	Description        string                 `json:"description"`
	ExtJSON            map[string]interface{} `json:"extJson"`
	Hidden             bool                   `json:"hidden"`
	IsHidden           bool                   `json:"isHidden"`
	SortOrder          int                    `json:"sortOrder"`
}

// Adapter returns the upstream adapter type for this model, which determines
// which endpoint can serve it.
func (m ModelInfo) Adapter() string {
	if m.ExtJSON == nil {
		return AdapterChatCompletions
	}
	if v, ok := m.ExtJSON["adapterType"].(string); ok {
		return v
	}
	return AdapterChatCompletions
}

// IsHiddenModel reports whether upstream marks this model as not user-selectable.
func (m ModelInfo) IsHiddenModel() bool {
	return m.Hidden || m.IsHidden
}

// APIName returns the identifier to send upstream. This is always chatApiModel,
// never the display label: for Claude models the label ("Claude-Opus-4.7") is
// accepted by the upstream gateway but silently served by a different, weaker
// model, while chatApiModel ("Claude-Opus-4.7-hq") routes correctly.
func (m ModelInfo) APIName() string {
	if m.ChatAPIModel != "" {
		return m.ChatAPIModel
	}
	return m.Label
}

// DisplayName returns the client-facing identifier for this model.
func (m ModelInfo) DisplayName() string {
	if m.Label != "" {
		return m.Label
	}
	return m.ChatAPIModel
}

// ContextWindowTokens returns the client-facing context window.
//
// JoyCode currently reports 200K for GPT-5.6 Sol, but live requests and
// needle-in-a-haystack aggregation tests verify that the backend accepts and
// uses roughly one million input tokens. Keep this override model-specific:
// the same 200K value is reported for many unrelated models whose real limits
// have not been measured.
//
// Kimi-K3 (upstream kimi-k3-large2): needle-in-a-haystack probes confirm the
// model itself attends to 510K+ prompt tokens, but the Color gateway rejects
// any request body over 5242880 bytes with
// {"code":"1","echo":"content length exceeded 5242880 bytes"}. That 5MB cap —
// not the model — is the effective ceiling (~770K tokens of English filler at
// the measured 6.8 bytes/token; ~450K CJK characters). 600K leaves headroom
// for tool schemas, system prompts, and output, and stays clear of the
// gateway's flaky HTML error page seen near the cap.
func (m ModelInfo) ContextWindowTokens() int {
	if normalizeModelName(m.DisplayName()) == normalizeModelName("GPT-5.6 Sol") ||
		normalizeModelName(m.APIName()) == normalizeModelName("GPT-5.6 Sol") {
		return 1_000_000
	}
	if normalizeModelName(m.DisplayName()) == normalizeModelName("Kimi-K3") ||
		normalizeModelName(m.APIName()) == normalizeModelName("Kimi-K3") {
		return 600_000
	}
	if m.MaxTotalTokens > 0 {
		return m.MaxTotalTokens
	}
	return 200_000
}

// MaxOutputTokens returns the model's output cap, falling back to a safe default.
func (m ModelInfo) MaxOutputTokens() int {
	if m.RespMaxTokens > 0 {
		return m.RespMaxTokens
	}
	return 8192
}

// EffectiveMaxTokens decides the max_tokens actually sent upstream.
//
// requested is the client's value (0 when absent) and floor is the configured
// default_max_tokens (0 when unset, meaning "use the model's cap"). The result
// is never above the model's own limit, because upstream silently truncates
// oversized values, and never below floor.
//
// A requested value below floor is raised rather than honoured. max_tokens is a
// ceiling, not a target: raising it costs nothing on its own, while a low value
// truncates reasoning models mid-thought before they emit any visible text.
// Callers that need short replies should ask for brevity in the prompt.
func (m ModelInfo) EffectiveMaxTokens(requested, floor int) int {
	limit := m.MaxOutputTokens()
	if floor <= 0 || floor > limit {
		floor = limit
	}
	if requested < floor {
		requested = floor
	}
	if requested > limit {
		requested = limit
	}
	return requested
}
