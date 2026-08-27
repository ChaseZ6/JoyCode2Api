package anthropic

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

// TranslateRequest converts an Anthropic MessageRequest to a JoyCode API body.
func TranslateRequest(req *MessageRequest, accountDefault string, systemDefault string) map[string]interface{} {
	model := resolveModel(req.Model, accountDefault, systemDefault)
	messages := buildMessages(req)

	body := map[string]interface{}{
		"model":      model,
		"messages":   messages,
		"stream":     req.Stream,
		"max_tokens": req.MaxTokens,
	}
	if req.Temperature != nil {
		body["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		body["top_p"] = *req.TopP
	}
	if len(req.StopSequences) > 0 {
		body["stop"] = req.StopSequences
	}
	if len(req.Tools) > 0 {
		if hasBuiltinTools(req.Tools) {
			// When the request includes Anthropic built-in tools (e.g. web_search_20250305)
			// we cannot convert them to OpenAI function format.  Pass the tools as-is in
			// Anthropic format; the upstream must support this layout.
			body["tools"] = req.Tools
		} else {
			body["tools"] = convertToolsToOpenAI(req.Tools)
		}
	}
	if len(req.ToolChoice) > 0 {
		if tc := convertToolChoice(req.ToolChoice); tc != nil {
			body["tool_choice"] = tc
		}
	}
	return body
}

// TranslateAnthropicRequest converts an Anthropic request to JoyCode's native
// Anthropic endpoint body. Claude-family models reject the legacy OpenAI path.
func TranslateAnthropicRequest(req *MessageRequest, accountDefault string, systemDefault string) map[string]interface{} {
	model := resolveNativeAnthropicModel(req.Model, accountDefault, systemDefault)
	body := map[string]interface{}{
		"model":      model,
		"messages":   req.Messages,
		"stream":     true,
		"max_tokens": req.MaxTokens,
		"thinking":   map[string]string{"type": "disabled"},
	}
	if req.System != nil {
		body["system"] = normalizeAnthropicSystem(req.System)
	}
	if len(req.StopSequences) > 0 {
		body["stop_sequences"] = req.StopSequences
	}
	if len(req.Tools) > 0 {
		body["tools"] = normalizeToolsForNativeAnthropic(req.Tools)
	}
	if len(req.ToolChoice) > 0 {
		body["tool_choice"] = json.RawMessage(req.ToolChoice)
	}
	return body
}

func normalizeAnthropicSystem(raw json.RawMessage) interface{} {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []map[string]interface{}{{"type": "text", "text": s}}
	}
	var blocks []map[string]interface{}
	if err := json.Unmarshal(raw, &blocks); err == nil {
		return blocks
	}
	return raw
}

// ClaudeNativeEnabled reports whether the native Anthropic code path is active.
// Guarded by the "enable_claude" store setting — defaults to false until the
// upstream JoyCode platform makes Claude models generally available.
func ClaudeNativeEnabled(s *store.Store) bool {
	if s == nil {
		return false
	}
	return s.GetSetting("enable_claude") == "true"
}

// maxRetriesOrDefault clamps invalid max_retries settings (0, negative, or
// unparsable values) to at least one upstream attempt. A setting of 0 used to
// skip the retry loop entirely, so requests failed before ever contacting the
// gateway.
func maxRetriesOrDefault(s *store.Store) int {
	maxRetries := 3
	if s != nil {
		maxRetries = s.GetIntSetting("max_retries", 3)
	}
	if maxRetries < 1 {
		maxRetries = 1
	}
	return maxRetries
}

func IsNativeAnthropicModel(model string) bool {
	m := strings.ToLower(model)
	return strings.HasPrefix(m, "claude") || strings.Contains(m, "claude-")
}

// resolveNativeAnthropicModel returns the upstream identifier to send to the
// native Anthropic endpoint. It resolves against the live model registry and
// returns chatApiModel (e.g. "Claude-Opus-4.7-hq"), never the display label:
// upstream accepts the label but silently serves it with a different model
// (label "Claude-Opus-4.7" is answered by claude-sonnet-4-6).
func resolveNativeAnthropicModel(model string, accountDefault string, systemDefault string) string {
	info := ResolveModelInfo(model, accountDefault, systemDefault)
	if info.Adapter() == joycode.AdapterAnthropic {
		return info.APIName()
	}
	// The caller decided this request belongs on the native path but the
	// resolved model is not an Anthropic-adapter model. Fall back to the newest
	// Claude the registry knows about.
	if best, ok := newestAnthropicModel(); ok {
		return best.APIName()
	}
	return info.APIName()
}

// newestAnthropicModel returns the highest-sorted Anthropic-adapter model from
// the registry, so the native path does not depend on a hardcoded version.
func newestAnthropicModel() (joycode.ModelInfo, bool) {
	var best joycode.ModelInfo
	found := false
	for _, m := range joycode.DefaultRegistry.Visible() {
		if m.Adapter() != joycode.AdapterAnthropic {
			continue
		}
		if !found || m.SortOrder > best.SortOrder {
			best = m
			found = true
		}
	}
	return best, found
}

// isBuiltinTool reports whether t is an Anthropic built-in tool (e.g.
// web_search_20250305, computer_20241022).  Built-in tools carry a non-empty
// Type field and have no InputSchema; they cannot be represented in the OpenAI
// function-calling format and must be handled separately.
func isBuiltinTool(t Tool) bool {
	return t.Type != "" && len(t.InputSchema) == 0
}

// hasBuiltinTools reports whether any tool in the slice is a built-in tool.
func hasBuiltinTools(tools []Tool) bool {
	for _, t := range tools {
		if isBuiltinTool(t) {
			return true
		}
	}
	return false
}

// normalizeToolsForNativeAnthropic prepares tools for JoyCode's native Anthropic
// endpoint. JoyCode does not support Anthropic built-in tool types such as
// "web_search_20250305"; it only accepts "custom" and a few internal types.
// The upstream validator is strict (Pydantic discriminated union) and rejects
// any extra fields, so we hand-build each tool map with only the known-safe
// fields instead of serialising the Tool struct directly.
func normalizeToolsForNativeAnthropic(tools []Tool) []map[string]interface{} {
	// Types natively supported by the JoyCode Anthropic endpoint.
	supportedTypes := map[string]bool{
		"bash_20250124":                   true,
		"custom":                          true,
		"memory_20250818":                 true,
		"text_editor_20250124":            true,
		"text_editor_20250429":            true,
		"text_editor_20250728":            true,
		"tool_search_tool_bm25":           true,
		"tool_search_tool_bm25_20251119":  true,
		"tool_search_tool_regex":          true,
		"tool_search_tool_regex_20251119": true,
	}
	emptySchema := json.RawMessage(`{"type":"object","properties":{}}`)
	webSearchSchema := json.RawMessage(`{"type":"object","properties":{"query":{"type":"string","description":"The search query"}},"required":["query"]}`)
	out := make([]map[string]interface{}, 0, len(tools))
	for _, t := range tools {
		toolType := t.Type
		if toolType == "" {
			toolType = "custom"
		} else if !supportedTypes[toolType] {
			// Unsupported built-in type (e.g. web_search_20250305): downgrade to
			// "custom" so the upstream validator accepts it.
			toolType = "custom"
		}

		schema := t.InputSchema
		if len(schema) == 0 {
			// For web_search, provide the full schema so the model knows to pass a query.
			if t.Name == "web_search" {
				schema = webSearchSchema
			} else {
				schema = emptySchema
			}
		}

		// Build the minimal field set accepted by JoyCode's strict validator.
		// Extra fields from the original tool (max_uses, allowed_domains,
		// user_location, cache_control, etc.) are intentionally omitted.
		m := map[string]interface{}{
			"type":         toolType,
			"name":         t.Name,
			"input_schema": schema,
		}
		if t.Description != "" {
			m["description"] = t.Description
		}
		out = append(out, m)
	}
	return out
}

// convertToolsToOpenAI converts Anthropic-format tools to OpenAI function-calling format.
// Built-in tools (type != "" && no input_schema) are skipped because they have no
// OpenAI equivalent — callers should check hasBuiltinTools first and avoid this path.
func convertToolsToOpenAI(tools []Tool) []interface{} {
	result := make([]interface{}, 0, len(tools))
	for _, t := range tools {
		if isBuiltinTool(t) {
			// Built-in tools (e.g. web_search_20250305) cannot be converted to
			// OpenAI function format; skip them silently.
			continue
		}
		tool := map[string]interface{}{
			"type": "function",
			"function": map[string]interface{}{
				"name":        t.Name,
				"description": t.Description,
				"parameters":  t.InputSchema,
			},
		}
		result = append(result, tool)
	}
	return result
}

// TranslateResponse converts a JoyCode API response to Anthropic Message format.
func TranslateResponse(jcResp map[string]interface{}, reqModel string) *MessageResponse {
	msgID := "msg_" + newID()
	usage := extractUsage(jcResp)

	choices, _ := jcResp["choices"].([]interface{})
	if len(choices) == 0 {
		return &MessageResponse{
			ID: msgID, Type: "message", Role: "assistant",
			Model: reqModel, Content: []ContentBlock{{Type: "text", Text: ""}},
			StopReason: strPtr("end_turn"), Usage: usage,
		}
	}
	choice, _ := choices[0].(map[string]interface{})
	msg, _ := choice["message"].(map[string]interface{})

	content := []ContentBlock{}
	stopReason := "end_turn"

	// Handle tool_calls from JoyCode response
	toolCalls, _ := msg["tool_calls"].([]interface{})
	if len(toolCalls) > 0 {
		stopReason = "tool_use"
		for _, tc := range toolCalls {
			tcMap, _ := tc.(map[string]interface{})
			fn, _ := tcMap["function"].(map[string]interface{})
			name, _ := fn["name"].(string)
			name = toolNameOrFallback(name, nil)
			argsStr := repairJsonBoundaries(normalizeToolArgsFromValue(fn["arguments"]))
			id, _ := tcMap["id"].(string)
			if id == "" {
				id = "toolu_" + newID()
			}
			if argsStr == "" || !json.Valid([]byte(argsStr)) {
				argsStr = "{}"
			}

			var input json.RawMessage = json.RawMessage(argsStr)
			content = append(content, ContentBlock{
				Type:  "tool_use",
				ID:    id,
				Name:  name,
				Input: input,
			})
		}
	} else {
		content = append(content, ContentBlock{Type: "text", Text: stripThinkTags(extractMessageText(msg["content"]))})
	}

	return &MessageResponse{
		ID:         msgID,
		Type:       "message",
		Role:       "assistant",
		Model:      reqModel,
		Content:    content,
		StopReason: &stopReason,
		Usage:      usage,
	}
}

// ResolveModelInfo resolves a requested model against the live registry,
// falling back to the account default, then the system default. Resolving
// against live data means models upstream adds (or removes) work without a code
// change; the previous hardcoded whitelist silently rewrote unknown models to
// the default, which looked to the caller like the request was ignored.
func ResolveModelInfo(model string, accountDefault string, systemDefault string) joycode.ModelInfo {
	return joycode.DefaultRegistry.ResolveRequest(model, accountDefault, systemDefault)
}

// resolveModel returns the upstream identifier for a chat/completions request.
func resolveModel(model string, accountDefault string, systemDefault string) string {
	return ResolveModelInfo(model, accountDefault, systemDefault).APIName()
}

// contentBlock represents a single content block in Anthropic format.
type contentBlock struct {
	Type  string          `json:"type"`
	Text  string          `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// tool_result fields
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`

	// image block source (type=image): {type:base64,media_type,data} or {type:url,url}
	Source json.RawMessage `json:"source,omitempty"`
}

func buildMessages(req *MessageRequest) []map[string]interface{} {
	msgs := make([]map[string]interface{}, 0, len(req.Messages)+1)

	if req.System != nil {
		if sys := parseContent(req.System); sys != "" {
			msgs = append(msgs, map[string]interface{}{
				"role": "system", "content": sys,
			})
		}
	}

	// Collect all tool_use IDs from assistant messages so we can strip orphaned tool_results
	toolUseIDs := map[string]bool{}
	for _, m := range req.Messages {
		if m.Role == "assistant" {
			var blocks []contentBlock
			if json.Unmarshal(m.Content, &blocks) == nil {
				for _, b := range blocks {
					if b.Type == "tool_use" && b.ID != "" {
						toolUseIDs[b.ID] = true
					}
				}
			}
		}
	}

	// Collect all tool_result IDs from user messages so we can detect tool_use
	// blocks that have no result — "dangling" calls left when a truncation
	// boundary removed the tool_result, or when the user interrupted between
	// the tool call and its result. OpenAI requires every tool_call to have a
	// matching tool message, so we downgrade those to text rather than emit a
	// protocol-incomplete sequence.
	toolResultIDs := map[string]bool{}
	for _, m := range req.Messages {
		if m.Role == "user" {
			var blocks []contentBlock
			if json.Unmarshal(m.Content, &blocks) == nil {
				for _, b := range blocks {
					if b.Type == "tool_result" && b.ToolUseID != "" {
						toolResultIDs[b.ToolUseID] = true
					}
				}
			}
		}
	}

	for _, m := range req.Messages {
		for _, converted := range convertMessage(m.Role, m.Content, toolUseIDs, toolResultIDs) {
			msgs = append(msgs, converted)
		}
	}
	return msgs
}

// convertMessage converts a single Anthropic message to one or more OpenAI format messages.
func convertMessage(role string, raw json.RawMessage, toolUseIDs map[string]bool, toolResultIDs map[string]bool) []map[string]interface{} {
	// Try simple string content first
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []map[string]interface{}{{"role": role, "content": s}}
	}

	// Try as content blocks
	var blocks []contentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return []map[string]interface{}{{"role": role, "content": string(raw)}}
	}

	switch role {
	case "assistant":
		return []map[string]interface{}{convertAssistantBlocks(blocks, toolResultIDs)}
	case "user":
		return convertUserBlocks(blocks, toolUseIDs)
	default:
		return []map[string]interface{}{{"role": role, "content": extractText(blocks)}}
	}
}

// convertAssistantBlocks handles assistant messages with tool_use blocks.
func convertAssistantBlocks(blocks []contentBlock, toolResultIDs map[string]bool) map[string]interface{} {
	textParts := []string{}
	toolCalls := []interface{}{}

	for _, b := range blocks {
		switch b.Type {
		case "text":
			textParts = append(textParts, b.Text)
		case "tool_use":
			args := "{}"
			if len(b.Input) > 0 {
				args = string(b.Input)
			}
			id := b.ID
			if id == "" {
				id = "call_" + newID()
			}
			// A tool_use with no matching tool_result anywhere in the retained
			// conversation would produce a tool_call with no tool response.
			// Downgrade it to a text note so the upstream sees a valid sequence.
			if !toolResultIDs[id] {
				textParts = append(textParts, "[previous tool call: "+b.Name+"("+args+") — result omitted]")
				continue
			}
			toolCalls = append(toolCalls, map[string]interface{}{
				"id":   id,
				"type": "function",
				"function": map[string]interface{}{
					"name":      b.Name,
					"arguments": args,
				},
			})
		}
	}

	msg := map[string]interface{}{
		"role":    "assistant",
		"content": strings.Join(textParts, "\n"),
	}
	if len(toolCalls) > 0 {
		msg["tool_calls"] = toolCalls
		if msg["content"] == "" {
			msg["content"] = nil
		}
	}
	return msg
}

// convertUserBlocks handles user messages that may contain tool_result blocks.
// tool_result blocks are converted to separate "tool" role messages in OpenAI format.
// Orphaned tool_results (whose tool_use was removed by truncation) are stripped.
func convertUserBlocks(blocks []contentBlock, toolUseIDs map[string]bool) []map[string]interface{} {
	var result []map[string]interface{}
	var textParts []string
	// orderedParts preserves text/image interleaving for multimodal content.
	var orderedParts []map[string]interface{}
	hasImage := false

	for _, b := range blocks {
		switch b.Type {
		case "text":
			textParts = append(textParts, b.Text)
			orderedParts = append(orderedParts, map[string]interface{}{"type": "text", "text": b.Text})
		case "image":
			// Translate Anthropic image blocks to OpenAI image_url parts instead
			// of dropping them (issue #4); mirrors what the OpenAI path forwards.
			if url := imageBlockToDataURL(b.Source); url != "" {
				hasImage = true
				orderedParts = append(orderedParts, map[string]interface{}{
					"type":      "image_url",
					"image_url": map[string]interface{}{"url": url},
				})
			}
		case "tool_result":
			// Skip orphaned tool_results whose tool_use was removed by truncation
			if b.ToolUseID != "" && !toolUseIDs[b.ToolUseID] {
				continue
			}
			// Convert each tool_result to an OpenAI "tool" role message
			resultText := extractToolResultContent(b.Content)
			result = append(result, map[string]interface{}{
				"role":         "tool",
				"tool_call_id": b.ToolUseID,
				"content":      resultText,
			})
		}
	}

	// User content is a multimodal parts array when images are present, otherwise
	// a plain joined string (unchanged text-only behavior).
	var userContent interface{}
	if hasImage {
		userContent = orderedParts
	} else {
		userContent = strings.Join(textParts, "\n")
	}
	hasUserContent := hasImage || len(textParts) > 0

	// If there's user content alongside tool results, add it after them.
	if hasUserContent && len(result) > 0 {
		result = append(result, map[string]interface{}{
			"role": "user", "content": userContent,
		})
	}

	// If no tool_result blocks, return as single user message
	if len(result) == 0 {
		return []map[string]interface{}{{"role": "user", "content": userContent}}
	}

	return result
}

// imageBlockToDataURL converts an Anthropic image block source to an OpenAI
// image_url value: base64 sources become a data: URL, url sources pass through.
// Returns "" if the source is missing or unrecognized.
func imageBlockToDataURL(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var src struct {
		Type      string `json:"type"`
		MediaType string `json:"media_type"`
		Data      string `json:"data"`
		URL       string `json:"url"`
	}
	if err := json.Unmarshal(raw, &src); err != nil {
		return ""
	}
	if src.Data != "" {
		mt := src.MediaType
		if mt == "" {
			mt = "image/png"
		}
		return "data:" + mt + ";base64," + src.Data
	}
	if src.URL != "" {
		return src.URL
	}
	return ""
}

func extractToolResultContent(raw json.RawMessage) string {
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
		parts := make([]string, 0, len(blocks))
		for _, b := range blocks {
			if b.Type == "text" {
				parts = append(parts, b.Text)
			}
		}
		return strings.Join(parts, "\n")
	}
	return string(raw)
}

func extractText(blocks []contentBlock) string {
	parts := make([]string, 0, len(blocks))
	for _, b := range blocks {
		if b.Type == "text" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n")
}

func parseContent(raw json.RawMessage) string {
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
		parts := make([]string, 0, len(blocks))
		for _, b := range blocks {
			if b.Type == "text" {
				parts = append(parts, b.Text)
			}
		}
		return strings.Join(parts, "\n")
	}
	return string(raw)
}

func extractUsage(jcResp map[string]interface{}) Usage {
	u := Usage{}
	usage, _ := jcResp["usage"].(map[string]interface{})
	if usage == nil {
		return u
	}
	if v, ok := usage["prompt_tokens"].(float64); ok {
		u.InputTokens = int(v)
	}
	if v, ok := usage["completion_tokens"].(float64); ok {
		u.OutputTokens = int(v)
	}
	return u
}

func newID() string {
	b := make([]byte, 12)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func strPtr(s string) *string { return &s }

// convertToolChoice converts Anthropic tool_choice to OpenAI format.
func convertToolChoice(raw json.RawMessage) interface{} {
	var tc struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if err := json.Unmarshal(raw, &tc); err != nil {
		return nil
	}
	switch tc.Type {
	case "auto":
		return "auto"
	case "any":
		return "required"
	case "none":
		return "none"
	case "tool":
		if tc.Name != "" {
			return map[string]interface{}{
				"type":     "function",
				"function": map[string]string{"name": tc.Name},
			}
		}
		return "auto"
	default:
		return nil
	}
}

// NewMessageID generates a message ID in Anthropic format.
func NewMessageID() string {
	return "msg_" + newID()
}

// NewToolUseID generates a tool_use block ID in Anthropic format.
func NewToolUseID() string {
	return "toolu_" + newID()
}

// streamArguments accepts both the protocol's usual JSON string and
// gateways that stream object fragments directly. Unlike ToolArguments,
// it does NOT require object fragments to be valid complete JSON: partial
// objects (e.g. `{"command": "ls"` followed by `}` in a later chunk) are
// kept verbatim so argument chunks can be accumulated as text by the
// caller. Requiring json.Valid here caused entire SSE chunks to be
// dropped when upstream fragmented arguments as objects, which made
// Claude Code see empty tool inputs ("required parameter missing").
type streamArguments string

func (a *streamArguments) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		*a = ""
		return nil
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return err
		}
		*a = streamArguments(s)
		return nil
	}
	// Object or partial object: pass through verbatim. Fragments are
	// accumulated and re-validated by the caller at emit time.
	*a = streamArguments(trimmed)
	return nil
}

// stripThinkTags removes Kimi/DeepSeek-style <think>...</think> blocks
// from assistant text. This content is the upstream model's internal
// reasoning; Anthropic clients would otherwise render it as ordinary
// text (including the raw <think> tags).
//
// stripThinkTags is only correct on a complete document. The streaming path
// instead uses thinkStripper, which holds back a trailing partial tag prefix
// so a "<think>" split across two flushes is still recognized.
func stripThinkTags(s string) string {
	var b strings.Builder
	for {
		start := strings.Index(s, "<think>")
		if start < 0 {
			b.WriteString(s)
			break
		}
		b.WriteString(s[:start])
		s = s[start+len("<think>"):]
		end := strings.Index(s, "</think>")
		if end < 0 {
			break // unterminated: drop remainder
		}
		s = s[end+len("</think>"):]
	}
	return b.String()
}

// thinkStripper is a stateful streaming <think> filter. Feed it successive
// text fragments; it returns the text that is safe to emit now, retaining any
// trailing partial "<think" prefix and any not-yet-closed think body so a tag
// split across fragments is never leaked to the client. Call flush at end of
// stream to force out any remaining visible text.
type thinkStripper struct {
	inThink bool
	// hold buffers text we cannot emit yet: a partial tag prefix, or the body
	// of an unterminated <think> block.
	hold strings.Builder
}

// thinkOpen / thinkClose are the tag literals we filter.
const thinkOpen = "<think>"
const thinkClose = "</think>"

// Feed processes one fragment and returns text safe to emit now.
func (t *thinkStripper) Feed(s string) string {
	t.hold.WriteString(s)
	return t.drain(false)
}

// Flush forces out any remaining emit-safe text at end of stream. Anything
// still held (an unterminated think block, or a dangling partial tag) is
// dropped as internal reasoning.
func (t *thinkStripper) Flush() string {
	return t.drain(true)
}

func (t *thinkStripper) drain(final bool) string {
	var out strings.Builder
	for {
		buf := t.hold.String()
		if t.inThink {
			end := strings.Index(buf, thinkClose)
			if end < 0 {
				// Whole buffer is think body; keep holding until the closer
				// arrives. On final flush we drop it.
				if final {
					t.hold.Reset()
					t.inThink = false
				}
				break
			}
			// Discard the think body up to and including the closer.
			t.hold.Reset()
			t.hold.WriteString(buf[end+len(thinkClose):])
			t.inThink = false
			continue
		}
		start := strings.Index(buf, thinkOpen)
		if start < 0 {
			// No complete opener. Emit everything except a trailing prefix of
			// "<think>" that a later fragment might complete.
			keep := trailingTagPrefixLen(buf, thinkOpen)
			emit := buf[:len(buf)-keep]
			if emit != "" {
				out.WriteString(emit)
			}
			t.hold.Reset()
			t.hold.WriteString(buf[len(buf)-keep:])
			if final && keep > 0 {
				// A dangling partial tag at end of stream is not visible text.
				t.hold.Reset()
			}
			break
		}
		// Emit text before the opener, enter the think block.
		if start > 0 {
			out.WriteString(buf[:start])
		}
		t.hold.Reset()
		t.hold.WriteString(buf[start+len(thinkOpen):])
		t.inThink = true
	}
	return out.String()
}

// trailingTagPrefixLen returns the length of the longest suffix of s that is a
// proper prefix of tag (i.e. could be completed into tag by a later fragment).
func trailingTagPrefixLen(s, tag string) int {
	max := len(tag) - 1
	if max > len(s) {
		max = len(s)
	}
	for l := max; l > 0; l-- {
		if strings.HasSuffix(s, tag[:l]) {
			return l
		}
	}
	return 0
}

// extractMessageText accepts the string and content-block shapes upstream
// returns for message.content and returns the visible text. Some gateways
// return multimodal blocks even for text-only replies; a plain string type
// assertion would silently drop the whole answer.
func extractMessageText(v interface{}) string {
	switch value := v.(type) {
	case string:
		return value
	case []interface{}:
		parts := make([]string, 0, len(value))
		for _, item := range value {
			block, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			blockType, _ := block["type"].(string)
			if blockType != "" && blockType != "text" && blockType != "output_text" {
				continue
			}
			if text, ok := block["text"].(string); ok {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "\n")
	default:
		return ""
	}
}

// repairJsonBoundaries repairs arguments text that contains adjacent top-level
// JSON values (e.g. `{"a":1}{"b":2}`) — a sign that a gateway merged multiple
// argument fragments. Only the last complete value is kept, matching how
// OpenAI/Anthropic clients use incremental deltas. A JSON decoder is used so
// `}{` inside a string value is not mistaken for a boundary.
func repairJsonBoundaries(args string) string {
	if json.Valid([]byte(args)) {
		return args
	}
	decoder := json.NewDecoder(strings.NewReader(args))
	var last json.RawMessage
	for {
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			break
		}
		last = value
	}
	if len(last) > 0 && !json.Valid(last) {
		return args
	}
	if len(last) > 0 {
		return string(last)
	}
	return args
}

// fallbackToolName guards against upstream tool_calls that arrive without a
// function name (observed from Kimi on parallel tool calls). Emitting a
// tool_use block with an empty name corrupts Anthropic clients: Claude Code
// persists the block and later crashes on `s.name.length` every turn. A
// harmless placeholder name keeps the turn valid — the client reports "no
// such tool" for that call and continues, instead of bricking the session.
const fallbackToolName = "unknown_tool"

func toolNameOrFallback(name string, logWarn func(msg string, args ...interface{})) string {
	if name != "" {
		return name
	}
	if logWarn != nil {
		logWarn("upstream tool_call missing function name, substituting fallback")
	}
	return fallbackToolName
}

// normalizeToolArgsFromValue accepts the object/string shapes upstream sends
// for tool arguments and returns the string form (or "" when unusable).
func normalizeToolArgsFromValue(v interface{}) string {
	switch val := v.(type) {
	case string:
		return val
	case map[string]interface{}:
		b, err := json.Marshal(val)
		if err != nil {
			return ""
		}
		return string(b)
	case []interface{}:
		b, err := json.Marshal(val)
		if err != nil {
			return ""
		}
		return string(b)
	default:
		return ""
	}
}

// StreamChunk represents a parsed SSE chunk from JoyCode.
type StreamChunk struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				ID       string `json:"id"`
				Index    int    `json:"index"`
				Type     string `json:"type"`
				Function struct {
					Name      string          `json:"name"`
					Arguments streamArguments `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
}

// ParseStreamChunk parses a single SSE data line into a StreamChunk.
func ParseStreamChunk(line string) *StreamChunk {
	line = strings.TrimPrefix(line, "data: ")
	line = strings.TrimSpace(line)
	if line == "" || line == "[DONE]" {
		return nil
	}
	var chunk StreamChunk
	if err := json.Unmarshal([]byte(line), &chunk); err != nil {
		return nil
	}
	return &chunk
}

// ParseStreamDelta extracts text content from an OpenAI SSE chunk.
func ParseStreamDelta(line string) string {
	chunk := ParseStreamChunk(line)
	if chunk == nil || len(chunk.Choices) == 0 {
		return ""
	}
	return chunk.Choices[0].Delta.Content
}

// FormatSSE writes a single SSE event to the writer.
func FormatSSE(w interface{ Write([]byte) (int, error) }, event string, data interface{}) {
	jsonData, _ := json.Marshal(data)
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, jsonData)
}
