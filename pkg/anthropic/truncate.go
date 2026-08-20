package anthropic

import (
	"encoding/json"
	"log/slog"
)

const (
	// Default upstream model context window size.
	contextWindowSize = 196608
	// Safety margin: start truncation when estimated tokens exceed this ratio of contextWindowSize
	preemptiveThresholdRatio = 0.85
	// Rough approximation: 1 token ≈ 3.5 bytes for mixed Chinese/English code content
	bytesPerToken = 3.5
	// Maximum truncation rounds before giving up
	maxTruncationRounds = 5
	contextSafetyTokens = 8192
	// max_tokens is a generation ceiling. Claude Code requests the model cap
	// (currently 64K) on every turn, but typical tool turns use far less. Treating
	// the ceiling as guaranteed output discarded useful history on every request.
	preemptiveOutputReserve = 16384
	truncationNotice        = "Earlier conversation messages were auto-truncated to fit the model context window. Continue from the remaining recent conversation and tool results."
)

// EstimateInputTokens provides a deterministic local estimate for Anthropic
// compatibility endpoints. It is not a model tokenizer, but includes the
// structural inputs that clients expect count_tokens to reflect.
func EstimateInputTokens(req *MessageRequest) int {
	totalBytes := len(req.System) + len(req.ToolChoice)
	for _, tool := range req.Tools {
		encoded, _ := json.Marshal(tool)
		totalBytes += len(encoded) + 32
	}
	for _, message := range req.Messages {
		totalBytes += 16 + len(message.Role)
		var blocks []contentBlock
		if json.Unmarshal(message.Content, &blocks) != nil {
			totalBytes += len(message.Content)
			continue
		}
		for _, block := range blocks {
			totalBytes += 24 + len(block.Type) + len(block.Text) + len(block.ID) + len(block.Name) + len(block.Input) + len(block.ToolUseID) + len(block.Content)
			if block.Type == "image" {
				// Avoid counting an entire base64 payload as text. A conservative
				// visual budget is closer to real multimodal tokenization.
				totalBytes += int(1600 * bytesPerToken)
			}
		}
	}
	if totalBytes == 0 {
		return 0
	}
	return int(float64(totalBytes)/bytesPerToken + 0.999)
}

func estimateTokens(req *MessageRequest) int {
	return EstimateInputTokens(req)
}

// PreemptiveTruncate checks if the request likely exceeds the model's context limit
// and proactively truncates before sending. Returns the number of truncation rounds performed.
// Returns -1 if truncation failed to bring estimated tokens below threshold.
func PreemptiveTruncate(req *MessageRequest) int {
	return PreemptiveTruncateWithWindow(req, contextWindowSize)
}

// PreemptiveTruncateWithWindow performs preflight truncation using the resolved
// model's advertised context window.
func PreemptiveTruncateWithWindow(req *MessageRequest, contextWindow int) int {
	if contextWindow <= 0 {
		contextWindow = contextWindowSize
	}
	window := float64(contextWindow)
	ratioThreshold := int(window * float64(preemptiveThresholdRatio))
	// Keep room for a normal long generation while allowing the upstream to use
	// more when the input is smaller. req.MaxTokens is a ceiling, not a target.
	outputBudget := req.MaxTokens
	if outputBudget < 0 {
		outputBudget = 0
	}
	if outputBudget > preemptiveOutputReserve {
		outputBudget = preemptiveOutputReserve
	}
	combinedThreshold := contextWindow - outputBudget - contextSafetyTokens
	if combinedThreshold < 1 {
		combinedThreshold = 1
	}
	threshold := ratioThreshold
	if combinedThreshold < threshold {
		threshold = combinedThreshold
	}

	rounds := 0
	for rounds < maxTruncationRounds {
		estimated := estimateTokens(req)
		if estimated <= threshold {
			if rounds > 0 {
				slog.Info("preemptive truncation complete", "rounds", rounds, "estimated_tokens", estimated, "threshold", threshold)
			}
			return rounds
		}
		if !truncateMessages(req) {
			slog.Warn("preemptive truncation: cannot truncate further", "estimated_tokens", estimated, "threshold", threshold, "rounds", rounds)
			return -1
		}
		rounds++
		slog.Warn("preemptive truncation round", "round", rounds, "estimated_tokens_before", estimated, "threshold", threshold)
	}
	// Check if we actually got below threshold
	if estimateTokens(req) > threshold {
		slog.Warn("preemptive truncation exhausted rounds without reaching threshold")
		return -1
	}
	return rounds
}

func messageBlockIDs(message MessageParam, blockType string) map[string]bool {
	ids := make(map[string]bool)
	var blocks []contentBlock
	if json.Unmarshal(message.Content, &blocks) != nil {
		return ids
	}
	for _, block := range blocks {
		if block.Type != blockType {
			continue
		}
		id := block.ID
		if blockType == "tool_result" {
			id = block.ToolUseID
		}
		if id != "" {
			ids[id] = true
		}
	}
	return ids
}

func isCompleteToolExchangeStart(messages []MessageParam, index int) bool {
	if index < 0 || index+1 >= len(messages) || messages[index].Role != "assistant" || messages[index+1].Role != "user" {
		return false
	}
	uses := messageBlockIDs(messages[index], "tool_use")
	results := messageBlockIDs(messages[index+1], "tool_result")
	if len(uses) == 0 || len(results) == 0 {
		return false
	}
	for id := range results {
		if !uses[id] {
			return false
		}
	}
	return true
}

// findSafeTruncationStart finds a coherent beginning for the retained suffix:
// either a regular user turn or an assistant tool_use followed by its results.
func findSafeTruncationStart(messages []MessageParam, target int) int {
	if target < 1 {
		target = 1
	}
	for i := target; i < len(messages); i++ {
		if messages[i].Role == "user" && len(messageBlockIDs(messages[i], "tool_result")) == 0 {
			return i
		}
		if isCompleteToolExchangeStart(messages, i) {
			return i
		}
	}
	return -1
}

func addTruncationNoticeToSystem(req *MessageRequest) {
	if len(req.System) == 0 || string(req.System) == "null" {
		req.System, _ = json.Marshal(truncationNotice)
		return
	}
	if json.Valid(req.System) && string(req.System) != "" {
		var text string
		if json.Unmarshal(req.System, &text) == nil {
			if text == truncationNotice || len(text) >= len(truncationNotice) && text[len(text)-len(truncationNotice):] == truncationNotice {
				return
			}
			req.System, _ = json.Marshal(text + "\n\n" + truncationNotice)
			return
		}

		var blocks []json.RawMessage
		if json.Unmarshal(req.System, &blocks) == nil {
			for _, block := range blocks {
				var parsed contentBlock
				if json.Unmarshal(block, &parsed) == nil && parsed.Type == "text" && parsed.Text == truncationNotice {
					return
				}
			}
			notice, _ := json.Marshal(map[string]string{"type": "text", "text": truncationNotice})
			blocks = append(blocks, notice)
			req.System, _ = json.Marshal(blocks)
		}
	}
}

// truncateMessages removes the oldest messages while retaining a coherent
// recent suffix. Each call removes roughly 40% of remaining messages.
// Returns true if truncation was performed.
func truncateMessages(req *MessageRequest) bool {
	n := len(req.Messages)
	if n <= 4 {
		return false
	}

	keepLast := int(float64(n) * 0.6)
	if keepLast < 4 {
		keepLast = 4
	}
	cutStart := findSafeTruncationStart(req.Messages, n-keepLast)
	if cutStart <= 0 || cutStart >= n {
		return false
	}

	removed := cutStart - 1
	truncated := make([]MessageParam, 0, n-cutStart+1)
	truncated = append(truncated, req.Messages[0])
	truncated = append(truncated, req.Messages[cutStart:]...)
	addTruncationNoticeToSystem(req)

	slog.Warn("auto-truncated messages for context limit",
		"original_count", n,
		"truncated_count", len(truncated),
		"removed", removed,
		"kept_first", 1,
		"kept_last", n-cutStart,
	)

	req.Messages = truncated
	return true
}
