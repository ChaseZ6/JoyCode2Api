package anthropic

// websearch_loop.go — agentic loop for intercepting web_search tool calls.
//
// When Claude Desktop sends a request with the "web_search_20250305" built-in
// tool, the upstream JoyCode Anthropic endpoint does not support it natively.
// This file implements a proxy-side agentic loop:
//
//  1. The model returns stop_reason="tool_use" with a web_search tool_use block.
//  2. We execute the search via joycode.Client.WebSearch().
//  3. We inject the results as a tool_result message and re-send to the model.
//  4. The model's final answer is forwarded to the client.
//
// All other tool types pass through unchanged.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"strings"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
)

const webSearchToolName = "web_search"

// nativeAnthropicRoundResult holds the parsed result of one upstream round-trip.
type nativeAnthropicRoundResult struct {
	// content blocks collected from the stream
	Content    []ContentBlock
	StopReason string
	InputTk    int
	OutputTk   int
	// raw SSE lines buffered for transparent pass-through (stream path)
	RawLines []string
}

// collectNativeAnthropicStream drains one SSE response from the upstream
// Anthropic endpoint into a nativeAnthropicRoundResult.
// It records both the structured content (for tool-use detection) and the
// raw SSE lines (for transparent pass-through to the client).
func collectNativeAnthropicStream(body io.ReadCloser) (*nativeAnthropicRoundResult, error) {
	res := &nativeAnthropicRoundResult{StopReason: "end_turn"}
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var current *ContentBlock
	var inputJSONBuf strings.Builder

	for scanner.Scan() {
		line := scanner.Text()
		res.RawLines = append(res.RawLines, line)

		// Strip ALL leading "data: " prefixes — upstream double-wraps each line.
		payload := line
		for strings.HasPrefix(payload, "data: ") {
			payload = strings.TrimPrefix(payload, "data: ")
		}
		payload = strings.TrimSpace(payload)

		if payload == "" || payload == "[DONE]" {
			continue
		}
		if strings.HasPrefix(payload, "event: ") {
			continue
		}
		if !strings.HasPrefix(payload, "{") {
			continue
		}

		updateNativeAnthropicUsage(payload, &res.InputTk, &res.OutputTk)

		var event struct {
			Type         string          `json:"type"`
			ContentBlock ContentBlock    `json:"content_block"`
			Delta        json.RawMessage `json:"delta"`
		}
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			continue
		}
		switch event.Type {
		case "content_block_start":
			block := event.ContentBlock
			inputJSONBuf.Reset()
			current = &block
		case "content_block_delta":
			if current == nil {
				block := ContentBlock{Type: "text"}
				current = &block
			}
			var delta struct {
				Type        string `json:"type"`
				Text        string `json:"text"`
				PartialJSON string `json:"partial_json"`
			}
			if err := json.Unmarshal(event.Delta, &delta); err != nil {
				continue
			}
			switch delta.Type {
			case "text_delta":
				current.Type = "text"
				current.Text += delta.Text
			case "input_json_delta":
				inputJSONBuf.WriteString(delta.PartialJSON)
			}
		case "content_block_stop":
			if current != nil {
				// For tool_use blocks, assemble input from the accumulated JSON fragments.
				if current.Type == "tool_use" && inputJSONBuf.Len() > 0 {
					assembled := inputJSONBuf.String()
					var v interface{}
					if err := json.Unmarshal([]byte(assembled), &v); err == nil {
						current.Input = v
					} else {
						current.Input = assembled
					}
				}
				res.Content = append(res.Content, *current)
				current = nil
				inputJSONBuf.Reset()
			}
		case "message_delta":
			var wrapper struct {
				Delta deltaStop `json:"delta"`
			}
			if err := json.Unmarshal([]byte(payload), &wrapper); err == nil && wrapper.Delta.StopReason != "" {
				res.StopReason = wrapper.Delta.StopReason
			}
		}
	}
	if current != nil {
		if current.Type == "tool_use" && inputJSONBuf.Len() > 0 {
			assembled := inputJSONBuf.String()
			var v interface{}
			if err := json.Unmarshal([]byte(assembled), &v); err == nil {
				current.Input = v
			} else {
				current.Input = assembled
			}
		}
		res.Content = append(res.Content, *current)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return res, nil
}

// extractWebSearchCalls returns all tool_use blocks with name "web_search".
func extractWebSearchCalls(content []ContentBlock) []ContentBlock {
	var calls []ContentBlock
	for _, b := range content {
		if b.Type == "tool_use" && b.Name == webSearchToolName {
			calls = append(calls, b)
		}
	}
	return calls
}

// buildToolResultInput extracts the query string from a tool_use input.
// Claude sends {"query": "..."} as the input for web_search.
func buildToolResultInput(input interface{}) string {
	if input == nil {
		return ""
	}
	switch v := input.(type) {
	case map[string]interface{}:
		if q, ok := v["query"].(string); ok {
			return q
		}
	case string:
		var m map[string]interface{}
		if err := json.Unmarshal([]byte(v), &m); err == nil {
			if q, ok := m["query"].(string); ok {
				return q
			}
		}
		return v
	}
	b, _ := json.Marshal(input)
	return string(b)
}

// formatSearchResults converts JoyCode web search results into a plain-text
// string suitable for returning as a tool_result content.
func formatSearchResults(results []interface{}) string {
	if len(results) == 0 {
		return "No results found."
	}
	var sb strings.Builder
	for i, r := range results {
		item, ok := r.(map[string]interface{})
		if !ok {
			continue
		}
		title, _ := item["title"].(string)
		url, _ := item["url"].(string)
		snippet, _ := item["content"].(string)
		if snippet == "" {
			snippet, _ = item["snippet"].(string)
		}
		fmt.Fprintf(&sb, "[%d] %s\nURL: %s\n%s\n\n", i+1, title, url, snippet)
	}
	return strings.TrimSpace(sb.String())
}

// executeWebSearchCalls runs all web_search tool_use calls in parallel (sequential
// for simplicity and safety) and returns a slice of tool_result message blocks
// ready to append to the conversation as a user message.
func executeWebSearchCalls(calls []ContentBlock, client *joycode.Client) []map[string]interface{} {
	var results []map[string]interface{}
	for _, call := range calls {
		query := buildToolResultInput(call.Input)
		toolUseID := call.ID
		if toolUseID == "" {
			toolUseID = "toolu_" + newID()
		}

		var resultText string
		if query == "" {
			resultText = "No query provided."
		} else {
			slog.Info("web_search interceptor: executing search", "tool_use_id", toolUseID, "query", query)
			searchResults, err := client.WebSearch(query)
			if err != nil {
				slog.Error("web_search interceptor: search failed", "query", query, "error", err)
				resultText = fmt.Sprintf("Search failed: %s", err.Error())
			} else {
				resultText = formatSearchResults(searchResults)
				slog.Info("web_search interceptor: search completed", "query", query, "results", len(searchResults))
			}
		}

		results = append(results, map[string]interface{}{
			"type":        "tool_result",
			"tool_use_id": toolUseID,
			"content":     resultText,
		})
	}
	return results
}

// appendWebSearchResults builds the follow-up request body by appending:
//  1. An assistant message with the tool_use blocks from the first round.
//  2. A user message with the tool_result blocks.
//
// web_search is removed from the tools list so the model cannot call it again
// in the second round (which would cause an infinite loop).
// The original request body is cloned so the caller's copy is not mutated.
func appendWebSearchResults(originalBody map[string]interface{}, assistantContent []ContentBlock, toolResults []map[string]interface{}) map[string]interface{} {
	// Shallow-clone the body map.
	newBody := make(map[string]interface{}, len(originalBody))
	for k, v := range originalBody {
		newBody[k] = v
	}

	// Remove web_search from the tools list so the model does not call it again.
	// The tools value is []map[string]interface{} (from normalizeToolsForNativeAnthropic).
	if toolsRaw, ok := newBody["tools"]; ok {
		toolsJSON, _ := json.Marshal(toolsRaw)
		var toolsList []map[string]interface{}
		if json.Unmarshal(toolsJSON, &toolsList) == nil {
			filtered := make([]map[string]interface{}, 0, len(toolsList))
			for _, t := range toolsList {
				if t["name"] != webSearchToolName {
					filtered = append(filtered, t)
				}
			}
			if len(filtered) > 0 {
				newBody["tools"] = filtered
			} else {
				delete(newBody, "tools")
				delete(newBody, "tool_choice")
			}
		}
	}

	// Convert assistant content blocks to the raw map format expected by the API.
	assistantBlocks := make([]interface{}, 0, len(assistantContent))
	for _, b := range assistantContent {
		block := map[string]interface{}{"type": b.Type}
		switch b.Type {
		case "text":
			block["text"] = b.Text
		case "tool_use":
			block["id"] = b.ID
			block["name"] = b.Name
			if b.Input != nil {
				block["input"] = b.Input
			} else {
				block["input"] = map[string]interface{}{}
			}
		}
		assistantBlocks = append(assistantBlocks, block)
	}

	// Build the user message carrying the tool results.
	toolResultBlocks := make([]interface{}, 0, len(toolResults))
	for _, tr := range toolResults {
		toolResultBlocks = append(toolResultBlocks, tr)
	}

	// Extend the messages array via JSON round-trip to handle any concrete type.
	var existingMsgs []interface{}
	if raw, err := json.Marshal(newBody["messages"]); err == nil {
		_ = json.Unmarshal(raw, &existingMsgs)
	}
	newMsgs := make([]interface{}, 0, len(existingMsgs)+2)
	newMsgs = append(newMsgs, existingMsgs...)
	newMsgs = append(newMsgs,
		map[string]interface{}{"role": "assistant", "content": assistantBlocks},
		map[string]interface{}{"role": "user", "content": toolResultBlocks},
	)
	newBody["messages"] = newMsgs
	return newBody
}
