package openai

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
)

func prepareSSE(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
}

func writeSSEData(w http.ResponseWriter, flusher http.Flusher, value interface{}) {
	data, err := json.Marshal(value)
	if err != nil {
		writeSSEError(w, flusher, "failed to encode stream event", "api_error", "stream_encode_error")
		return
	}
	fmt.Fprintf(w, "data: %s\n\n", data)
	flusher.Flush()
}

func writeSSEError(w http.ResponseWriter, flusher http.Flusher, message, errorType, code string) {
	errBody := map[string]interface{}{"message": message, "type": errorType}
	if code != "" {
		errBody["code"] = code
	}
	writeSSEData(w, flusher, map[string]interface{}{"error": errBody})
}

func writeSSEDone(w http.ResponseWriter, flusher http.Flusher) {
	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func writeSSEHeartbeat(w http.ResponseWriter, flusher http.Flusher) {
	fmt.Fprint(w, ": keepalive\n\n")
	flusher.Flush()
}

type chatStreamRead struct {
	Data         string
	Done         bool
	FinishReason string
	InputTokens  int
	OutputTokens int
	BusinessErr  error
	Err          error
}

func readChatSSE(body io.Reader) <-chan chatStreamRead {
	out := make(chan chatStreamRead)
	go func() {
		defer close(out)
		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		for scanner.Scan() {
			line := strings.TrimSpace(scanner.Text())
			if line == "" || strings.HasPrefix(line, ":") || strings.HasPrefix(line, "event:") {
				continue
			}
			for strings.HasPrefix(line, "data: ") {
				line = strings.TrimSpace(strings.TrimPrefix(line, "data: "))
			}
			if line == "[DONE]" {
				out <- chatStreamRead{Done: true}
				continue
			}
			if !strings.HasPrefix(line, "{") {
				continue
			}
			var raw map[string]interface{}
			if err := json.Unmarshal([]byte(line), &raw); err != nil {
				out <- chatStreamRead{Err: fmt.Errorf("invalid chat stream JSON: %w", err)}
				return
			}
			if err := joycode.DetectBusinessError(raw); err != nil {
				out <- chatStreamRead{BusinessErr: err}
				return
			}
			item := chatStreamRead{Data: line}
			if usage, ok := raw["usage"].(map[string]interface{}); ok {
				item.InputTokens = numberToInt(usage["prompt_tokens"])
				item.OutputTokens = numberToInt(usage["completion_tokens"])
			}
			if choices, ok := raw["choices"].([]interface{}); ok {
				for _, choice := range choices {
					if m, ok := choice.(map[string]interface{}); ok {
						if reason, ok := m["finish_reason"].(string); ok && reason != "" {
							item.FinishReason = reason
						}
					}
				}
			}
			out <- item
		}
		if err := scanner.Err(); err != nil {
			out <- chatStreamRead{Err: fmt.Errorf("read chat stream: %w", err)}
		}
	}()
	return out
}

func numberToInt(v interface{}) int {
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
