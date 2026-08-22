package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/common"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
)

func prepareSSE(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
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

func readChatSSE(ctx context.Context, body io.Reader) <-chan chatStreamRead {
	out := make(chan chatStreamRead)
	go func() {
		defer close(out)
		send := func(item chatStreamRead) bool {
			select {
			case out <- item:
				return true
			case <-ctx.Done():
				return false
			}
		}
		for read := range common.ReadSSE(ctx, body) {
			if read.Err != nil {
				send(chatStreamRead{Err: fmt.Errorf("read chat stream: %w", read.Err)})
				return
			}
			if read.Event.Done {
				// Do NOT return here: with stream_options.include_usage the
				// usage-only chunk arrives AFTER [DONE]. Returning on [DONE]
				// would drop it and the token stats would always be 0. Forward
				// the Done marker and keep reading until EOF so the trailing
				// usage chunk is captured.
				if !send(chatStreamRead{Done: true}) {
					return
				}
				continue
			}
			data := common.UnwrapSSEData(read.Event.Data)
			if data == "" || !strings.HasPrefix(data, "{") {
				continue
			}
			var raw map[string]interface{}
			if err := json.Unmarshal([]byte(data), &raw); err != nil {
				send(chatStreamRead{Err: fmt.Errorf("invalid chat stream JSON: %w", err)})
				return
			}
			if err := joycode.DetectBusinessError(raw); err != nil {
				send(chatStreamRead{BusinessErr: err})
				return
			}
			item := chatStreamRead{Data: data}
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
			if !send(item) {
				return
			}
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
