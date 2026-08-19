package openai

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

// handleNativeResponses exposes the OpenAI Responses endpoint used by Codex and
// CC Switch. The payload is intentionally preserved as a generic JSON object so
// newer Responses fields pass through without waiting for a proxy release.
func (s *Server) handleNativeResponses(w http.ResponseWriter, r *http.Request) {
	if !requirePOST(w, r) {
		return
	}
	decoder := json.NewDecoder(r.Body)
	decoder.UseNumber()
	var body map[string]interface{}
	if err := decoder.Decode(&body); err != nil {
		writeOpenAIError(w, http.StatusBadRequest, "Invalid Responses request: "+err.Error(),
			"invalid_request_error", "invalid_json", "")
		return
	}
	requestedModel, _ := body["model"].(string)
	if strings.TrimSpace(requestedModel) == "" {
		writeOpenAIError(w, http.StatusBadRequest, "model is required",
			"invalid_request_error", "invalid_request", "model")
		return
	}

	systemDefault := ""
	configuredMaxTokens := 0
	if s.store != nil {
		systemDefault = s.store.GetSetting("default_model")
		configuredMaxTokens = s.store.GetIntSetting("default_max_tokens", 0)
	}
	model := ResolveModelInfo(requestedModel, store.GetAccountDefaultModel(r), systemDefault)
	if model.Adapter() != joycode.AdapterOpenAIResponse {
		writeOpenAIError(w, http.StatusBadRequest,
			fmt.Sprintf("model %s is not served by the Responses API", model.DisplayName()),
			"invalid_request_error", "unsupported_model", "model")
		return
	}
	body["model"] = model.APIName()
	store.SetModel(r, model.DisplayName())

	// Codex sends max_output_tokens directly. Apply the same model-aware ceiling
	// as Chat Completions while preserving the public Responses field name.
	requestedMax := intFromJSON(body["max_output_tokens"])
	body["max_output_tokens"] = model.EffectiveMaxTokens(requestedMax, configuredMaxTokens)
	stream, _ := body["stream"].(bool)
	if stream {
		s.handleNativeResponsesStream(w, r, s.getClient(r), body)
		return
	}
	s.handleNativeResponsesNonStream(w, r, s.getClient(r), body)
}

func (s *Server) handleNativeResponsesNonStream(w http.ResponseWriter, r *http.Request, client *joycode.Client, body map[string]interface{}) {
	response, err := client.PostResponses(body)
	if err != nil {
		status := upstreamHTTPStatus(err)
		errorType := "api_error"
		code := "upstream_error"
		if status == http.StatusBadRequest {
			errorType, code = "invalid_request_error", "invalid_request"
		}
		writeOpenAIError(w, status, err.Error(), errorType, code, "")
		return
	}
	if usage, ok := response["usage"].(map[string]interface{}); ok {
		store.SetTokenUsage(r, intFromJSON(usage["input_tokens"]), intFromJSON(usage["output_tokens"]))
	}
	writeJSON(w, http.StatusOK, response)
}

func (s *Server) handleNativeResponsesStream(w http.ResponseWriter, r *http.Request, client *joycode.Client, body map[string]interface{}) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "streaming unsupported by server")
		return
	}
	response, committed := connectResponsesStream(w, flusher, r, client, body)
	if response == nil {
		return
	}
	defer response.Body.Close()
	if !committed {
		prepareSSE(w)
		w.WriteHeader(http.StatusOK)
	}

	reads := readNativeResponsesSSE(response.Body)
	heartbeat := time.NewTicker(5 * time.Second)
	defer heartbeat.Stop()
	var inTk, outTk int
	var hasUsage, usageCommitted, sawTerminal bool
	eventCount := 0
	terminalType := ""
	commitUsage := func() {
		if hasUsage && !usageCommitted {
			store.SetTokenUsage(r, inTk, outTk)
			usageCommitted = true
		}
	}
	defer commitUsage()
	for {
		select {
		case <-r.Context().Done():
			slog.Info("native responses stream client cancelled",
				"events", eventCount, "terminal", terminalType, "in", inTk, "out", outTk)
			return
		case <-heartbeat.C:
			writeSSEHeartbeat(w, flusher)
		case item, open := <-reads:
			if !open {
				if !sawTerminal {
					message := "upstream Responses stream closed before a terminal event"
					store.SetRequestError(r, http.StatusBadGateway, "stream_truncated: "+message)
					slog.Warn("native responses stream truncated",
						"events", eventCount, "in", inTk, "out", outTk)
					writeSSEError(w, flusher, message, "api_error", "stream_truncated")
				}
				return
			}
			if item.Err != nil {
				status := upstreamHTTPStatus(item.Err)
				code := "stream_read_error"
				if status == http.StatusTooManyRequests {
					code = "rate_limit_error"
				} else if _, ok := item.Err.(*joycode.UpstreamError); ok {
					code = "upstream_error"
				}
				store.SetRequestError(r, status, code+": "+item.Err.Error())
				slog.Error("native responses stream read error", "error", item.Err, "events", eventCount, "status", status)
				writeSSEError(w, flusher, item.Err.Error(), "api_error", code)
				return
			}
			eventCount++
			if item.HasUsage {
				inTk, outTk = item.InputTokens, item.OutputTokens
				hasUsage = true
			}
			if item.Terminal {
				terminalType = item.EventType
				switch item.EventType {
				case "response.completed":
					sawTerminal = true
				case "response.incomplete":
					if item.IncompleteReason == "max_output_tokens" {
						sawTerminal = true
					} else {
						message := "upstream response incomplete: " + item.IncompleteReason
						store.SetRequestError(r, http.StatusBadGateway, "response_incomplete: "+message)
						writeSSEError(w, flusher, message, "api_error", "response_incomplete")
						return
					}
				case "response.failed", "error":
					message := item.FailureMessage
					if message == "" || message == "null" {
						message = "upstream response failed"
					}
					store.SetRequestError(r, http.StatusBadGateway, "upstream_error: "+message)
					writeSSEError(w, flusher, message, "api_error", "upstream_error")
					return
				}
				if sawTerminal {
					commitUsage()
				}
			}
			fmt.Fprintf(w, "data: %s\n\n", item.Data)
			flusher.Flush()
		}
	}
}

type nativeResponsesRead struct {
	Data             string
	EventType        string
	IncompleteReason string
	FailureMessage   string
	InputTokens      int
	OutputTokens     int
	HasUsage         bool
	Terminal         bool
	Err              error
}

func readNativeResponsesSSE(body interface{ Read([]byte) (int, error) }) <-chan nativeResponsesRead {
	out := make(chan nativeResponsesRead)
	go func() {
		defer close(out)
		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		var dataLines []string
		emit := func() bool {
			if len(dataLines) == 0 {
				return true
			}
			data := strings.Join(dataLines, "\n")
			dataLines = dataLines[:0]
			for strings.HasPrefix(data, "data:") {
				data = strings.TrimSpace(strings.TrimPrefix(data, "data:"))
			}
			if data == "" || data == "[DONE]" || !strings.HasPrefix(data, "{") {
				return true
			}
			var raw map[string]interface{}
			decoder := json.NewDecoder(strings.NewReader(data))
			decoder.UseNumber()
			if err := decoder.Decode(&raw); err != nil {
				out <- nativeResponsesRead{Err: fmt.Errorf("invalid Responses SSE event: %w", err)}
				return false
			}
			if err := joycode.DetectBusinessError(raw); err != nil {
				out <- nativeResponsesRead{Err: err}
				return false
			}
			item := nativeResponsesRead{Data: data}
			eventType, _ := raw["type"].(string)
			item.EventType = eventType
			switch eventType {
			case "response.completed", "response.incomplete", "response.failed", "error":
				item.Terminal = true
			}
			if response, ok := raw["response"].(map[string]interface{}); ok {
				if usage, ok := response["usage"].(map[string]interface{}); ok {
					item.InputTokens = intFromJSON(usage["input_tokens"])
					item.OutputTokens = intFromJSON(usage["output_tokens"])
					item.HasUsage = true
				}
				if details, ok := response["incomplete_details"].(map[string]interface{}); ok {
					item.IncompleteReason, _ = details["reason"].(string)
				}
				if responseErr := response["error"]; responseErr != nil {
					item.FailureMessage = responseErrorMessage(responseErr)
				}
			}
			if eventType == "error" && item.FailureMessage == "" {
				item.FailureMessage = responseErrorMessage(raw["error"])
			}
			if !item.HasUsage {
				if usage, ok := raw["usage"].(map[string]interface{}); ok {
					item.InputTokens = intFromJSON(usage["input_tokens"])
					item.OutputTokens = intFromJSON(usage["output_tokens"])
					item.HasUsage = true
				}
			}
			out <- item
			return true
		}

		for scanner.Scan() {
			line := scanner.Text()
			if strings.TrimSpace(line) == "" {
				if !emit() {
					return
				}
				continue
			}
			trimmed := strings.TrimLeft(line, " \t")
			if strings.HasPrefix(trimmed, ":") || strings.HasPrefix(trimmed, "event:") ||
				strings.HasPrefix(trimmed, "id:") || strings.HasPrefix(trimmed, "retry:") {
				continue
			}
			if strings.HasPrefix(trimmed, "data:") {
				dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(trimmed, "data:")))
			}
		}
		if !emit() {
			return
		}
		if err := scanner.Err(); err != nil {
			out <- nativeResponsesRead{Err: fmt.Errorf("read Responses SSE: %w", err)}
		}
	}()
	return out
}

func intFromJSON(value interface{}) int {
	switch number := value.(type) {
	case json.Number:
		n, _ := number.Int64()
		return int(n)
	case float64:
		return int(number)
	case int:
		return number
	case int64:
		return int(number)
	}
	return 0
}
