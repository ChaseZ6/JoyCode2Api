package openai

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

type chatToolStream struct {
	Index  int
	CallID string
	Name   string
}

type chatResponsesStreamState struct {
	w             http.ResponseWriter
	flusher       http.Flusher
	request       *http.Request
	id            string
	created       int64
	model         string
	includeUsage  bool
	finishReason  string
	inputTokens   int
	outputTokens  int
	terminal      bool
	failed        bool
	tools         map[string]*chatToolStream
	nextToolIndex int
}

func newChatResponsesStreamState(w http.ResponseWriter, flusher http.Flusher, r *http.Request, model string, includeUsage bool) *chatResponsesStreamState {
	return &chatResponsesStreamState{
		w: w, flusher: flusher, request: r,
		id: "chatcmpl-" + newShortID(), created: time.Now().Unix(), model: model,
		includeUsage: includeUsage, finishReason: "stop", tools: make(map[string]*chatToolStream),
	}
}

func (s *chatResponsesStreamState) emit(delta map[string]interface{}, finish interface{}) {
	writeSSEData(s.w, s.flusher, map[string]interface{}{
		"id": s.id, "object": "chat.completion.chunk", "created": s.created, "model": s.model,
		"choices": []map[string]interface{}{{"index": 0, "delta": delta, "finish_reason": finish}},
	})
}

func (s *chatResponsesStreamState) emitStart() {
	s.emit(map[string]interface{}{"role": "assistant", "content": ""}, nil)
}

func (s *chatResponsesStreamState) captureUsage(event *responsesStreamEvent) {
	if event.Response.Usage != nil {
		s.inputTokens = event.Response.Usage.InputTokens
		s.outputTokens = event.Response.Usage.OutputTokens
	}
}

func (s *chatResponsesStreamState) fail(message, code string) {
	if s.terminal || s.failed {
		return
	}
	s.failed = true
	writeSSEError(s.w, s.flusher, message, "api_error", code)
}

func (s *chatResponsesStreamState) toolKey(itemID, callID string) string {
	if itemID != "" {
		return "item:" + itemID
	}
	if callID != "" {
		return "call:" + callID
	}
	return ""
}

func (s *chatResponsesStreamState) addTool(itemID, callID, name string) error {
	key := s.toolKey(itemID, callID)
	if key == "" || callID == "" || name == "" {
		return fmt.Errorf("function_call item is missing id, call_id, or name")
	}
	tool := &chatToolStream{Index: s.nextToolIndex, CallID: callID, Name: name}
	s.nextToolIndex++
	s.tools[key] = tool
	// Register both keys because later argument events vary by upstream version.
	s.tools["call:"+callID] = tool
	if itemID != "" {
		s.tools["item:"+itemID] = tool
	}
	s.finishReason = "tool_calls"
	s.emit(map[string]interface{}{"tool_calls": []map[string]interface{}{{
		"index": tool.Index, "id": tool.CallID, "type": "function",
		"function": map[string]interface{}{"name": tool.Name, "arguments": ""},
	}}}, nil)
	return nil
}

func (s *chatResponsesStreamState) toolDelta(itemID, callID, delta string) error {
	var tool *chatToolStream
	if itemID != "" {
		tool = s.tools["item:"+itemID]
	}
	if tool == nil && callID != "" {
		tool = s.tools["call:"+callID]
	}
	if tool == nil {
		unique := make(map[int]*chatToolStream)
		for _, candidate := range s.tools {
			unique[candidate.Index] = candidate
		}
		if len(unique) == 1 {
			for _, candidate := range unique {
				tool = candidate
			}
		}
	}
	if tool == nil {
		return fmt.Errorf("cannot route function arguments delta to a tool call")
	}
	if delta != "" {
		s.emit(map[string]interface{}{"tool_calls": []map[string]interface{}{{
			"index": tool.Index, "function": map[string]interface{}{"arguments": delta},
		}}}, nil)
	}
	return nil
}

func (s *chatResponsesStreamState) complete(finish string) {
	if s.terminal || s.failed {
		return
	}
	s.terminal = true
	if finish != "" {
		s.finishReason = finish
	}
	s.emit(map[string]interface{}{}, s.finishReason)
	if s.includeUsage {
		writeSSEData(s.w, s.flusher, map[string]interface{}{
			"id": s.id, "object": "chat.completion.chunk", "created": s.created, "model": s.model,
			"choices": []interface{}{},
			"usage": map[string]interface{}{
				"prompt_tokens": s.inputTokens, "completion_tokens": s.outputTokens,
				"total_tokens": s.inputTokens + s.outputTokens,
			},
		})
	}
	writeSSEDone(s.w, s.flusher)
	if s.inputTokens > 0 || s.outputTokens > 0 {
		store.SetTokenUsage(s.request, s.inputTokens, s.outputTokens)
	}
}

type upstreamStreamResult struct {
	Response *http.Response
	Err      error
}

var upstreamConnectHeartbeatInterval = 5 * time.Second

func connectUpstreamStream(
	w http.ResponseWriter,
	flusher http.Flusher,
	r *http.Request,
	connect func() (*http.Response, error),
) (*http.Response, bool) {
	result := make(chan upstreamStreamResult, 1)
	go func() {
		response, err := connect()
		result <- upstreamStreamResult{Response: response, Err: err}
	}()

	ticker := time.NewTicker(upstreamConnectHeartbeatInterval)
	defer ticker.Stop()
	committed := false
	for {
		select {
		case <-r.Context().Done():
			return nil, committed
		case <-ticker.C:
			if !committed {
				prepareSSE(w)
				w.WriteHeader(http.StatusOK)
				committed = true
			}
			writeSSEHeartbeat(w, flusher)
		case outcome := <-result:
			if outcome.Err != nil {
				if committed {
					writeSSEError(w, flusher, outcome.Err.Error(), "api_error", "upstream_error")
				} else {
					writeOpenAIError(w, upstreamHTTPStatus(outcome.Err), outcome.Err.Error(), "api_error", "upstream_error", "")
				}
				return nil, committed
			}
			return outcome.Response, committed
		}
	}
}

func connectResponsesStream(
	w http.ResponseWriter,
	flusher http.Flusher,
	r *http.Request,
	client *joycode.Client,
	body map[string]interface{},
) (*http.Response, bool) {
	return connectUpstreamStream(w, flusher, r, func() (*http.Response, error) {
		return client.PostResponsesStream(body)
	})
}

func decodeResponsesStreamEvent(data []byte) (responsesStreamEvent, error) {
	var event responsesStreamEvent
	if err := json.Unmarshal(data, &event); err != nil {
		return event, err
	}
	if event.Type == "" {
		return event, fmt.Errorf("stream event is missing type")
	}
	return event, nil
}

type responseStreamRead struct {
	Event *responsesStreamEvent
	Err   error
}

func readResponsesSSE(body io.Reader) <-chan responseStreamRead {
	out := make(chan responseStreamRead)
	go func() {
		defer close(out)
		scanner := bufio.NewScanner(body)
		scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		for scanner.Scan() {
			line := scanner.Text()
			for strings.HasPrefix(line, "data: ") {
				line = strings.TrimPrefix(line, "data: ")
			}
			line = strings.TrimSpace(line)
			if line == "" || line == "[DONE]" || !strings.HasPrefix(line, "{") {
				continue
			}
			var raw map[string]interface{}
			if err := json.Unmarshal([]byte(line), &raw); err != nil {
				out <- responseStreamRead{Err: fmt.Errorf("invalid Responses stream event: %w", err)}
				return
			}
			if err := joycode.DetectBusinessError(raw); err != nil {
				out <- responseStreamRead{Err: err}
				return
			}
			event, err := decodeResponsesStreamEvent([]byte(line))
			if err != nil {
				out <- responseStreamRead{Err: fmt.Errorf("invalid Responses stream event: %w", err)}
				return
			}
			out <- responseStreamRead{Event: &event}
		}
		if err := scanner.Err(); err != nil {
			out <- responseStreamRead{Err: fmt.Errorf("read Responses stream: %w", err)}
		}
	}()
	return out
}
