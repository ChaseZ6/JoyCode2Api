package anthropic

import (
	"encoding/json"
	"strings"
	"testing"
)

// sseEvt is one parsed Anthropic SSE event.
type sseEvt struct {
	Event string
	Data  map[string]interface{}
}

func parseSSE(t *testing.T, out string) []sseEvt {
	t.Helper()
	var events []sseEvt
	var cur sseEvt
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "event: ") {
			cur = sseEvt{Event: strings.TrimPrefix(line, "event: ")}
		} else if strings.HasPrefix(line, "data: ") {
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &cur.Data); err == nil {
				events = append(events, cur)
			}
			cur = sseEvt{}
		}
	}
	return events
}

// checkBlockInvariants validates Anthropic protocol block ordering:
//  1. at most one open block at a time (start while another is open => fail)
//  2. every delta targets a currently-open block
//  3. stop targets the currently-open block
//  4. every started block is stopped
func checkBlockInvariants(t *testing.T, events []sseEvt) {
	t.Helper()
	open := map[int]bool{}
	var order []int
	top := func() (int, bool) {
		if len(order) == 0 {
			return 0, false
		}
		return order[len(order)-1], true
	}
	for _, e := range events {
		idxF, _ := e.Data["index"].(float64)
		idx := int(idxF)
		switch e.Event {
		case "content_block_start":
			if len(order) > 0 {
				cur, _ := top()
				t.Errorf("content_block_start index=%d while block %d still open", idx, cur)
			}
			if open[idx] {
				t.Errorf("content_block_start reuses index %d", idx)
			}
			open[idx] = true
			order = append(order, idx)
		case "content_block_delta":
			cur, ok := top()
			if !ok {
				t.Errorf("delta for index %d with no open block", idx)
			} else if cur != idx {
				t.Errorf("delta for index %d but open block is %d", idx, cur)
			}
		case "content_block_stop":
			cur, ok := top()
			if !ok {
				t.Errorf("stop for index %d with no open block", idx)
			} else if cur != idx {
				t.Errorf("stop for index %d but open block is %d", idx, cur)
			}
			delete(open, idx)
			if len(order) > 0 {
				order = order[:len(order)-1]
			}
		}
	}
	for idx := range open {
		t.Errorf("block %d never stopped", idx)
	}
}

func idxsOf(events []sseEvt, event string) []int {
	var out []int
	for _, e := range events {
		if e.Event == event {
			if f, ok := e.Data["index"].(float64); ok {
				out = append(out, int(f))
			}
		}
	}
	return out
}

// Regression: interleaved parallel tool calls (tool0 start, tool1 start,
// tool0 args) must be re-chunked into sequential Anthropic blocks — a block
// is stopped before the next starts, and deltas always target the open block.
func TestStream_InterleavedToolBlocks(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c0\",\"function\":{\"name\":\"fa\",\"arguments\":\"{\\\"a\\\":\"}}]}}]}\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"c1\",\"function\":{\"name\":\"fb\",\"arguments\":\"{\\\"b\\\":\"}}]}}]}\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"1}\"}}]}}]}\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"2}\"}}]}}]}\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n"
	out := runStream(t, bodyOf(sse))
	t.Logf("start idxs: %v stop idxs: %v", idxsOf(parseSSE(t, out), "content_block_start"), idxsOf(parseSSE(t, out), "content_block_stop"))
	checkBlockInvariants(t, parseSSE(t, out))
}

// Regression: tool call followed by text followed by another tool call must
// not leak the text block or emit stops with a stale index.
func TestStream_ToolTextToolBlocks(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c0\",\"function\":{\"name\":\"fa\",\"arguments\":\"{}\"}}]}}]}\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"thinking out loud\"}}]}\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"c1\",\"function\":{\"name\":\"fb\",\"arguments\":\"{}\"}}]}}]}\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n"
	out := runStream(t, bodyOf(sse))
	t.Logf("start idxs: %v stop idxs: %v", idxsOf(parseSSE(t, out), "content_block_start"), idxsOf(parseSSE(t, out), "content_block_stop"))
	checkBlockInvariants(t, parseSSE(t, out))
}

func TestStream_ToolCallWithStopFinishReasonUsesToolUse(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c0\",\"function\":{\"name\":\"Write\",\"arguments\":\"{\\\"file_path\\\":\\\"/tmp/a.go\\\"}\"}}]}}]}\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n"
	out := runStream(t, bodyOf(sse))
	if !strings.Contains(out, `"stop_reason":"tool_use"`) {
		t.Fatalf("tool call with finish_reason=stop must remain tool_use:\n%s", out)
	}
	if strings.Contains(out, `"stop_reason":"end_turn"`) {
		t.Fatalf("tool call must not be reported as end_turn:\n%s", out)
	}
}
