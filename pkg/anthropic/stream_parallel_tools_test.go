package anthropic

import (
	"strings"
	"testing"
)

// Verify the exact failure from the bug report: two parallel tool calls with
// interleaved deltas produce two complete, separate tool_use blocks with
// correct un-merged JSON arguments.
func TestStream_ParallelToolsNotMerged(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c0\",\"function\":{\"name\":\"Read\",\"arguments\":\"{\\\"file_path\\\": \\\"/a.md\\\"\"}}]}}]}\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"c1\",\"function\":{\"name\":\"Read\",\"arguments\":\"{\\\"file_path\\\": \\\"/b.md\\\"\"}}]}}]}\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"}\"}}]}}]}\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"}\"}}]}}]}\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n"
	out := runStream(t, bodyOf(sse))
	events := parseSSE(t, out)
	checkBlockInvariants(t, events)

	// Collect full arguments per tool_use block by accumulating deltas
	blocks := []map[string]interface{}{}
	argsPerBlock := make(map[int]string)
	currentIdx := -1
	for _, e := range events {
		switch e.Event {
		case "content_block_start":
			if cb, ok := e.Data["content_block"].(map[string]interface{}); ok && cb["type"] == "tool_use" {
				blocks = append(blocks, cb)
				currentIdx = len(blocks) - 1
			}
		case "content_block_delta":
			if d, ok := e.Data["delta"].(map[string]interface{}); ok && d["type"] == "input_json_delta" {
				if currentIdx >= 0 {
					argsPerBlock[currentIdx] += d["partial_json"].(string)
				}
			}
		}
	}
	if len(blocks) != 2 {
		t.Fatalf("expected 2 tool_use blocks, got %d:\n%s", len(blocks), out)
	}
	if argsPerBlock[0] != `{"file_path": "/a.md"}` {
		t.Errorf("tool 0 args = %q", argsPerBlock[0])
	}
	if argsPerBlock[1] != `{"file_path": "/b.md"}` {
		t.Errorf("tool 1 args = %q", argsPerBlock[1])
	}
	if !strings.Contains(out, `"name":"Read"`) {
		t.Errorf("missing tool name:\n%s", out)
	}
}

// TestStream_ToolArgsStreamIncrementally verifies that tool argument fragments
// are streamed as input_json_delta in real time, not buffered until finish.
// This is the fix for Claude Code seeing empty tool inputs when upstream
// (e.g. Kimi K3) fragments arguments across many chunks.
func TestStream_ToolArgsStreamIncrementally(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"Bash","arguments":"{\"command\":\"ls\",\"timeout\":120"}}]}}]}
` + `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"}"}}]}}]}
` + `data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
`
	out := runStream(t, bodyOf(sse))
	events := parseSSE(t, out)
	checkBlockInvariants(t, events)

	// The first content_block_delta must arrive BEFORE content_block_stop,
	// not buffered until the end.
	var deltaIdx, stopIdx int
	for i, e := range events {
		if e.Event == "content_block_delta" && deltaIdx == 0 {
			deltaIdx = i
		}
		if e.Event == "content_block_stop" {
			stopIdx = i
		}
	}
	if deltaIdx == 0 {
		t.Fatal("no input_json_delta found")
	}
	if stopIdx == 0 {
		t.Fatal("no content_block_stop found")
	}
	if deltaIdx >= stopIdx {
		t.Errorf("input_json_delta must arrive before content_block_stop, got delta at %d, stop at %d", deltaIdx, stopIdx)
	}

	// Accumulated args must form valid JSON containing the expected fields
	var args string
	for _, e := range events {
		d, ok := e.Data["delta"].(map[string]interface{})
		if !ok || d["type"] != "input_json_delta" {
			continue
		}
		if pj, ok2 := d["partial_json"].(string); ok2 {
			args += pj
		}
	}
	if !strings.Contains(args, `"command"`) || !strings.Contains(args, `"ls"`) {
		t.Errorf("accumulated args missing expected fields: %q", args)
	}
}

// TestStream_TextBuffering verifies that small text fragments are coalesced
// into fewer text_delta events instead of being emitted one per upstream
// chunk. This fixes Kimi K3 sending single CJK characters per SSE event.
func TestStream_TextBuffering(t *testing.T) {
	// Simulate Kimi K3 style: one character per SSE chunk
	sse := `data: {"choices":[{"delta":{"content":"这"}}]}
` + `data: {"choices":[{"delta":{"content":"是"}}]}
` + `data: {"choices":[{"delta":{"content":"一"}}]}
` + `data: {"choices":[{"delta":{"content":"个"}}]}
` + `data: {"choices":[{"delta":{"content":"测"}}]}
` + `data: {"choices":[{"delta":{"content":"试"}}]}
` + `data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
`
	out := runStream(t, bodyOf(sse))
	events := parseSSE(t, out)
	checkBlockInvariants(t, events)

	// Count text_delta events; buffering should produce fewer than 6
	var textDeltas int
	for _, e := range events {
		if d, ok := e.Data["delta"].(map[string]interface{}); ok && d["type"] == "text_delta" {
			textDeltas++
		}
	}
	if textDeltas >= 6 {
		t.Errorf("expected fewer than 6 text_delta events due to buffering, got %d", textDeltas)
	}
	// But the full text must still be present
	var fullText string
	for _, e := range events {
		if d, ok := e.Data["delta"].(map[string]interface{}); ok && d["type"] == "text_delta" {
			if txt, ok2 := d["text"].(string); ok2 {
				fullText += txt
			}
		}
	}
	if fullText != "这是一个测试" {
		t.Errorf("full text = %q, want %q", fullText, "这是一个测试")
	}
}

// TestStream_PartialObjectArguments verifies that object-form argument
// fragments (partial, unclosed JSON objects) are NOT dropped. Previously
// joycode.ToolArguments required json.Valid for object fragments, so an
// unclosed object like {"command": "ls" caused the whole SSE chunk to be
// discarded and Claude Code saw empty tool inputs.
func TestStream_PartialObjectArguments(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"Bash","arguments":"{\"command\":\"ls\""}}]}}]}
` + `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":",\"timeout\":120}"}}]}}]}
` + `data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
`
	out := runStream(t, bodyOf(sse))
	events := parseSSE(t, out)
	checkBlockInvariants(t, events)

	var args string
	var blocks int
	for _, e := range events {
		if e.Event == "content_block_start" {
			if cb, ok := e.Data["content_block"].(map[string]interface{}); ok && cb["type"] == "tool_use" {
				blocks++
			}
		}
		if e.Event == "content_block_delta" {
			if d, ok := e.Data["delta"].(map[string]interface{}); ok && d["type"] == "input_json_delta" {
				if pj, ok2 := d["partial_json"].(string); ok2 {
					args += pj
				}
			}
		}
	}
	if blocks != 1 {
		t.Fatalf("expected 1 tool_use block, got %d:\n%s", blocks, out)
	}
	if !strings.Contains(args, `"command"`) || !strings.Contains(args, `"ls"`) {
		t.Errorf("accumulated args missing expected fields: %q", args)
	}
}

// TestStream_ThinkTagsStripped verifies that <think>...</think> reasoning
// fragments inside delta.content are removed before forwarding to Anthropic
// clients, so raw tags never show up in Claude Code.
func TestStream_ThinkTagsStripped(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"content":"<think>Let me reason about this carefully.</think>"}}]}
` + `data: {"choices":[{"delta":{"content":"你好"}}]}
` + `data: {"choices":[{"delta":{"content":"世界"}}]}
` + `data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
`
	out := runStream(t, bodyOf(sse))
	events := parseSSE(t, out)
	checkBlockInvariants(t, events)

	if strings.Contains(out, "think") {
		t.Errorf("output must not contain think tags or reasoning text:\n%s", out)
	}
	var fullText string
	for _, e := range events {
		if d, ok := e.Data["delta"].(map[string]interface{}); ok && d["type"] == "text_delta" {
			if txt, ok2 := d["text"].(string); ok2 {
				fullText += txt
			}
		}
	}
	if fullText != "你好世界" {
		t.Errorf("full text = %q, want %q", fullText, "你好世界")
	}
}

// TestStream_ObjectFormArguments verifies that when upstream sends arguments
// as a complete JSON object (not a string), it is correctly serialized to
// text and streamed as input_json_delta.
func TestStream_ObjectFormArguments(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"Bash","arguments":{"command":"ls","timeout":120}}}]}}]}
` + `data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
`
	out := runStream(t, bodyOf(sse))
	events := parseSSE(t, out)
	checkBlockInvariants(t, events)

	var args string
	for _, e := range events {
		if d, ok := e.Data["delta"].(map[string]interface{}); ok && d["type"] == "input_json_delta" {
			if pj, ok2 := d["partial_json"].(string); ok2 {
				args += pj
			}
		}
	}
	// Object-form arguments should be compacted and streamed
	if !strings.Contains(args, `"command"`) || !strings.Contains(args, `"ls"`) {
		t.Errorf("accumulated args missing expected fields: %q", args)
	}
}

func TestStream_TextAndToolSameChunkPreservesOrder(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"content":"先说明","tool_calls":[{"index":0,"id":"c0","function":{"name":"Bash","arguments":"{\"command\":\"ls\"}"}}]}}]}
` + `data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
`
	out := runStream(t, bodyOf(sse))
	events := parseSSE(t, out)
	checkBlockInvariants(t, events)
	var starts []string
	for _, e := range events {
		if e.Event == "content_block_start" {
			if cb, ok := e.Data["content_block"].(map[string]interface{}); ok {
				if typ, ok := cb["type"].(string); ok {
					starts = append(starts, typ)
				}
			}
		}
	}
	if len(starts) != 2 || starts[0] != "text" || starts[1] != "tool_use" {
		t.Fatalf("block order = %v, want [text tool_use]:\n%s", starts, out)
	}
}

func TestRepairJsonBoundaries(t *testing.T) {
	if got := repairJsonBoundaries(`{"a":1}{"b":2}`); got != `{"b":2}` {
		t.Fatalf("repairJsonBoundaries = %q", got)
	}
	if got := repairJsonBoundaries(`{"a":1}`); got != `{"a":1}` {
		t.Fatalf("repairJsonBoundaries changed valid JSON: %q", got)
	}
	if got := repairJsonBoundaries(`{"pattern":"a}{b"}`); got != `{"pattern":"a}{b"}` {
		t.Fatalf("repairJsonBoundaries must not split inside strings: %q", got)
	}
	if got := repairJsonBoundaries(`{"pattern":"a}{b"}{"command":"ls"}`); got != `{"command":"ls"}` {
		t.Fatalf("repairJsonBoundaries = %q, want last complete object", got)
	}
	if got := repairJsonBoundaries(`{"command":"ls"`); got != `{"command":"ls"` {
		t.Fatalf("repairJsonBoundaries must preserve incomplete JSON: %q", got)
	}
}

// TestStream_JsonBoundaryRepair verifies that when upstream emits adjacent
// JSON objects in tool arguments ({"a":1}{"b":2}), only the last complete
// object reaches the client.
func TestStream_JsonBoundaryRepair(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"Bash","arguments":"{\"command\":\"echo hi\"}{\"command\":\"ls -la\"}"}}]}}]}
` + `data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
`
	out := runStream(t, bodyOf(sse))
	events := parseSSE(t, out)
	checkBlockInvariants(t, events)
	var args string
	for _, e := range events {
		if d, ok := e.Data["delta"].(map[string]interface{}); ok && d["type"] == "input_json_delta" {
			if pj, ok2 := d["partial_json"].(string); ok2 {
				args += pj
			}
		}
	}
	if !strings.Contains(args, `"ls -la"`) {
		t.Errorf("expected repaired (last) object args, got %q\n%s", args, out)
	}
	if strings.Contains(args, `"echo hi"`) {
		t.Errorf("first object must be dropped, got %q", args)
	}
}

// TestStream_JsonBoundaryRepair_SingleActiveTool: when a SINGLE tool receives
// adjacent JSON objects in one chunk (the case that opens its block
// immediately), the boundary must still be repaired at open time. Previously
// repair only ran for buffered tools, so the active tool streamed both objects
// verbatim and the client accumulated invalid JSON.
func TestStream_JsonBoundaryRepair_SingleActiveTool(t *testing.T) {
	sse := `data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"Bash","arguments":"{\"command\":\"echo hi\"}{\"command\":\"ls -la\"}"}}]}}]}
` + `data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
`
	out := runStream(t, bodyOf(sse))
	var args string
	for _, e := range parseSSE(t, out) {
		if d, ok := e.Data["delta"].(map[string]interface{}); ok && d["type"] == "input_json_delta" {
			if pj, ok2 := d["partial_json"].(string); ok2 {
				args += pj
			}
		}
	}
	if !strings.Contains(args, `"ls -la"`) || strings.Contains(args, `"echo hi"`) {
		t.Errorf("single active tool args not repaired, got %q", args)
	}
}

// TestThinkStripper_CrossBoundary: a <think> tag split across two flushes must
// still be stripped, and the partial "<thi" prefix must never reach the client.
func TestThinkStripper_CrossBoundary(t *testing.T) {
	var ts thinkStripper
	got := ts.Feed("hello world <thi")
	got += ts.Feed("nk>secret</think> visible")
	got += ts.Flush()
	if got != "hello world  visible" {
		t.Fatalf("cross-boundary strip = %q, want %q", got, "hello world  visible")
	}
}

// TestThinkStripper_CharByChar: Kimi-style single-character fragments that
// spell out a full think block must all be stripped.
func TestThinkStripper_CharByChar(t *testing.T) {
	var ts thinkStripper
	var got string
	for _, ch := range []string{"<", "t", "h", "i", "n", "k", ">", "推", "理", "<", "/", "t", "h", "i", "n", "k", ">", "答", "案"} {
		got += ts.Feed(ch)
	}
	got += ts.Flush()
	if got != "答案" {
		t.Fatalf("char-by-char strip = %q, want %q", got, "答案")
	}
}

// TestThinkStripper_LessThanNotTag: a literal "<" that never becomes a tag must
// be emitted, not swallowed by the partial-prefix hold-back.
func TestThinkStripper_LessThanNotTag(t *testing.T) {
	var ts thinkStripper
	got := ts.Feed("compare a <") + ts.Feed(" b and c") + ts.Flush()
	if got != "compare a < b and c" {
		t.Fatalf("literal < = %q, want %q", got, "compare a < b and c")
	}
}

// Regression: a parallel tool whose first chunk never carries function.name
// used to be emitted as a tool_use block with an empty name. Claude Code
// persists that block, then crashes on `s.name.length` every subsequent turn.
// The stream must substitute a placeholder name instead of emitting an empty
// one.
func TestStream_ToolCallMissingNameGetsFallback(t *testing.T) {
	sse := "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c0\",\"function\":{\"name\":\"Bash\",\"arguments\":\"{\\\"command\\\":\\\"ls\\\"\"}}]}}]}\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"{\\\"command\\\":\\\"cat SITE_GUIDE.md\\\"\"}}]}}]}\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"}\"}}]}}]}\n" +
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"}\"}}]}}]}\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n"
	out := runStream(t, bodyOf(sse))
	events := parseSSE(t, out)
	checkBlockInvariants(t, events)

	names := []string{}
	for _, e := range events {
		if e.Event != "content_block_start" {
			continue
		}
		cb, ok := e.Data["content_block"].(map[string]interface{})
		if !ok || cb["type"] != "tool_use" {
			continue
		}
		name, _ := cb["name"].(string)
		names = append(names, name)
	}
	if len(names) != 2 {
		t.Fatalf("expected 2 tool_use blocks, got names %v:\n%s", names, out)
	}
	for _, name := range names {
		if name == "" {
			t.Fatalf("tool_use emitted with empty name — this bricks Claude Code sessions:\n%s", out)
		}
	}
	found := false
	for _, name := range names {
		if name == fallbackToolName {
			found = true
		}
	}
	if !found {
		t.Errorf("expected one block to use fallback name %q, got %v", fallbackToolName, names)
	}
}
