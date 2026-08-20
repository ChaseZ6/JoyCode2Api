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

	// Collect full arguments per tool_use block
	var blocks []map[string]interface{}
	var args []string
	for _, e := range events {
		if e.Event == "content_block_start" {
			if cb, ok := e.Data["content_block"].(map[string]interface{}); ok && cb["type"] == "tool_use" {
				blocks = append(blocks, cb)
			}
		}
		if e.Event == "content_block_delta" {
			if d, ok := e.Data["delta"].(map[string]interface{}); ok && d["type"] == "input_json_delta" {
				args = append(args, d["partial_json"].(string))
			}
		}
	}
	if len(blocks) != 2 {
		t.Fatalf("expected 2 tool_use blocks, got %d:\n%s", len(blocks), out)
	}
	if len(args) != 2 {
		t.Fatalf("expected 2 arg deltas, got %d:\n%s", len(args), out)
	}
	if args[0] != `{"file_path": "/a.md"}` {
		t.Errorf("tool 0 args = %q", args[0])
	}
	if args[1] != `{"file_path": "/b.md"}` {
		t.Errorf("tool 1 args = %q", args[1])
	}
	if !strings.Contains(out, `"name":"Read"`) {
		t.Errorf("missing tool name:\n%s", out)
	}
}
