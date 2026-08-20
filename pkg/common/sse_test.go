package common

import (
	"context"
	"io"
	"strings"
	"testing"
	"time"
)

func collectSSE(ctx context.Context, input string) []SSERead {
	var reads []SSERead
	for item := range ReadSSE(ctx, strings.NewReader(input)) {
		reads = append(reads, item)
	}
	return reads
}

func TestReadSSEFraming(t *testing.T) {
	input := ": keepalive\n" +
		"event: response.completed\n" +
		"id: event-1\n" +
		"retry: 5000\n" +
		"data:{\"type\":\"response.completed\",\n" +
		"data: \"status\":\"completed\"}\n\n" +
		"data: [DONE]\n\n"
	reads := collectSSE(context.Background(), input)
	if len(reads) != 2 {
		t.Fatalf("reads = %#v", reads)
	}
	first := reads[0].Event
	if first.Event != "response.completed" || first.ID != "event-1" || first.Retry != "5000" ||
		first.Data != "{\"type\":\"response.completed\",\n\"status\":\"completed\"}" || first.Done {
		t.Errorf("first event = %+v", first)
	}
	if !reads[1].Event.Done || reads[1].Event.Data != "[DONE]" {
		t.Errorf("done event = %+v", reads[1].Event)
	}
}

func TestReadSSEFlushesAtEOFAndDoesNotMatchLiteralDone(t *testing.T) {
	reads := collectSSE(context.Background(), `data: {"text":"literal [DONE]"}`)
	if len(reads) != 1 || reads[0].Event.Done || reads[0].Event.Data != `{"text":"literal [DONE]"}` {
		t.Fatalf("reads = %#v", reads)
	}
}

func TestReadSSECancellationClosesChannel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	reader, writer := io.Pipe()
	reads := ReadSSE(ctx, reader)
	writer.Write([]byte("data: one\n\n"))
	if item := <-reads; item.Event.Data != "one" {
		t.Fatalf("first item = %#v", item)
	}
	cancel()
	writer.Close()
	select {
	case _, ok := <-reads:
		if ok {
			t.Fatal("channel still produced events after cancellation")
		}
	case <-time.After(time.Second):
		t.Fatal("SSE reader did not stop after cancellation and body close")
	}
}

func TestUnwrapSSEData(t *testing.T) {
	if got := UnwrapSSEData(` data: data: {"type":"response.completed"} `); got != `{"type":"response.completed"}` {
		t.Errorf("UnwrapSSEData() = %q", got)
	}
}
