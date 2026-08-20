package common

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"strings"
)

// SSEEvent is one Server-Sent Event after field framing has been decoded.
type SSEEvent struct {
	Event string
	Data  string
	ID    string
	Retry string
	Done  bool
}

// SSERead carries either one event or a terminal read error.
type SSERead struct {
	Event SSEEvent
	Err   error
}

// UnwrapSSEData removes repeated data: wrappers used by some upstream gateways
// after standard SSE framing has already been decoded.
func UnwrapSSEData(data string) string {
	data = strings.TrimSpace(data)
	for strings.HasPrefix(data, "data:") {
		data = strings.TrimSpace(strings.TrimPrefix(data, "data:"))
	}
	return data
}

// ReadSSE decodes SSE framing until EOF or ctx cancellation. JSON and protocol
// semantics intentionally remain the caller's responsibility.
func ReadSSE(ctx context.Context, reader io.Reader) <-chan SSERead {
	out := make(chan SSERead)
	go func() {
		defer close(out)
		scanner := bufio.NewScanner(reader)
		scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
		var event SSEEvent
		var data []string

		send := func(read SSERead) bool {
			select {
			case out <- read:
				return true
			case <-ctx.Done():
				return false
			}
		}
		emit := func() bool {
			if len(data) == 0 {
				event = SSEEvent{}
				return true
			}
			event.Data = strings.Join(data, "\n")
			event.Done = strings.TrimSpace(event.Data) == "[DONE]"
			ok := send(SSERead{Event: event})
			event = SSEEvent{}
			data = data[:0]
			return ok
		}

		for scanner.Scan() {
			line := strings.TrimSuffix(scanner.Text(), "\r")
			if line == "" {
				if !emit() {
					return
				}
				continue
			}
			if strings.HasPrefix(line, ":") {
				continue
			}
			field, value, found := strings.Cut(line, ":")
			if !found {
				field, value = line, ""
			}
			if strings.HasPrefix(value, " ") {
				value = value[1:]
			}
			switch field {
			case "data":
				data = append(data, value)
			case "event":
				event.Event = value
			case "id":
				event.ID = value
			case "retry":
				event.Retry = value
			}
		}
		if !emit() {
			return
		}
		if err := scanner.Err(); err != nil {
			send(SSERead{Err: fmt.Errorf("read SSE: %w", err)})
		}
	}()
	return out
}
