package store

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"
)

type RequestLogEntry struct {
	UserID, Model, Endpoint, ErrorMessage string
	Stream                                bool
	StatusCode, InputTokens, OutputTokens int
	LatencyMs                             int64
}

type RequestLogWriter struct {
	store     *Store
	queue     chan RequestLogEntry
	stop      chan struct{}
	done      chan struct{}
	mu        sync.RWMutex
	accepting bool
	dropped   atomic.Uint64
}

func NewRequestLogWriter(store *Store, capacity int) *RequestLogWriter {
	if capacity < 1 {
		capacity = 1
	}
	writer := &RequestLogWriter{store: store, queue: make(chan RequestLogEntry, capacity), stop: make(chan struct{}), done: make(chan struct{}), accepting: true}
	go writer.run()
	return writer
}

func (w *RequestLogWriter) Enqueue(entry RequestLogEntry) bool {
	if w == nil {
		return false
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	if !w.accepting {
		return false
	}
	select {
	case w.queue <- entry:
		return true
	default:
		count := w.dropped.Add(1)
		if count == 1 || count%100 == 0 {
			slog.Warn("request log queue full; dropping entries", "dropped", count, "capacity", cap(w.queue))
		}
		return false
	}
}

func (w *RequestLogWriter) Dropped() uint64 {
	if w == nil {
		return 0
	}
	return w.dropped.Load()
}

func (w *RequestLogWriter) write(entry RequestLogEntry) {
	if err := w.store.LogRequest(entry.UserID, entry.Model, entry.Endpoint, entry.Stream, entry.StatusCode, entry.LatencyMs, entry.ErrorMessage, entry.InputTokens, entry.OutputTokens); err != nil {
		slog.Error("request log writer failed", "endpoint", entry.Endpoint, "error", err)
	}
}

func (w *RequestLogWriter) run() {
	defer close(w.done)
	for {
		select {
		case entry := <-w.queue:
			w.write(entry)
		case <-w.stop:
			for {
				select {
				case entry := <-w.queue:
					w.write(entry)
				default:
					return
				}
			}
		}
	}
}

func (w *RequestLogWriter) Shutdown(ctx context.Context) error {
	if w == nil {
		return nil
	}
	w.mu.Lock()
	if w.accepting {
		w.accepting = false
		close(w.stop)
	}
	w.mu.Unlock()
	select {
	case <-w.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
