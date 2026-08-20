package store

import (
	"context"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestRequestLogWriterShutdownDrainsAcceptedEntries(t *testing.T) {
	s, err := Open(filepath.Join(t.TempDir(), "proxy.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	writer := NewRequestLogWriter(s, 128)
	accepted := 0
	for i := 0; i < 100; i++ {
		if writer.Enqueue(RequestLogEntry{UserID: "u", Model: "m", Endpoint: "/responses", StatusCode: 200, InputTokens: i, OutputTokens: 1}) {
			accepted++
		}
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := writer.Shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	logs, err := s.GetRecentLogs(200)
	if err != nil {
		t.Fatal(err)
	}
	if len(logs) != accepted {
		t.Fatalf("persisted=%d accepted=%d", len(logs), accepted)
	}
	if writer.Enqueue(RequestLogEntry{}) {
		t.Fatal("enqueue succeeded after shutdown")
	}
}

func TestRequestLogWriterConcurrentEnqueueAndShutdown(t *testing.T) {
	s, err := Open(filepath.Join(t.TempDir(), "proxy.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	writer := NewRequestLogWriter(s, 32)
	var wg sync.WaitGroup
	for worker := 0; worker < 8; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				writer.Enqueue(RequestLogEntry{UserID: "u", Endpoint: "/responses", StatusCode: 200})
			}
		}()
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := writer.Shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	wg.Wait()
}

func TestRequestLogWriterQueueIsBounded(t *testing.T) {
	s, err := Open(filepath.Join(t.TempDir(), "proxy.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	writer := NewRequestLogWriter(s, 1)
	if cap(writer.queue) != 1 {
		t.Fatalf("queue capacity=%d", cap(writer.queue))
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := writer.Shutdown(ctx); err != nil {
		t.Fatal(err)
	}
}
