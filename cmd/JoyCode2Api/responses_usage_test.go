package main

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

func TestIsProxyInferencePath(t *testing.T) {
	cases := map[string]bool{
		"/v1/responses":        true,
		"/v1/chat/completions": true,
		"/v1/messages":         true,
		"/responses":           true,
		"/responses/extra":     false,
		"/api/stats":           false,
		"/":                    false,
	}
	for path, want := range cases {
		if got := isProxyInferencePath(path); got != want {
			t.Errorf("isProxyInferencePath(%q) = %v, want %v", path, got, want)
		}
	}
}

func TestRequestLogMiddleware_RecordsResponsesAliasUsage(t *testing.T) {
	s, err := store.Open(t.TempDir() + "/proxy.db")
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.AddAccount("account-1", "pt-key", "test", true, "GPT-5.6 Sol"); err != nil {
		t.Fatal(err)
	}

	next := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		store.SetTokenUsage(r, 321, 45)
		store.SetModel(r, "GPT-5.6 Sol")
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"completed"}`))
	})
	handler := requestLogMiddleware(next, s)
	req := httptest.NewRequest(http.MethodPost, "/responses", strings.NewReader(
		`{"model":"gpt-5.6-sol","stream":true,"input":"hello"}`))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	deadline := time.Now().Add(2 * time.Second)
	for {
		logs, err := s.GetRecentLogs(10)
		if err != nil {
			t.Fatal(err)
		}
		if len(logs) > 0 {
			if len(logs) != 1 {
				t.Fatalf("logs = %d, want exactly 1", len(logs))
			}
			log := logs[0]
			if log.Endpoint != "/responses" || !log.Stream || log.InputTokens != 321 || log.OutputTokens != 45 || log.Model != "GPT-5.6 Sol" {
				t.Errorf("unexpected log: %+v", log)
			}
			return
		}
		if time.Now().After(deadline) {
			t.Fatal("timed out waiting for async request log")
		}
		time.Sleep(10 * time.Millisecond)
	}
}
