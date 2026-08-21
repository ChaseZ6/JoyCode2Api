package anthropic

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
)

func TestConnectStreamWithRetryStopsAfterGateway504(t *testing.T) {
	var calls atomic.Int32
	client := joycode.NewClient("pt-test", "user-test")
	client.SetHTTPClient(&http.Client{Transport: rtFunc(func(r *http.Request) (*http.Response, error) {
		calls.Add(1)
		return &http.Response{
			StatusCode: http.StatusGatewayTimeout,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader("504 Gateway Time-out")),
		}, nil
	})})
	h := NewHandler(client, nil)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader("{}"))

	_, err := h.connectStreamWithRetry(req, map[string]interface{}{"stream": true}, client)
	if err == nil || !isTimeoutError(err) {
		t.Fatalf("error = %v, want timeout", err)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("upstream calls = %d, want 1 after gateway 504", got)
	}
}

func TestConnectStreamWithRetryStopsWhenRequestCanceled(t *testing.T) {
	var calls atomic.Int32
	client := joycode.NewClient("pt-test", "user-test")
	client.SetHTTPClient(&http.Client{Transport: rtFunc(func(r *http.Request) (*http.Response, error) {
		calls.Add(1)
		return nil, r.Context().Err()
	})})
	h := NewHandler(client, nil)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader("{}")).WithContext(ctx)

	_, err := h.connectStreamWithRetry(req, map[string]interface{}{"stream": true}, client)
	if err == nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("error = %v, want context canceled", err)
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("upstream calls = %d, want 0 for an already-canceled request", got)
	}
}

func TestAnthropicNonStreamGateway504Returns504WithoutUpstreamURL(t *testing.T) {
	client := joycode.NewClient("pt-test", "user-test")
	client.SetHTTPClient(&http.Client{Transport: rtFunc(func(r *http.Request) (*http.Response, error) {
		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body: io.NopCloser(strings.NewReader(
				`{"error":{"code":"COLOR_FORWARD_EXCEPTION","message":"HttpStatus=504"}}`,
			)),
		}, nil
	})})
	h := NewHandler(client, nil)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", strings.NewReader(
		`{"model":"GLM-5.1","max_tokens":32,"messages":[{"role":"user","content":"hi"}]}`,
	))
	rec := httptest.NewRecorder()
	h.handleMessages(rec, req)
	if rec.Code != http.StatusGatewayTimeout {
		t.Fatalf("status = %d, want 504; body = %s", rec.Code, rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), "api-ai.jd.com") || strings.Contains(rec.Body.String(), "COLOR_FORWARD_EXCEPTION") {
		t.Fatalf("response leaks upstream details: %s", rec.Body.String())
	}
}
