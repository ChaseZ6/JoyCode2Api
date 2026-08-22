package joycode

import (
	"errors"
	"net/http"
	"strings"
	"testing"
)

func TestDetectBusinessError(t *testing.T) {
	cases := []struct {
		name        string
		body        map[string]interface{}
		wantError   bool
		wantCode    string
		wantMessage string
	}{
		{"plain success", map[string]interface{}{"id": "resp_1"}, false, "", ""},
		{"numeric zero", map[string]interface{}{"code": float64(0)}, false, "", ""},
		{"string zero", map[string]interface{}{"code": "0"}, false, "", ""},
		{"HTTP-style 200 code", map[string]interface{}{"code": float64(200)}, false, "", ""},
		{"responses nil error", map[string]interface{}{"status": "completed", "error": nil}, false, "", ""},
		{
			"gateway body limit",
			map[string]interface{}{"code": "1", "echo": "content length exceeded 5242880 bytes"},
			true, "1", "content length exceeded 5242880 bytes",
		},
		{
			"nested 1032",
			map[string]interface{}{"error": map[string]interface{}{"code": "1032", "message": "HTTP调用异常"}},
			true, "1032", "HTTP调用异常",
		},
		{
			"context cause",
			map[string]interface{}{"error": map[string]interface{}{
				"code": float64(400), "message": "模型服务调用失败",
				"cause": `{"error":{"code":"context_length_exceeded"}}`,
			}},
			true, "400", "模型服务调用失败",
		},
		{"string error", map[string]interface{}{"error": "upstream unavailable"}, true, "", "upstream unavailable"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := DetectBusinessError(tc.body)
			if (err != nil) != tc.wantError {
				t.Fatalf("error = %v, wantError=%v", err, tc.wantError)
			}
			if !tc.wantError {
				return
			}
			var upstream *UpstreamError
			if !errors.As(err, &upstream) {
				t.Fatalf("error type = %T, want *UpstreamError", err)
			}
			if upstream.Code != tc.wantCode || upstream.Message != tc.wantMessage {
				t.Errorf("error = code %q message %q, want %q %q",
					upstream.Code, upstream.Message, tc.wantCode, tc.wantMessage)
			}
		})
	}
}

func TestHTTPErrorPreservesStatusAndRetryAfter(t *testing.T) {
	header := http.Header{"Retry-After": []string{"7"}, "X-Upstream": []string{"value"}}
	err := NewHTTPError(http.StatusTooManyRequests, `{"error":"rate limit"}`, header)
	header.Set("Retry-After", "changed")
	if err.StatusCode != http.StatusTooManyRequests || err.RetryAfter() != "7" {
		t.Errorf("HTTPError = %+v", err)
	}
	if err.Error() != `API error 429: {"error":"rate limit"}` {
		t.Errorf("Error() = %q", err.Error())
	}
}

// TestUpstreamErrorIncludesCause: the structured cause (e.g.
// context_length_exceeded) must appear in Error() so callers that string-match
// for context-limit keywords can detect it even when the outer message is a
// generic gateway phrase like "模型服务调用失败".
func TestUpstreamErrorIncludesCause(t *testing.T) {
	err := DetectBusinessError(map[string]interface{}{
		"error": map[string]interface{}{
			"code": float64(400), "message": "模型服务调用失败",
			"cause": `{"error":{"code":"context_length_exceeded"}}`,
		},
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if got := err.Error(); !strings.Contains(got, "context_length_exceeded") {
		t.Errorf("Error() = %q, must surface cause identifier", got)
	}
	// A cause already present in the message must not be duplicated.
	dup := (&UpstreamError{Code: "1", Message: "x context_length_exceeded", Cause: "context_length_exceeded"}).Error()
	if n := strings.Count(dup, "context_length_exceeded"); n != 1 {
		t.Errorf("duplicated cause in Error() = %q (x%d)", dup, n)
	}
}

