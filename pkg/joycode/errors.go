package joycode

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
)

// HTTPError preserves an upstream non-2xx response for protocol-aware mapping.
type HTTPError struct {
	StatusCode int
	Body       string
	Header     http.Header
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("API error %d: %s", e.StatusCode, e.Body)
}

func (e *HTTPError) RetryAfter() string {
	if e.Header == nil {
		return ""
	}
	return e.Header.Get("Retry-After")
}

func NewHTTPError(status int, body string, header http.Header) *HTTPError {
	return &HTTPError{StatusCode: status, Body: body, Header: header.Clone()}
}

// UpstreamError is a business error returned inside an HTTP 200 response.
// JoyCode uses several envelope shapes, so callers should use errors.As rather
// than parse error strings.
type UpstreamError struct {
	Code    string
	Message string
	Cause   string
}

func (e *UpstreamError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("upstream error %s: %s", e.Code, e.Message)
	}
	return "upstream error: " + e.Message
}

// DetectBusinessError recognizes JoyCode and Responses error envelopes. A nil
// error field and code 0 are successful responses.
func DetectBusinessError(v map[string]interface{}) error {
	if v == nil {
		return nil
	}
	if raw, exists := v["error"]; exists && raw != nil {
		code, message, cause := errorFields(raw)
		if message == "" {
			message = firstString(v, "message", "msg", "echo")
		}
		if message == "" {
			b, _ := json.Marshal(raw)
			message = string(b)
		}
		return &UpstreamError{Code: code, Message: message, Cause: cause}
	}

	if raw, exists := v["code"]; exists {
		code := scalarString(raw)
		if code != "" && code != "0" && code != "200" {
			message := firstString(v, "message", "msg", "echo")
			if message == "" {
				message = "unknown error"
			}
			return &UpstreamError{Code: code, Message: message}
		}
	}
	return nil
}

func errorFields(raw interface{}) (code, message, cause string) {
	switch e := raw.(type) {
	case string:
		message = e
	case map[string]interface{}:
		code = scalarString(e["code"])
		message = firstString(e, "message", "msg")
		cause = firstString(e, "cause")
		if message == "" {
			message = cause
		}
	default:
		message = scalarString(raw)
	}
	return
}

func scalarString(v interface{}) string {
	switch n := v.(type) {
	case string:
		return strings.TrimSpace(n)
	case json.Number:
		return n.String()
	case float64:
		return strconv.FormatFloat(n, 'f', -1, 64)
	case float32:
		return strconv.FormatFloat(float64(n), 'f', -1, 32)
	case int:
		return strconv.Itoa(n)
	case int64:
		return strconv.FormatInt(n, 10)
	}
	return ""
}

func firstString(m map[string]interface{}, keys ...string) string {
	for _, key := range keys {
		if s, ok := m[key].(string); ok && strings.TrimSpace(s) != "" {
			return strings.TrimSpace(s)
		}
	}
	return ""
}
