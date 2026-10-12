package common

import (
	"net/http"
	"strings"
)

// IsResponsesWebSocketUpgrade identifies transport negotiation for the two
// Responses aliases, rather than a model inference request or an ordinary GET.
func IsResponsesWebSocketUpgrade(r *http.Request) bool {
	if r.Method != http.MethodGet || (r.URL.Path != "/responses" && r.URL.Path != "/v1/responses") {
		return false
	}
	return headerHasToken(r.Header, "Connection", "upgrade") && headerHasToken(r.Header, "Upgrade", "websocket")
}

func headerHasToken(header http.Header, name, token string) bool {
	for _, value := range header.Values(name) {
		for _, part := range strings.Split(value, ",") {
			if strings.EqualFold(strings.TrimSpace(part), token) {
				return true
			}
		}
	}
	return false
}
