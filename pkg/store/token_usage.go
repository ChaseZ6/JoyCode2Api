package store

import (
	"context"
	"net/http"
)

type ctxKey struct{}

type modelCtxKey struct{}

type accountModelCtxKey struct{}

type requestOutcomeCtxKey struct{}

// RequestOutcome records an internal proxy failure after streaming headers were
// already committed. It affects request logging, not the HTTP status already
// observed by the client.
type RequestOutcome struct {
	StatusCode int
	Error      string
}

// InitTokenUsage initializes token usage tracking in the request context.
// Returns a pointer that handlers can update with SetTokenUsage.
func InitTokenUsage(r *http.Request) *http.Request {
	usage := &[2]int{0, 0}
	return r.WithContext(context.WithValue(r.Context(), ctxKey{}, usage))
}

// SetTokenUsage updates the token usage for the request.
func SetTokenUsage(r *http.Request, inputTokens, outputTokens int) {
	if usage, ok := r.Context().Value(ctxKey{}).(*[2]int); ok {
		usage[0] = inputTokens
		usage[1] = outputTokens
	}
}

// GetTokenUsage retrieves the token usage from the request context.
func GetTokenUsage(r *http.Request) (int, int) {
	if usage, ok := r.Context().Value(ctxKey{}).(*[2]int); ok {
		return usage[0], usage[1]
	}
	return 0, 0
}

func InitRequestOutcome(r *http.Request) *http.Request {
	outcome := new(RequestOutcome)
	return r.WithContext(context.WithValue(r.Context(), requestOutcomeCtxKey{}, outcome))
}

// SetRequestError preserves the first internal failure, which is usually the
// most specific root cause. Later cleanup errors must not overwrite it.
func SetRequestError(r *http.Request, statusCode int, message string) {
	if outcome, ok := r.Context().Value(requestOutcomeCtxKey{}).(*RequestOutcome); ok && outcome.Error == "" {
		outcome.StatusCode = statusCode
		outcome.Error = message
	}
}

func GetRequestOutcome(r *http.Request) RequestOutcome {
	if outcome, ok := r.Context().Value(requestOutcomeCtxKey{}).(*RequestOutcome); ok {
		return *outcome
	}
	return RequestOutcome{}
}

// SetModel stores the resolved (translated) model name in the request context.
func SetModel(r *http.Request, model string) {
	if m, ok := r.Context().Value(modelCtxKey{}).(*string); ok {
		*m = model
	}
}

// GetModel retrieves the resolved model name from the request context.
func GetModel(r *http.Request) string {
	if m, ok := r.Context().Value(modelCtxKey{}).(*string); ok {
		return *m
	}
	return ""
}

// InitModel initializes model tracking in the request context.
func InitModel(r *http.Request) *http.Request {
	m := new(string)
	return r.WithContext(context.WithValue(r.Context(), modelCtxKey{}, m))
}

// SetAccountDefaultModel stores the account's default model in the request context.
func SetAccountDefaultModel(r *http.Request, defaultModel string) {
	if m, ok := r.Context().Value(accountModelCtxKey{}).(*string); ok {
		*m = defaultModel
	}
}

// GetAccountDefaultModel retrieves the account's default model from the request context.
func GetAccountDefaultModel(r *http.Request) string {
	if m, ok := r.Context().Value(accountModelCtxKey{}).(*string); ok {
		return *m
	}
	return ""
}

// InitAccountModel initializes account default model tracking in the request context.
func InitAccountModel(r *http.Request) *http.Request {
	m := new(string)
	return r.WithContext(context.WithValue(r.Context(), accountModelCtxKey{}, m))
}
