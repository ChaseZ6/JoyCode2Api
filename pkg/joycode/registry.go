package joycode

import (
	"log/slog"
	"strings"
	"sync"
	"time"
)

// DefaultTTL is how long a fetched model list stays fresh before the registry
// re-queries upstream.
const DefaultTTL = 5 * time.Minute

// Registry caches the upstream model list so request handlers can resolve model
// names against live data instead of a hardcoded list. Upstream adds and removes
// models without any change on our side; a stale hardcoded list silently
// rewrites unknown models to the default, which is indistinguishable from a bug
// for the caller.
type Registry struct {
	mu        sync.RWMutex
	models    []ModelInfo
	fetchedAt time.Time
	ttl       time.Duration
	fetch     func() ([]ModelInfo, error)

	// refreshMu serializes refreshes so a burst of concurrent requests triggers
	// one upstream call rather than one per request.
	refreshMu sync.Mutex
}

// DefaultRegistry is the process-wide registry used by the request handlers.
var DefaultRegistry = NewRegistry(DefaultTTL)

// NewRegistry creates a registry with the given freshness window.
func NewRegistry(ttl time.Duration) *Registry {
	if ttl <= 0 {
		ttl = DefaultTTL
	}
	return &Registry{ttl: ttl}
}

// SetFetcher installs the upstream fetch function. Until this is called (or if
// it fails), the registry serves StaticModels().
func (r *Registry) SetFetcher(f func() ([]ModelInfo, error)) {
	r.mu.Lock()
	r.fetch = f
	r.mu.Unlock()
}

// Seed installs a known model list directly, bypassing the fetcher. Intended for
// tests and for callers that already hold a fresh list.
func (r *Registry) Seed(models []ModelInfo) {
	r.mu.Lock()
	r.models = models
	r.fetchedAt = time.Now()
	r.mu.Unlock()
}

// Models returns the cached model list, refreshing it if stale. It never returns
// empty: on fetch failure it falls back to the last known list, then to
// StaticModels().
func (r *Registry) Models() []ModelInfo {
	r.mu.RLock()
	fresh := time.Since(r.fetchedAt) < r.ttl && len(r.models) > 0
	cached := r.models
	hasFetch := r.fetch != nil
	r.mu.RUnlock()

	if fresh || !hasFetch {
		if len(cached) > 0 {
			return cached
		}
		return StaticModels()
	}

	r.refresh()

	r.mu.RLock()
	cached = r.models
	r.mu.RUnlock()
	if len(cached) > 0 {
		return cached
	}
	return StaticModels()
}

// refresh performs one upstream fetch, collapsing concurrent callers.
func (r *Registry) refresh() {
	r.refreshMu.Lock()
	defer r.refreshMu.Unlock()

	// Another goroutine may have refreshed while we waited for the lock.
	r.mu.RLock()
	if time.Since(r.fetchedAt) < r.ttl && len(r.models) > 0 {
		r.mu.RUnlock()
		return
	}
	fetch := r.fetch
	r.mu.RUnlock()

	if fetch == nil {
		return
	}
	models, err := fetch()
	if err != nil {
		slog.Warn("model registry refresh failed; serving cached/static list", "error", err)
		// Back off so a persistently failing upstream is not hammered on every
		// request: treat the failure as a refresh for TTL purposes only if we
		// have something to serve.
		r.mu.Lock()
		if len(r.models) > 0 {
			r.fetchedAt = time.Now()
		}
		r.mu.Unlock()
		return
	}
	if len(models) == 0 {
		slog.Warn("model registry refresh returned no models; keeping previous list")
		return
	}
	r.mu.Lock()
	r.models = models
	r.fetchedAt = time.Now()
	r.mu.Unlock()
	slog.Info("model registry refreshed", "count", len(models))
}

// Visible returns the models that should be offered to clients, dropping the
// ones upstream marks hidden (internal completion models such as
// JoyCode-Base-V3).
func (r *Registry) Visible() []ModelInfo {
	all := r.Models()
	out := make([]ModelInfo, 0, len(all))
	for _, m := range all {
		if m.IsHiddenModel() {
			continue
		}
		out = append(out, m)
	}
	return out
}

// normalizeModelName makes model matching tolerant of the spelling differences
// between UIs and CLIs. Upstream labels contain spaces ("GPT-5.6 Sol") which
// many clients and shells turn into dashes or underscores.
func normalizeModelName(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.ReplaceAll(s, " ", "-")
	s = strings.ReplaceAll(s, "_", "-")
	for strings.Contains(s, "--") {
		s = strings.ReplaceAll(s, "--", "-")
	}
	return normalizeVersionSeparator(s)
}

// normalizeVersionSeparator rewrites a dash between two digits as a dot, so the
// dotted upstream labels ("Claude-Opus-4.8") match the all-dash IDs that
// Anthropic clients send ("claude-opus-4-8"). Without this, Claude Code and
// Claude Desktop requests for a Claude model fall through to the default model.
//
// Only digit-dash-digit is rewritten, leaving family names intact:
// "glm-5-jcloud" and "claude-3-5-sonnet-20241022" keep their remaining dashes.
func normalizeVersionSeparator(s string) string {
	if len(s) < 3 {
		return s
	}
	b := []byte(s)
	for i := 1; i < len(b)-1; i++ {
		if b[i] == '-' && isASCIIDigit(b[i-1]) && isASCIIDigit(b[i+1]) {
			b[i] = '.'
		}
	}
	return string(b)
}

func isASCIIDigit(c byte) bool { return c >= '0' && c <= '9' }

// Resolve looks up a model by its display label or its upstream chatApiModel,
// case-insensitively and tolerant of space/dash differences.
func (r *Registry) Resolve(name string) (ModelInfo, bool) {
	if strings.TrimSpace(name) == "" {
		return ModelInfo{}, false
	}
	want := normalizeModelName(name)
	models := r.Models()

	// Exact label or chatApiModel match wins.
	for _, m := range models {
		if normalizeModelName(m.Label) == want || normalizeModelName(m.ChatAPIModel) == want {
			return m, true
		}
	}

	// Upstream revises models in place by appending a version suffix
	// ("JoyAI-Code" became "JoyAI-Code-1.5"), which strands existing configs.
	// Accept the base name when exactly one model extends it at a version
	// boundary; ambiguity means we cannot guess safely, so we do not.
	var match ModelInfo
	found := 0
	for _, m := range models {
		if versionExtends(normalizeModelName(m.Label), want) ||
			versionExtends(normalizeModelName(m.ChatAPIModel), want) {
			if found == 0 || m.Label != match.Label {
				match = m
				found++
			}
		}
	}
	if found == 1 {
		slog.Info("resolved model by version-suffix match",
			"requested", name, "matched", match.DisplayName())
		return match, true
	}
	return ModelInfo{}, false
}

// versionExtends reports whether candidate is base plus a version suffix, e.g.
// "joyai-code-1.5" extends "joyai-code". It requires a separator at the boundary
// so "glm-52" does not match "glm-5".
func versionExtends(candidate, base string) bool {
	if candidate == "" || base == "" || !strings.HasPrefix(candidate, base) {
		return false
	}
	rest := candidate[len(base):]
	if rest == "" {
		return false
	}
	if rest[0] != '-' && rest[0] != '.' {
		return false
	}
	// The remainder must look like a version, not a different model family.
	for _, c := range rest[1:] {
		if (c < '0' || c > '9') && c != '.' && c != '-' {
			return false
		}
	}
	return len(rest) > 1
}

// ResolveRequest picks the model for a request: the client's choice if upstream
// knows it, else the account default, else the system default, else the
// built-in default.
//
// A requested model that upstream does not know is logged rather than swapped
// silently: silent substitution is indistinguishable from the request being
// ignored, which is how unavailable models previously went unnoticed.
func (r *Registry) ResolveRequest(requested, accountDefault, systemDefault string) ModelInfo {
	if m, ok := r.Resolve(requested); ok {
		return m
	}
	if strings.TrimSpace(requested) != "" {
		slog.Warn("requested model not offered by upstream; falling back to default",
			"requested", requested, "account_default", accountDefault,
			"system_default", systemDefault)
	}
	for _, candidate := range []string{accountDefault, systemDefault, DefaultModel} {
		if m, ok := r.Resolve(candidate); ok {
			return m
		}
	}
	// Upstream is unreachable and the default is not in the static list; send
	// the default through unchanged rather than failing the request.
	return ModelInfo{Label: DefaultModel, ChatAPIModel: DefaultModel}
}

// StaticModels is the offline fallback, used only when upstream is unreachable
// and nothing has been cached yet. It is a snapshot and will drift; the registry
// prefers live data whenever it can get it.
func StaticModels() []ModelInfo {
	return []ModelInfo{
		{Label: "JoyAI-Code-1.5", ChatAPIModel: "JoyAI-Code-1.5", MaxTotalTokens: 200000, RespMaxTokens: 64000},
		{Label: "MiniMax-M3", ChatAPIModel: "MiniMax-M3", MaxTotalTokens: 200000, RespMaxTokens: 64000},
		{Label: "MiniMax-M2.7", ChatAPIModel: "MiniMax-M2.7", MaxTotalTokens: 200000, RespMaxTokens: 64000},
		{Label: "Kimi-K2.6", ChatAPIModel: "Kimi-K2.6", MaxTotalTokens: 200000, RespMaxTokens: 64000},
		{Label: "Kimi-K3", ChatAPIModel: "Kimi-K3", MaxTotalTokens: 200000, RespMaxTokens: 64000},
		{Label: "GLM-5.2", ChatAPIModel: "GLM-5.2", MaxTotalTokens: 200000, RespMaxTokens: 64000},
		{Label: "GLM-5.2-jcloud", ChatAPIModel: "GLM-5.2-jcloud", MaxTotalTokens: 200000, RespMaxTokens: 64000},
		{Label: "GLM-5.1", ChatAPIModel: "GLM-5.1", MaxTotalTokens: 200000, RespMaxTokens: 64000},
		{Label: "GLM-5", ChatAPIModel: "GLM-5", MaxTotalTokens: 200000, RespMaxTokens: 64000},
		{Label: "GLM-5-jcloud", ChatAPIModel: "GLM-5-jcloud", MaxTotalTokens: 200000, RespMaxTokens: 64000},
		{Label: "DeepSeek-V4-Pro", ChatAPIModel: "DeepSeek-V4-Pro", MaxTotalTokens: 200000, RespMaxTokens: 64000},
		{Label: "Doubao-Seed-2.0-pro", ChatAPIModel: "Doubao-Seed-2.0-pro", MaxTotalTokens: 200000, RespMaxTokens: 64000},
		// Keep token limits unset here until upstream reports them. ModelInfo's
		// conservative fallbacks avoid inventing Astra-specific limits while the
		// live registry remains the source of truth in normal operation.
		{Label: "GPT-6 Astra", ChatAPIModel: "gpt-6-astra",
			ExtJSON: map[string]interface{}{"adapterType": AdapterOpenAIResponse}},
		{Label: "GPT-5.6 Sol", ChatAPIModel: "GPT-5.6 Sol", MaxTotalTokens: 1_000_000, RespMaxTokens: 64000,
			ExtJSON: map[string]interface{}{"adapterType": AdapterOpenAIResponse}},
		{Label: "Claude-Opus-4.8", ChatAPIModel: "Claude-Opus-4.8-hq", MaxTotalTokens: 200000, RespMaxTokens: 64000,
			ExtJSON: map[string]interface{}{"adapterType": AdapterAnthropic}},
		{Label: "Claude-Opus-4.7", ChatAPIModel: "Claude-Opus-4.7-hq", MaxTotalTokens: 200000, RespMaxTokens: 64000,
			ExtJSON: map[string]interface{}{"adapterType": AdapterAnthropic}},
		{Label: "Claude-Sonnet-4.6", ChatAPIModel: "Claude-Sonnet-4.6-hq", MaxTotalTokens: 200000, RespMaxTokens: 64000,
			ExtJSON: map[string]interface{}{"adapterType": AdapterAnthropic}},
		{Label: "Claude-Opus-4.6", ChatAPIModel: "Claude-Opus-4.6-hq", MaxTotalTokens: 200000, RespMaxTokens: 64000,
			ExtJSON: map[string]interface{}{"adapterType": AdapterAnthropic}},
	}
}
