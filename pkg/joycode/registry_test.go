package joycode

import (
	"errors"
	"sync"
	"testing"
	"time"
)

func testModels() []ModelInfo {
	return []ModelInfo{
		{Label: "JoyAI-Code-1.5", ChatAPIModel: "JoyAI-Code-1.5"},
		{Label: "GLM-5.2", ChatAPIModel: "GLM-5.2"},
		{Label: "GLM-5", ChatAPIModel: "GLM-5"},
		{Label: "GPT-6 Astra", ChatAPIModel: "gpt-6-astra",
			ExtJSON: map[string]interface{}{"adapterType": AdapterOpenAIResponse}},
		{Label: "GPT-5.6 Sol", ChatAPIModel: "GPT-5.6 Sol",
			ExtJSON: map[string]interface{}{"adapterType": AdapterOpenAIResponse}},
		{Label: "Claude-Opus-4.7", ChatAPIModel: "Claude-Opus-4.7-hq", SortOrder: 50,
			ExtJSON: map[string]interface{}{"adapterType": AdapterAnthropic}},
		{Label: "JoyCode-Base-V3", ChatAPIModel: "JoyCode-Base-V3", Hidden: true},
	}
}

func TestRegistry_ResolveByLabelAndAPIName(t *testing.T) {
	r := NewRegistry(time.Minute)
	r.Seed(testModels())

	// The display label resolves to the upstream chatApiModel, which is what
	// must be sent: upstream accepts the label but silently serves it with a
	// different model.
	m, ok := r.Resolve("Claude-Opus-4.7")
	if !ok {
		t.Fatal("expected Claude-Opus-4.7 to resolve")
	}
	if m.APIName() != "Claude-Opus-4.7-hq" {
		t.Errorf("APIName() = %q, want Claude-Opus-4.7-hq", m.APIName())
	}

	// The chatApiModel itself must also resolve.
	if _, ok := r.Resolve("Claude-Opus-4.7-hq"); !ok {
		t.Error("expected chatApiModel to resolve")
	}
}

func TestStaticModels_ContainsKimiK3(t *testing.T) {
	for _, model := range StaticModels() {
		if model.Label == "Kimi-K3" && model.APIName() == "Kimi-K3" {
			return
		}
	}
	t.Fatal("static model fallback must contain Kimi-K3")
}

func TestRegistry_ResolveNormalisesSpacesAndCase(t *testing.T) {
	r := NewRegistry(time.Minute)
	r.Seed(testModels())

	// Upstream labels can contain spaces; clients and shells commonly turn
	// those into dashes or underscores.
	for _, name := range []string{"GPT-5.6 Sol", "gpt-5.6-sol", "GPT-5.6_Sol", "gpt-5.6 sol"} {
		m, ok := r.Resolve(name)
		if !ok {
			t.Errorf("expected %q to resolve", name)
			continue
		}
		if m.APIName() != "GPT-5.6 Sol" {
			t.Errorf("Resolve(%q).APIName() = %q, want %q", name, m.APIName(), "GPT-5.6 Sol")
		}
	}
}

func TestRegistry_ResolveGPT6AstraOfficialID(t *testing.T) {
	r := NewRegistry(time.Minute)
	r.Seed(testModels())

	for _, name := range []string{"GPT-6 Astra", "gpt-6-astra", "GPT-6_Astra"} {
		m, ok := r.Resolve(name)
		if !ok {
			t.Errorf("expected %q to resolve", name)
			continue
		}
		if m.APIName() != "gpt-6-astra" {
			t.Errorf("Resolve(%q).APIName() = %q, want %q", name, m.APIName(), "gpt-6-astra")
		}
		if m.Adapter() != AdapterOpenAIResponse {
			t.Errorf("Resolve(%q).Adapter() = %q, want %q", name, m.Adapter(), AdapterOpenAIResponse)
		}
	}
}

func TestRegistry_ResolveVersionSuffix(t *testing.T) {
	r := NewRegistry(time.Minute)
	r.Seed(testModels())

	// Upstream renamed "JoyAI-Code" to "JoyAI-Code-1.5"; existing configs must
	// keep working.
	m, ok := r.Resolve("JoyAI-Code")
	if !ok {
		t.Fatal("expected JoyAI-Code to resolve to its versioned successor")
	}
	if m.Label != "JoyAI-Code-1.5" {
		t.Errorf("resolved to %q, want JoyAI-Code-1.5", m.Label)
	}
}

func TestRegistry_ResolveVersionSuffixIsUnambiguous(t *testing.T) {
	r := NewRegistry(time.Minute)
	r.Seed([]ModelInfo{
		{Label: "GLM-5.1", ChatAPIModel: "GLM-5.1"},
		{Label: "GLM-5.2", ChatAPIModel: "GLM-5.2"},
	})
	// Two candidates extend "GLM-5", so guessing would be wrong; the exact
	// model "GLM-5" is genuinely gone and must not silently become 5.1 or 5.2.
	if m, ok := r.Resolve("GLM-5"); ok {
		t.Errorf("expected ambiguous prefix not to resolve, got %q", m.Label)
	}
	// An exact match must still win over any suffix candidates.
	r.Seed([]ModelInfo{
		{Label: "GLM-5", ChatAPIModel: "GLM-5"},
		{Label: "GLM-5.2", ChatAPIModel: "GLM-5.2"},
	})
	m, ok := r.Resolve("GLM-5")
	if !ok || m.Label != "GLM-5" {
		t.Errorf("exact match should win, got %q ok=%v", m.Label, ok)
	}
}

func TestRegistry_ResolveRejectsUnknown(t *testing.T) {
	r := NewRegistry(time.Minute)
	r.Seed(testModels())
	for _, name := range []string{"", "   ", "no-such-model", "gpt-4o"} {
		if _, ok := r.Resolve(name); ok {
			t.Errorf("expected %q not to resolve", name)
		}
	}
}

func TestRegistry_ResolveRequestFallbackChain(t *testing.T) {
	r := NewRegistry(time.Minute)
	r.Seed(testModels())

	if got := r.ResolveRequest("GLM-5.2", "GLM-5", "").APIName(); got != "GLM-5.2" {
		t.Errorf("requested model should win, got %q", got)
	}
	if got := r.ResolveRequest("unknown", "GLM-5", "GLM-5.2").APIName(); got != "GLM-5" {
		t.Errorf("account default should be next, got %q", got)
	}
	if got := r.ResolveRequest("unknown", "", "GLM-5.2").APIName(); got != "GLM-5.2" {
		t.Errorf("system default should be next, got %q", got)
	}
	if got := r.ResolveRequest("unknown", "", "").APIName(); got != DefaultModel {
		t.Errorf("built-in default should be last, got %q", got)
	}
}

func TestRegistry_VisibleOmitsHidden(t *testing.T) {
	r := NewRegistry(time.Minute)
	r.Seed(testModels())
	for _, m := range r.Visible() {
		if m.Label == "JoyCode-Base-V3" {
			t.Fatal("hidden model must not be offered to clients")
		}
	}
	if len(r.Visible()) != len(testModels())-1 {
		t.Errorf("expected exactly one model hidden, got %d of %d",
			len(r.Visible()), len(testModels()))
	}
}

func TestRegistry_FallsBackToStaticWithoutFetcher(t *testing.T) {
	r := NewRegistry(time.Minute)
	if len(r.Models()) == 0 {
		t.Fatal("expected static fallback rather than an empty list")
	}
	// Responses-native GPT models must be reachable offline too, with the
	// right adapter and upstream identifier.
	for _, tc := range []struct {
		requested string
		apiName   string
	}{
		{requested: "GPT-5.6 Sol", apiName: "GPT-5.6 Sol"},
		{requested: "gpt-6-astra", apiName: "gpt-6-astra"},
	} {
		m, ok := r.Resolve(tc.requested)
		if !ok {
			t.Errorf("expected %s in the static fallback list", tc.requested)
			continue
		}
		if m.APIName() != tc.apiName {
			t.Errorf("%s APIName() = %q, want %q", tc.requested, m.APIName(), tc.apiName)
		}
		if m.Adapter() != AdapterOpenAIResponse {
			t.Errorf("%s adapter = %q, want %q", tc.requested, m.Adapter(), AdapterOpenAIResponse)
		}
	}
}

func TestRegistry_FetchFailureKeepsServingCache(t *testing.T) {
	r := NewRegistry(time.Millisecond)
	r.Seed(testModels())
	r.SetFetcher(func() ([]ModelInfo, error) {
		return nil, errors.New("upstream down")
	})
	time.Sleep(5 * time.Millisecond) // let the cache go stale

	models := r.Models()
	if len(models) == 0 {
		t.Fatal("expected cached list to survive a fetch failure")
	}
	if _, ok := r.Resolve("GPT-5.6 Sol"); !ok {
		t.Error("expected cached models to remain resolvable")
	}
}

func TestRegistry_RefreshCollapsesConcurrentCalls(t *testing.T) {
	r := NewRegistry(time.Millisecond)
	var mu sync.Mutex
	calls := 0
	r.SetFetcher(func() ([]ModelInfo, error) {
		mu.Lock()
		calls++
		mu.Unlock()
		time.Sleep(10 * time.Millisecond)
		return testModels(), nil
	})

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			r.Models()
		}()
	}
	wg.Wait()

	mu.Lock()
	got := calls
	mu.Unlock()
	// A burst of concurrent requests must not fan out into one upstream call
	// each.
	if got > 2 {
		t.Errorf("expected refresh to collapse concurrent calls, got %d fetches", got)
	}
}

func TestModelInfo_Adapter(t *testing.T) {
	// No extJson at all means the default chat/completions endpoint.
	if got := (ModelInfo{Label: "GLM-5"}).Adapter(); got != AdapterChatCompletions {
		t.Errorf("adapter = %q, want empty", got)
	}
	m := ModelInfo{ExtJSON: map[string]interface{}{"adapterType": AdapterAnthropic}}
	if m.Adapter() != AdapterAnthropic {
		t.Errorf("adapter = %q, want %q", m.Adapter(), AdapterAnthropic)
	}
	// A non-string adapterType must not panic.
	weird := ModelInfo{ExtJSON: map[string]interface{}{"adapterType": 42}}
	if weird.Adapter() != AdapterChatCompletions {
		t.Errorf("unexpected adapter for non-string value: %q", weird.Adapter())
	}
}

func TestModelInfo_APIName(t *testing.T) {
	m := ModelInfo{Label: "Claude-Opus-4.7", ChatAPIModel: "Claude-Opus-4.7-hq"}
	if m.APIName() != "Claude-Opus-4.7-hq" {
		t.Errorf("APIName() = %q, want the chatApiModel", m.APIName())
	}
	if m.DisplayName() != "Claude-Opus-4.7" {
		t.Errorf("DisplayName() = %q, want the label", m.DisplayName())
	}
	// Falling back when one side is missing.
	if got := (ModelInfo{Label: "OnlyLabel"}).APIName(); got != "OnlyLabel" {
		t.Errorf("APIName() = %q, want OnlyLabel", got)
	}
	if got := (ModelInfo{ChatAPIModel: "OnlyAPI"}).DisplayName(); got != "OnlyAPI" {
		t.Errorf("DisplayName() = %q, want OnlyAPI", got)
	}
}

// Anthropic clients (Claude Code, Claude Desktop) send all-dash model IDs
// ("claude-opus-4-7") while upstream labels use a dot ("Claude-Opus-4.7").
// Without separator normalisation these fell through to the default model, so a
// request for Claude was silently answered by a different family.
func TestRegistry_ResolveDashedVersionID(t *testing.T) {
	r := NewRegistry(time.Minute)
	r.Seed(testModels())

	for _, name := range []string{"claude-opus-4-7", "Claude-Opus-4-7", "claude-opus-4-7-hq"} {
		m, ok := r.Resolve(name)
		if !ok {
			t.Errorf("expected %q to resolve", name)
			continue
		}
		if m.APIName() != "Claude-Opus-4.7-hq" {
			t.Errorf("Resolve(%q).APIName() = %q, want Claude-Opus-4.7-hq", name, m.APIName())
		}
	}

	// The same applies to the space-containing label.
	if m, ok := r.Resolve("gpt-5-6-sol"); !ok || m.APIName() != "GPT-5.6 Sol" {
		t.Errorf("Resolve(gpt-5-6-sol) = %q, %v; want GPT-5.6 Sol, true", m.APIName(), ok)
	}
}

// Normalisation must not merge distinct model families: only a dash between two
// digits is a version separator.
func TestRegistry_ResolveDoesNotMergeFamilies(t *testing.T) {
	r := NewRegistry(time.Minute)
	r.Seed(testModels())

	// A legacy Anthropic ID with no upstream counterpart must not resolve; the
	// caller needs the fallback warning rather than a wrong model.
	if m, ok := r.Resolve("claude-3-5-sonnet-20241022"); ok {
		t.Errorf("claude-3-5-sonnet-20241022 must not resolve, got %q", m.APIName())
	}

	// GLM-5 and GLM-5.2 both exist; a dashed request must hit the exact one.
	if m, ok := r.Resolve("glm-5-2"); !ok || m.APIName() != "GLM-5.2" {
		t.Errorf("Resolve(glm-5-2) = %q, %v; want GLM-5.2, true", m.APIName(), ok)
	}
	if m, ok := r.Resolve("glm-5"); !ok || m.APIName() != "GLM-5" {
		t.Errorf("Resolve(glm-5) = %q, %v; want GLM-5, true", m.APIName(), ok)
	}
}

func TestNormalizeVersionSeparator(t *testing.T) {
	cases := map[string]string{
		"claude-opus-4-7":            "claude-opus-4.7",
		"claude-opus-4.7":            "claude-opus-4.7",
		"gpt-5-6-sol":                "gpt-5.6-sol",
		"glm-5-jcloud":               "glm-5-jcloud", // dash before a letter stays
		"claude-3-5-sonnet-20241022": "claude-3.5-sonnet-20241022",
		"joyai-code-1-5":             "joyai-code-1.5",
		"":                           "",
		"a":                          "a",
		"4-8":                        "4.8",
	}
	for in, want := range cases {
		if got := normalizeVersionSeparator(in); got != want {
			t.Errorf("normalizeVersionSeparator(%q) = %q, want %q", in, got, want)
		}
	}
}
