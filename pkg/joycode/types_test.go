package joycode

import "testing"

func TestEffectiveMaxTokens(t *testing.T) {
	m64 := ModelInfo{Label: "GPT-5.6 Sol", RespMaxTokens: 64000}
	small := ModelInfo{Label: "JoyCode-Base-V3", RespMaxTokens: 4096}
	noLimit := ModelInfo{Label: "Unknown"} // falls back to 8192

	cases := []struct {
		name      string
		m         ModelInfo
		requested int
		floor     int
		want      int
	}{
		{"absent goes to model max", m64, 0, 0, 64000},
		{"low client value is raised", m64, 8192, 0, 64000},
		{"the old 32768 clamp is gone", m64, 32768, 0, 64000},
		{"model max passes through", m64, 64000, 0, 64000},
		{"over model max is clamped", m64, 200000, 0, 64000},
		{"configured floor is respected", m64, 1000, 16000, 16000},
		{"client above floor wins", m64, 40000, 16000, 40000},
		{"floor above model max is capped", m64, 0, 999999, 64000},
		{"small model keeps its own cap", small, 64000, 0, 4096},
		{"small model absent", small, 0, 0, 4096},
		{"missing RespMaxTokens falls back", noLimit, 0, 0, 8192},
		{"negative requested treated as absent", m64, -5, 0, 64000},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.m.EffectiveMaxTokens(tc.requested, tc.floor); got != tc.want {
				t.Errorf("EffectiveMaxTokens(%d, %d) = %d, want %d",
					tc.requested, tc.floor, got, tc.want)
			}
		})
	}
}

func TestContextWindowTokens(t *testing.T) {
	cases := []struct {
		name string
		m    ModelInfo
		want int
	}{
		{
			name: "GPT live metadata is overridden",
			m: ModelInfo{
				Label: "GPT-5.6 Sol", ChatAPIModel: "GPT-5.6 Sol",
				MaxTotalTokens: 200_000,
			},
			want: 1_000_000,
		},
		{
			name: "GPT dash alias is overridden",
			m: ModelInfo{
				Label: "gpt-5-6-sol", MaxTotalTokens: 200_000,
			},
			want: 1_000_000,
		},
		{
			name: "Kimi-K3 limited by gateway 5MB cap, not model",
			m: ModelInfo{
				Label: "Kimi-K3", ChatAPIModel: "Kimi-K3",
				MaxTotalTokens: 200_000,
			},
			want: 600_000,
		},
		{
			name: "other model keeps upstream value",
			m: ModelInfo{
				Label: "Claude-Opus-4.8", MaxTotalTokens: 200_000,
			},
			want: 200_000,
		},
		{
			name: "missing metadata uses conservative fallback",
			m:    ModelInfo{Label: "Unknown"},
			want: 200_000,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.m.ContextWindowTokens(); got != tc.want {
				t.Errorf("ContextWindowTokens() = %d, want %d", got, tc.want)
			}
		})
	}
}
