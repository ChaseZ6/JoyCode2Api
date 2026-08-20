package common

import (
	"net/http/httptest"
	"testing"
)

func TestApplyLocalCORS(t *testing.T) {
	for _, tc := range []struct{ origin, want string }{
		{"http://localhost:3000", "http://localhost:3000"},
		{"https://127.0.0.1:34891", "https://127.0.0.1:34891"},
		{"http://[::1]:3000", "http://[::1]:3000"},
		{"https://evil.example", ""},
	} {
		req := httptest.NewRequest("OPTIONS", "/", nil)
		req.Header.Set("Origin", tc.origin)
		rec := httptest.NewRecorder()
		ApplyLocalCORS(rec, req, "GET", "Content-Type")
		if got := rec.Header().Get("Access-Control-Allow-Origin"); got != tc.want {
			t.Errorf("origin %q => %q, want %q", tc.origin, got, tc.want)
		}
	}
}
