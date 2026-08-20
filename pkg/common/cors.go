package common

import (
	"net/http"
	"net/url"
	"strings"
)

// ApplyLocalCORS allows browser clients served from localhost while rejecting
// arbitrary websites. Non-browser clients usually send no Origin.
func ApplyLocalCORS(w http.ResponseWriter, r *http.Request, methods, headers string) {
	origin := r.Header.Get("Origin")
	if origin != "" && isLocalOrigin(origin) {
		w.Header().Set("Access-Control-Allow-Origin", origin)
		w.Header().Set("Vary", "Origin")
	}
	w.Header().Set("Access-Control-Allow-Methods", methods)
	w.Header().Set("Access-Control-Allow-Headers", headers)
}

func isLocalOrigin(origin string) bool {
	u, err := url.Parse(origin)
	if err != nil {
		return false
	}
	host := strings.ToLower(u.Hostname())
	return host == "localhost" || host == "127.0.0.1" || host == "::1"
}
