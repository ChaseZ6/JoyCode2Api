package openai

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

// ClientResolver returns the appropriate joycode.Client for a request.
type ClientResolver func(r *http.Request) *joycode.Client

// Server implements the OpenAI-compatible HTTP API.
type Server struct {
	Client   *joycode.Client
	Resolver ClientResolver
	store    *store.Store
}

// NewServer creates a new OpenAI-compatible proxy server.
func NewServer(c *joycode.Client, s *store.Store) *Server {
	return &Server{Client: c, store: s}
}

func (s *Server) getClient(r *http.Request) *joycode.Client {
	if s.Resolver != nil {
		return s.Resolver(r)
	}
	return s.Client
}

// RegisterRoutes registers all OpenAI-compatible endpoints on the mux.
func (s *Server) RegisterRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/v1/chat/completions", s.handleChat)
	mux.HandleFunc("/v1/responses", s.handleNativeResponses)
	mux.HandleFunc("/responses", s.handleNativeResponses)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/v1/web-search", s.handleWebSearch)
	mux.HandleFunc("/v1/rerank", s.handleRerank)
	mux.HandleFunc("/v1/", handleUnknownV1)
	mux.HandleFunc("/health", s.handleHealth)
}

const corsAllowedHeaders = "Authorization, Content-Type, Accept, OpenAI-Organization, OpenAI-Project, X-API-Key, X-Request-ID"

func writeCORS(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", corsAllowedHeaders)
}

func writeJSON(w http.ResponseWriter, code int, v interface{}) {
	b, err := json.Marshal(v)
	if err != nil {
		slog.Error("writeJSON: marshal failed", "error", err)
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.WriteHeader(code)
	w.Write(b)
}

func writeError(w http.ResponseWriter, code int, msg string) {
	writeOpenAIError(w, code, msg, "api_error", "", "")
}

func writeOpenAIError(w http.ResponseWriter, status int, message, errorType, code, param string) {
	errBody := map[string]interface{}{"message": message, "type": errorType}
	if code != "" {
		errBody["code"] = code
	}
	if param != "" {
		errBody["param"] = param
	}
	writeJSON(w, status, map[string]interface{}{"error": errBody})
}

func upstreamHTTPStatus(err error) int {
	if err == nil {
		return http.StatusBadGateway
	}
	if isTimeoutError(err.Error()) {
		return http.StatusGatewayTimeout
	}
	var upstream *joycode.UpstreamError
	if errors.As(err, &upstream) {
		combined := strings.ToLower(upstream.Code + " " + upstream.Message + " " + upstream.Cause)
		if strings.Contains(combined, "rate_limit") || strings.Contains(combined, "rate limit") || upstream.Code == "429" {
			return http.StatusTooManyRequests
		}
		if strings.Contains(combined, "context") || upstream.Code == "400" {
			return http.StatusBadRequest
		}
	}
	return http.StatusBadGateway
}

func requirePOST(w http.ResponseWriter, r *http.Request) bool {
	if r.Method == http.MethodOptions {
		writeCORS(w)
		w.WriteHeader(200)
		return false
	}
	if r.Method != http.MethodPost {
		writeError(w, 405, "method not allowed")
		return false
	}
	return true
}

func requireGET(w http.ResponseWriter, r *http.Request) bool {
	if r.Method == http.MethodOptions {
		writeCORS(w)
		w.WriteHeader(200)
		return false
	}
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "method not allowed")
		return false
	}
	return true
}

func handleUnknownV1(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodOptions {
		writeCORS(w)
		w.WriteHeader(http.StatusOK)
		return
	}
	writeOpenAIError(w, http.StatusNotFound,
		fmt.Sprintf("Unknown API endpoint: %s", r.URL.Path),
		"invalid_request_error", "endpoint_not_found", "")
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodOptions {
		writeCORS(w)
		w.WriteHeader(200)
		return
	}
	writeJSON(w, 200, map[string]interface{}{
		"status": "ok", "service": "joycode-openai-proxy",
		"endpoints": []string{
			"/v1/chat/completions", "/v1/models",
			"/v1/web-search", "/v1/rerank",
		},
	})
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if !requireGET(w, r) {
		return
	}
	// Prefer this request's own account, so the list reflects what that account
	// can actually use. Fall back to the registry (cached live data, or the
	// static snapshot) so a brief upstream outage does not empty the list.
	models, err := s.getClient(r).ListModelsContext(r.Context())
	if err != nil {
		slog.Warn("list models upstream failed; serving registry list", "error", err)
		models = joycode.DefaultRegistry.Visible()
	}
	writeJSON(w, 200, TranslateModels(models))
}
