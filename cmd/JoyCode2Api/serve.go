package main

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"log"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/spf13/cobra"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/anthropic"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/auth"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/dashboard"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/keepalive"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/logrot"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/openai"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/proxy"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

var (
	serveHost      string
	servePort      int
	serveTLS       bool
	requestCounter uint64
)

var serveCmd = &cobra.Command{
	Use:     "serve",
	Short:   "启动代理服务器",
	Long:    "启动 OpenAI/Anthropic 兼容的 API 代理服务器，将请求转换为 JoyCode API 格式。",
	GroupID: "core",
	Example: `  # 默认启动（127.0.0.1:34891）
  joycode-proxy serve

  # 指定端口
  joycode-proxy serve -p 8080

  # 启用调试日志
  joycode-proxy -v serve

  # 跳过凭据验证（用于测试）
  joycode-proxy serve --skip-validation`,
	RunE: func(cmd *cobra.Command, args []string) error {
		if os.Getenv("_JOYCODE_DAEMON_CHILD") == "1" {
			runAsDaemonChild()
		}

		setupLogRotation()

		client, err := resolveClient()
		if err != nil {
			return err
		}

		// Open store for dashboard
		s, err := store.Open("")
		if err != nil {
			log.Printf("Warning: dashboard store unavailable: %v", err)
		}

		// Migrate historical request_logs: map old api_keys to first account
		if s != nil {
			accounts, _ := s.ListAccounts()
			if len(accounts) > 0 {
				if n, err := s.ReassignLogs([]string{"", "joycode", "default"}, accounts[0].UserID); err == nil && n > 0 {
					log.Printf("Migrated %d request logs to account %q", n, accounts[0].UserID)
				}
			}
		}
		if s != nil {
			if n, err := s.MigrateTokenLogs(); err == nil && n > 0 {
				log.Printf("Migrated %d token-based request logs to account api_keys", n)
			}
		}

		// Feed the model registry from upstream so model resolution, the
		// /v1/models endpoint and the dashboard all reflect live availability
		// instead of a hardcoded list.
		joycode.DefaultRegistry.SetFetcher(client.ListModels)

		var requestLogWriter *store.RequestLogWriter
		if s != nil {
			requestLogWriter = store.NewRequestLogWriter(s, 4096)
		}

		srv := openai.NewServer(client, s)
		anth := anthropic.NewHandler(client, s)

		// Start credential keepalive: check every 10min, refresh accounts older than 1h
		keeper := keepalive.NewKeeper(s, 1*time.Hour)
		keeper.Start(10 * time.Minute)

		// stopCh is closed on shutdown so background goroutines can exit.
		stopCh := make(chan struct{})

		// Per-request client resolution from database accounts
		if s != nil {
			transportManager := joycode.NewTransportManager(s.GetIntSetting("max_connections", 8))
			client.SetTransport(transportManager)

			// Background goroutine atomically swaps immutable transports when the
			// connection limit changes.
			go func() {
				ticker := time.NewTicker(10 * time.Second)
				defer ticker.Stop()
				for {
					select {
					case <-ticker.C:
						transportManager.Update(s.GetIntSetting("max_connections", 8))
					case <-stopCh:
						transportManager.CloseIdleConnections()
						return
					}
				}
			}()

			resolver := func(r *http.Request) *joycode.Client {
				systemClient := client
				if systemClient != nil && systemClient.PtKey == "placeholder" {
					if creds, err := auth.LoadFromSystem(); err == nil {
						systemClient = joycode.NewClient(creds.PtKey, creds.UserID)
						systemClient.SetTransport(transportManager)
						systemClient.SetColorContext(creds.ColorBaseURL, creds.MasterBaseURL, creds.Tenant, creds.LoginType, creds.OrgFullName)
					}
				}
				apiKey := extractAPIKey(r)
				timeout := s.GetIntSetting("request_timeout", 1800)
				if timeout < 60 {
					timeout = 60
				}

				// newAccountClient builds a client for a DB account. The gateway
				// routing context (tenant, loginType, color base URL) must be
				// carried over from the local login: without it the client falls
				// back to tenant "JOYCODE" / loginType "N_PIN_PC", which the
				// upstream gateway rejects with "账号未登录" for JD-cloud logins.
				newAccountClient := func(ptKey, userID string) *joycode.Client {
					cl := joycode.NewClient(ptKey, userID)
					cl.SetTransport(transportManager)
					if creds, err := auth.LoadFromSystem(); err == nil {
						cl.SetColorContext(creds.ColorBaseURL, creds.MasterBaseURL, creds.Tenant, creds.LoginType, creds.OrgFullName)
					}
					if systemClient != nil && systemClient.PtKey != "" && systemClient.PtKey != "placeholder" && systemClient.UserID == userID {
						cl.SetAnthropicPtKey(systemClient.PtKey)
					}
					cl.SetTimeout(time.Duration(timeout) * time.Second)
					return cl
				}

				if apiKey != "" {
					if account, _ := s.GetAccountByToken(apiKey); account != nil {
						return newAccountClient(account.PtKey, account.UserID)
					}
					if account, _ := s.GetAccount(apiKey); account != nil {
						return newAccountClient(account.PtKey, account.UserID)
					}
				}
				if account, _ := s.GetDefaultAccount(); account != nil {
					return newAccountClient(account.PtKey, account.UserID)
				}
				return systemClient
			}
			srv.Resolver = resolver
			anth.Resolver = resolver
		}

		// Background log cleanup goroutine (nil-guarded; s may be nil if store.Open failed)
		go func() {
			if s == nil {
				return
			}
			ticker := time.NewTicker(1 * time.Hour)
			defer ticker.Stop()
			cleanup := func() {
				s.CleanupPreviousMonthLogs()
			}
			cleanup()
			for {
				select {
				case <-ticker.C:
					cleanup()
				case <-stopCh:
					return
				}
			}
		}()

		mux := http.NewServeMux()
		srv.RegisterRoutes(mux)
		anth.RegisterRoutes(mux)

		// Register dashboard API routes + static file serving
		if s != nil {
			subFS, _ := fs.Sub(staticFiles, "static")
			dash := dashboard.NewHandler(s, subFS, keeper)
			dash.Version = Version
			dash.RegisterRoutes(mux)
			mux.HandleFunc("/", dash.ServeStatic)
		}

		var handler http.Handler = mux
		if s != nil {
			handler = inferenceAuthMiddleware(handler, s, serveHost)
			handler = auth.JWTMiddleware(s, handler)
			handler = requestLogMiddleware(handler, s, requestLogWriter)
		}
		if verbose {
			handler = loggingMiddleware(handler)
		}

		addr := fmt.Sprintf("%s:%d", serveHost, servePort)
		httpSrv := &http.Server{
			Addr:    addr,
			Handler: handler,
		}

		var tlsCfg *tls.Config
		scheme := "http"
		if serveTLS {
			tlsCfg, err = ensureTLS()
			if err != nil {
				log.Printf("Warning: TLS setup failed (%v), falling back to HTTP", err)
			} else {
				httpSrv.TLSConfig = tlsCfg
				scheme = "https"
			}
		}

		go func() {
			fmt.Println()
			fmt.Printf("  JoyCode Proxy %s\n", Version)
			fmt.Println("  ─────────────────────────────────────────────────")
			fmt.Println()
			fmt.Println("  Endpoints:")
			fmt.Println("    POST /v1/chat/completions  — Chat (OpenAI format)")
			fmt.Println("    POST /v1/messages          — Chat (Anthropic/Claude Code format)")
			fmt.Println("    POST /v1/web-search        — Web Search")
			fmt.Println("    POST /v1/rerank            — Rerank documents")
			fmt.Println("    GET  /v1/models            — Model list")
			fmt.Println("    GET  /health               — Health check")
			fmt.Println()
			fmt.Println("  Dashboard:")
			if tlsCfg != nil {
				fmt.Printf("    https://%s — Web UI (also accepts HTTP)\n", addr)
			} else {
				fmt.Printf("    http://%s — Web UI\n", addr)
			}
			fmt.Println()
			fmt.Println("  Claude Code setup:")
			fmt.Printf("    export ANTHROPIC_BASE_URL=http://%s\n", addr)
			fmt.Println("    export ANTHROPIC_API_KEY=joycode")
			if verbose {
				fmt.Println()
				fmt.Println("  Verbose logging: enabled")
			}
			fmt.Println()

			var listenErr error
			if tlsCfg != nil {
				httpSrv.TLSConfig = tlsCfg
				ln, err := net.Listen("tcp", addr)
				if err != nil {
					log.Fatalf("Listen error: %v", err)
				}
				dualLn := newDualListener(ln, tlsCfg, handler)
				log.Printf("JoyCode Proxy running on %s://%s (also accepts HTTP)", scheme, addr)
				listenErr = httpSrv.Serve(dualLn)
			} else {
				log.Printf("JoyCode Proxy running on %s://%s", scheme, addr)
				listenErr = httpSrv.ListenAndServe()
			}
			if listenErr != nil && listenErr != http.ErrServerClosed {
				log.Fatalf("Server error: %v", listenErr)
			}
		}()

		quit := make(chan os.Signal, 1)
		signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
		<-quit
		log.Println("Shutting down server...")

		// Signal background goroutines to stop.
		close(stopCh)

		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := httpSrv.Shutdown(ctx); err != nil {
			log.Printf("Server shutdown error: %v", err)
		}
		keeper.Stop()
		if requestLogWriter != nil {
			if err := requestLogWriter.Shutdown(ctx); err != nil {
				log.Printf("Request log writer shutdown error: %v", err)
			}
		}
		if s != nil {
			s.Close()
		}
		log.Println("Server stopped")
		return nil
	},
}

func init() {
	serveCmd.Flags().StringVarP(&serveHost, "host", "H", "127.0.0.1", "绑定地址")
	serveCmd.Flags().IntVarP(&servePort, "port", "p", 34891, "绑定端口")
	serveCmd.Flags().BoolVar(&serveTLS, "tls", true, "启用 HTTPS（自签名证书）")
	rootCmd.AddCommand(serveCmd)
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		log.Printf("-> %s %s %s", r.Method, r.URL.Path, r.RemoteAddr)
		next.ServeHTTP(w, r)
		log.Printf("<- %s %s %s", r.Method, r.URL.Path, time.Since(start))
	})
}

func requestLogMiddleware(next http.Handler, s *store.Store, writer *store.RequestLogWriter) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		r = store.InitTokenUsage(r)
		r = store.InitModel(r)
		r = store.InitAccountModel(r)
		r = store.InitRequestOutcome(r)

		// Assign request ID for log correlation
		reqID := atomic.AddUint64(&requestCounter, 1)
		r = anthropic.WithRequestID(r, reqID)

		// Resolve account from API key/token (used for model, session tracking, logging)
		var resolvedAccount *store.Account
		if s != nil {
			ak := extractAPIKey(r)
			if ak != "" {
				if a, _ := s.GetAccountByToken(ak); a != nil {
					resolvedAccount = a
				} else if a, _ := s.GetAccount(ak); a != nil {
					resolvedAccount = a
				}
			}
			if resolvedAccount == nil {
				if a, _ := s.GetDefaultAccount(); a != nil {
					resolvedAccount = a
				}
			}
			if resolvedAccount != nil {
				store.SetAccountDefaultModel(r, resolvedAccount.DefaultModel)
			}
		}

		// Peek at body to extract model + session_id + stream flag before handler consumes it
		var model string
		var sessionID string
		var bodyStream bool
		if r.Method == "POST" && r.Body != nil {
			bodyBytes, _ := io.ReadAll(io.LimitReader(r.Body, 100<<20))
			r.Body.Close()
			r.Body = io.NopCloser(bytes.NewReader(bodyBytes))
			var body map[string]interface{}
			if json.Unmarshal(bodyBytes, &body) == nil {
				if m, ok := body["model"].(string); ok {
					model = m
				}
				if s, ok := body["stream"].(bool); ok {
					bodyStream = s
				}
				// Extract session_id from Claude Code metadata
				if meta, ok := body["metadata"].(map[string]interface{}); ok {
					if sid, ok := meta["session_id"].(string); ok && sid != "" {
						sessionID = sid
					}
				}
			}
		}

		// Record session activity for /v1/ proxy requests
		if isProxyInferencePath(r.URL.Path) && resolvedAccount != nil {
			if sessionID == "" {
				sessionID = r.RemoteAddr // fallback: unique per client IP
			}
			proxy.RecordSession(resolvedAccount.UserID, sessionID)
		}

		rw := &responseWriter{ResponseWriter: w, statusCode: 200, bodyLimit: 64 << 10}
		next.ServeHTTP(rw, r)

		// Log /v1/ requests (skip GET /v1/models — it's a lightweight list
		// endpoint with no model/tokens and would pollute stats with empty rows).
		path := r.URL.Path
		if isProxyInferencePath(path) && !(r.Method == http.MethodGet && path == "/v1/models") {
			apiKey := extractAPIKey(r)
			if apiKey != "" {
				if account, _ := s.GetAccountByToken(apiKey); account != nil {
					apiKey = account.UserID
				}
			}
			if apiKey == "" {
				if a, _ := s.GetDefaultAccount(); a != nil {
					apiKey = a.UserID
				}
			}

			isStream := bodyStream || r.URL.Query().Get("stream") != ""
			latency := time.Since(start).Milliseconds()

			loggedStatus := rw.statusCode
			var errMsg string
			outcome := store.GetRequestOutcome(r)
			if outcome.Error != "" {
				loggedStatus = outcome.StatusCode
				if loggedStatus == 0 {
					loggedStatus = http.StatusBadGateway
				}
				errMsg = outcome.Error
			}
			if rw.statusCode >= 400 && errMsg == "" {
				errMsg = fmt.Sprintf("HTTP %d on %s %s", rw.statusCode, r.Method, path)
				if body := strings.TrimSpace(rw.body.String()); body != "" {
					errMsg = fmt.Sprintf("%s\n%s", errMsg, body)
				}
			}
			if errMsg != "" {
				slog.Error("proxy error response",
					"request_id", reqID,
					"status", loggedStatus,
					"http_status", rw.statusCode,
					"method", r.Method,
					"path", path,
					"model", model,
					"latency_ms", latency,
					"api_key", apiKey,
					"error", errMsg,
				)
			}

			var inTk, outTk int
			inTk, outTk = store.GetTokenUsage(r)
			resolvedModel := store.GetModel(r)
			if resolvedModel != "" {
				model = resolvedModel
			}
			if s.GetSetting("enable_request_logging") != "false" {
				writer.Enqueue(store.RequestLogEntry{
					UserID: apiKey, Model: model, Endpoint: path, Stream: isStream,
					StatusCode: loggedStatus, LatencyMs: latency, ErrorMessage: errMsg,
					InputTokens: inTk, OutputTokens: outTk,
				})
			}
		}
	})
}

type responseWriter struct {
	http.ResponseWriter
	statusCode int
	body       bytes.Buffer
	bodyLimit  int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

func (rw *responseWriter) Write(p []byte) (int, error) {
	if rw.bodyLimit > 0 && rw.body.Len() < rw.bodyLimit {
		remaining := rw.bodyLimit - rw.body.Len()
		if len(p) > remaining {
			rw.body.Write(p[:remaining])
		} else {
			rw.body.Write(p)
		}
	}
	return rw.ResponseWriter.Write(p)
}

func (rw *responseWriter) Flush() {
	if f, ok := rw.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func inferenceAuthMiddleware(next http.Handler, s *store.Store, bindHost string) http.Handler {
	ip := net.ParseIP(bindHost)
	if bindHost == "localhost" || (ip != nil && ip.IsLoopback()) {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !isProxyInferencePath(r.URL.Path) || r.Method == http.MethodOptions || (r.Method == http.MethodGet && r.URL.Path == "/v1/models") {
			next.ServeHTTP(w, r)
			return
		}
		key := extractAPIKey(r)
		var account *store.Account
		if key != "" {
			account, _ = s.GetAccountByToken(key)
			if account == nil {
				account, _ = s.GetAccount(key)
			}
		}
		if account == nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error":{"type":"authentication_error","code":"invalid_api_key","message":"A valid account API token is required for non-loopback access"}}`))
			return
		}
		next.ServeHTTP(w, r)
	})
}

func isProxyInferencePath(path string) bool {
	if path == "/v1/messages/count_tokens" {
		return false
	}
	return strings.HasPrefix(path, "/v1/") || path == "/responses"
}

// extractAPIKey pulls the caller's API key from either the x-api-key header
// (Anthropic convention) or the Authorization: Bearer <key> header (OpenAI
// convention). Returns "" if neither is present.
func extractAPIKey(r *http.Request) string {
	if k := r.Header.Get("x-api-key"); k != "" {
		return k
	}
	if a := r.Header.Get("Authorization"); strings.HasPrefix(a, "Bearer ") {
		return strings.TrimPrefix(a, "Bearer ")
	}
	return ""
}

// setupLogRotation initializes rotating log writers for slog and log.
// Also truncates stdout.log if launchd has let it grow too large.
func setupLogRotation() {
	home, err := os.UserHomeDir()
	if err != nil {
		return
	}
	logDir := filepath.Join(home, logDir)

	cfg := logrot.DefaultConfig(logDir, "stderr")
	rw, err := logrot.New(cfg)
	if err != nil {
		log.Printf("Warning: log rotation init failed: %v", err)
		return
	}

	slog.SetDefault(slog.New(slog.NewTextHandler(rw, &slog.HandlerOptions{Level: slog.LevelInfo})))
	log.SetOutput(rw)

	// Truncate stdout.log if launchd has let it grow too large
	stdoutPath := filepath.Join(logDir, "stdout.log")
	logrot.TruncateFileIfNeeded(stdoutPath, cfg.MaxFileSize)
}
