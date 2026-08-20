package keepalive

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

// CredentialStatus represents the health of an account's credentials.
type CredentialStatus struct {
	Valid         bool      `json:"valid"`
	LastChecked   time.Time `json:"last_checked"`
	LastRefreshed time.Time `json:"last_refreshed,omitempty"`
	ErrorMessage  string    `json:"error_message,omitempty"`
}

// Keeper runs periodic keep-alive checks for all accounts.
type Keeper struct {
	store      *store.Store
	mu         sync.RWMutex
	status     map[string]*CredentialStatus
	lifecycle  sync.Mutex
	checkMu    sync.Mutex
	running    bool
	cancel     context.CancelFunc
	wg         sync.WaitGroup
	refreshTTL time.Duration
	newClient  func(string, string) *joycode.Client
}

func NewKeeper(s *store.Store, refreshTTL time.Duration) *Keeper {
	return &Keeper{store: s, status: make(map[string]*CredentialStatus), refreshTTL: refreshTTL, newClient: joycode.NewClient}
}

func (k *Keeper) GetStatus(apiKey string) *CredentialStatus {
	k.mu.RLock()
	defer k.mu.RUnlock()
	if value := k.status[apiKey]; value != nil {
		copy := *value
		return &copy
	}
	return nil
}

func (k *Keeper) GetAllStatuses() map[string]*CredentialStatus {
	k.mu.RLock()
	defer k.mu.RUnlock()
	result := make(map[string]*CredentialStatus, len(k.status))
	for key, value := range k.status {
		copy := *value
		result[key] = &copy
	}
	return result
}

// Start begins the periodic keep-alive loop. Calling Start while already
// running is a no-op; a stopped Keeper can be started again safely.
func (k *Keeper) Start(checkInterval time.Duration) {
	k.lifecycle.Lock()
	if k.running || k.store == nil {
		k.lifecycle.Unlock()
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	k.cancel = cancel
	k.running = true
	k.wg.Add(2)
	k.lifecycle.Unlock()

	go func() {
		defer k.wg.Done()
		k.checkStale(ctx)
	}()
	go func() {
		defer k.wg.Done()
		ticker := time.NewTicker(checkInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				k.checkStale(ctx)
			case <-ctx.Done():
				return
			}
		}
	}()
	slog.Info("keepalive: started", "check_interval", checkInterval, "refresh_ttl", k.refreshTTL)
}

// Stop cancels in-flight checks and waits for all Keeper goroutines to exit.
func (k *Keeper) Stop() {
	k.lifecycle.Lock()
	if !k.running {
		k.lifecycle.Unlock()
		return
	}
	cancel := k.cancel
	k.running = false
	k.cancel = nil
	k.lifecycle.Unlock()
	cancel()
	k.wg.Wait()
	slog.Info("keepalive: stopped")
}

func (k *Keeper) checkStale(ctx context.Context) {
	k.checkMu.Lock()
	defer k.checkMu.Unlock()
	if ctx.Err() != nil {
		return
	}
	accounts, err := k.store.ListStaleAccounts(k.refreshTTL)
	if err != nil {
		slog.Error("keepalive: failed to list stale accounts", "error", err)
		return
	}
	if len(accounts) == 0 {
		return
	}
	startTime := time.Now()
	slog.Info("keepalive: round started", "stale_count", len(accounts), "refresh_ttl", k.refreshTTL)
	var validCount, refreshedCount, failedCount int
	for i, account := range accounts {
		if ctx.Err() != nil {
			return
		}
		slog.Info("keepalive: checking account", "progress", fmt.Sprintf("%d/%d", i+1, len(accounts)), "user_id", account.UserID)
		switch k.checkOne(ctx, account.UserID, account.PtKey, account.UserID) {
		case "valid":
			validCount++
		case "refreshed":
			refreshedCount++
		case "failed":
			failedCount++
		}
		if i < len(accounts)-1 {
			timer := time.NewTimer(5 * time.Second)
			select {
			case <-timer.C:
			case <-ctx.Done():
				timer.Stop()
				return
			}
		}
	}
	slog.Info("keepalive: round completed", "duration", time.Since(startTime).Round(time.Millisecond),
		"valid", validCount, "refreshed", refreshedCount, "failed", failedCount, "total", len(accounts))
}

func (k *Keeper) setStatus(apiKey string, status CredentialStatus) {
	k.mu.Lock()
	defer k.mu.Unlock()
	copy := status
	k.status[apiKey] = &copy
}

// checkOne validates a single account and verifies a refreshed key before it is
// persisted. The known-working key is never overwritten by an unverified one.
func (k *Keeper) checkOne(ctx context.Context, apiKey, ptKey, userID string) string {
	if userID == "" {
		return "failed"
	}
	checkStart := time.Now()
	client := k.newClient(ptKey, userID)
	client.SetTimeout(30 * time.Second)
	refreshedPtKey, err := client.UserInfoWithRefreshContext(ctx)
	now := time.Now()
	if err != nil {
		if ctx.Err() != nil {
			return "failed"
		}
		slog.Warn("keepalive: account check failed", "user_id", userID, "error", err,
			"duration", time.Since(checkStart), "pt_key_prefix", maskKey(ptKey))
		k.setStatus(apiKey, CredentialStatus{Valid: false, LastChecked: now, ErrorMessage: err.Error()})
		if dbErr := k.store.SetCredentialValid(apiKey, false); dbErr != nil {
			slog.Error("keepalive: failed to persist invalid status", "user_id", userID, "error", dbErr)
		}
		return "failed"
	}

	status := CredentialStatus{Valid: true, LastChecked: now}
	result := "valid"
	if refreshedPtKey != "" && refreshedPtKey != ptKey {
		verifyClient := k.newClient(refreshedPtKey, userID)
		verifyClient.SetTimeout(15 * time.Second)
		if verifyErr := verifyClient.ValidateContext(ctx); verifyErr != nil {
			status.ErrorMessage = "refresh candidate rejected: " + verifyErr.Error()
			slog.Warn("keepalive: refreshed pt_key verification failed; keeping existing key", "user_id", userID, "error", verifyErr)
		} else if saveErr := k.store.UpdatePtKey(apiKey, refreshedPtKey); saveErr != nil {
			status.ErrorMessage = "failed to save refreshed credential: " + saveErr.Error()
			slog.Error("keepalive: failed to save verified pt_key", "user_id", userID, "error", saveErr)
		} else {
			status.LastRefreshed = now
			result = "refreshed"
			slog.Info("keepalive: refreshed pt_key verified and saved", "user_id", userID)
		}
	} else if err := k.store.UpdateCredentialRefreshedAt(apiKey); err != nil {
		status.ErrorMessage = "failed to update refresh time: " + err.Error()
		slog.Error("keepalive: failed to persist refresh time", "user_id", userID, "error", err)
	}
	if err := k.store.SetCredentialValid(apiKey, true); err != nil {
		slog.Error("keepalive: failed to persist valid status", "user_id", userID, "error", err)
	}
	k.setStatus(apiKey, status)
	return result
}

func maskKey(key string) string {
	if len(key) <= 12 {
		return "***"
	}
	return key[:6] + "..." + key[len(key)-6:]
}
