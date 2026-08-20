package keepalive

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/vibe-coding-labs/JoyCode2Api/pkg/joycode"
	"github.com/vibe-coding-labs/JoyCode2Api/pkg/store"
)

func newTestStore(t *testing.T) *store.Store {
	t.Helper()
	s, err := store.Open(filepath.Join(t.TempDir(), "proxy.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

func TestCheckOneVerifiesRefreshBeforeSaving(t *testing.T) {
	s := newTestStore(t)
	if err := s.AddAccount("user-1", "old-key", "test", true, ""); err != nil {
		t.Fatal(err)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Header.Get("ptKey") == "old-key" {
			json.NewEncoder(w).Encode(map[string]interface{}{"code": 0, "data": map[string]interface{}{"ptKey": "bad-new-key"}})
			return
		}
		json.NewEncoder(w).Encode(map[string]interface{}{"code": 401, "msg": "invalid candidate"})
	}))
	defer server.Close()
	keeper := NewKeeper(s, time.Hour)
	keeper.newClient = func(key, user string) *joycode.Client {
		client := joycode.NewClient(key, user)
		client.ColorBaseURL = ""
		client.MasterBaseURL = server.URL
		return client
	}
	if result := keeper.checkOne(context.Background(), "user-1", "old-key", "user-1"); result != "valid" {
		t.Fatalf("result=%q, want valid with old key retained", result)
	}
	account, err := s.GetAccount("user-1")
	if err != nil {
		t.Fatal(err)
	}
	if account.PtKey != "old-key" {
		t.Fatalf("stored key=%q, want old-key", account.PtKey)
	}
	status := keeper.GetStatus("user-1")
	if status == nil || !status.Valid || status.ErrorMessage == "" {
		t.Fatalf("status=%+v", status)
	}
}

func TestStopCancelsInFlightCheck(t *testing.T) {
	s := newTestStore(t)
	if err := s.AddAccount("user-1", "old-key", "test", true, ""); err != nil {
		t.Fatal(err)
	}
	started := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		close(started)
		select {
		case <-r.Context().Done():
		case <-time.After(100 * time.Millisecond):
		}
	}))
	defer server.Close()
	keeper := NewKeeper(s, 0)
	keeper.newClient = func(key, user string) *joycode.Client {
		client := joycode.NewClient(key, user)
		client.ColorBaseURL = ""
		client.MasterBaseURL = server.URL
		return client
	}
	keeper.Start(time.Hour)
	<-started
	done := make(chan struct{})
	go func() { keeper.Stop(); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Stop did not cancel and wait for in-flight check")
	}
	keeper.Stop() // idempotent
}

func TestStatusGettersReturnCopies(t *testing.T) {
	keeper := NewKeeper(nil, time.Hour)
	keeper.setStatus("a", CredentialStatus{Valid: true})
	one := keeper.GetStatus("a")
	one.Valid = false
	all := keeper.GetAllStatuses()
	all["a"].Valid = false
	if !keeper.GetStatus("a").Valid {
		t.Fatal("caller mutated internal status")
	}
}
