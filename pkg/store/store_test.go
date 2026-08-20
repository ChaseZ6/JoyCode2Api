package store

import (
	"bytes"
	"database/sql"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

func openTestStore(t *testing.T) *Store {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")
	s, err := Open(dbPath)
	if err != nil {
		t.Fatalf("open store: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

// --- Database init ---

func TestOpenCreatesDB(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")
	s, err := Open(dbPath)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer s.Close()

	if _, err := os.Stat(dbPath); os.IsNotExist(err) {
		t.Error("database file was not created")
	}
}

func TestOpenCreatesDir(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "sub", "dir", "test.db")
	s, err := Open(dbPath)
	if err != nil {
		t.Fatalf("open with nested dir: %v", err)
	}
	defer s.Close()
}

func TestDefaultDBPath(t *testing.T) {
	path, err := DefaultDBPath()
	if err != nil {
		t.Fatalf("default path: %v", err)
	}
	if path == "" {
		t.Error("expected non-empty default path")
	}
}

// --- Encryption ---

func TestEncryptDecrypt(t *testing.T) {
	s := openTestStore(t)

	plaintext := "super-secret-pt-key-12345"
	encrypted, err := s.encrypt(plaintext)
	if err != nil {
		t.Fatalf("encrypt: %v", err)
	}
	if encrypted == plaintext {
		t.Error("encrypted should differ from plaintext")
	}

	decrypted, err := s.decrypt(encrypted)
	if err != nil {
		t.Fatalf("decrypt: %v", err)
	}
	if decrypted != plaintext {
		t.Errorf("decrypt = %q, want %q", decrypted, plaintext)
	}
}

func TestEncryptDifferentEachTime(t *testing.T) {
	s := openTestStore(t)

	plaintext := "same-input"
	enc1, _ := s.encrypt(plaintext)
	enc2, _ := s.encrypt(plaintext)
	if enc1 == enc2 {
		t.Error("two encryptions of same input should produce different ciphertext (random nonce)")
	}
}

func TestDecryptInvalidCiphertext(t *testing.T) {
	s := openTestStore(t)

	_, err := s.decrypt("not-valid-hex!")
	if err == nil {
		t.Error("expected error for invalid hex")
	}
}

func TestDecryptTooShort(t *testing.T) {
	s := openTestStore(t)

	_, err := s.decrypt("ab")
	if err == nil {
		t.Error("expected error for too-short ciphertext")
	}
}

// --- Account CRUD ---

func TestAddAndListAccounts(t *testing.T) {
	s := openTestStore(t)

	err := s.AddAccount("key1", "pt1", "user1", true, "JoyAI-Code")
	if err != nil {
		t.Fatalf("add: %v", err)
	}

	accounts, err := s.ListAccounts()
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if len(accounts) != 1 {
		t.Fatalf("len = %d, want 1", len(accounts))
	}
	if accounts[0].UserID != "key1" {
		t.Errorf("UserID = %q, want %q", accounts[0].UserID, "key1")
	}
	if accounts[0].Nickname != "user1" {
		t.Errorf("Nickname = %q, want %q", accounts[0].Nickname, "user1")
	}
	if !accounts[0].IsDefault {
		t.Error("expected IsDefault = true")
	}
	if accounts[0].DefaultModel != "JoyAI-Code" {
		t.Errorf("DefaultModel = %q, want %q", accounts[0].DefaultModel, "JoyAI-Code")
	}
}

func TestListAccountsEmpty(t *testing.T) {
	s := openTestStore(t)

	accounts, err := s.ListAccounts()
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	if accounts != nil {
		t.Errorf("expected nil for empty, got %v", accounts)
	}
}

func TestAddMultipleAccounts(t *testing.T) {
	s := openTestStore(t)

	s.AddAccount("key1", "pt1", "user1", true, "")
	s.AddAccount("key2", "pt2", "user2", false, "GLM-5.1")

	accounts, _ := s.ListAccounts()
	if len(accounts) != 2 {
		t.Fatalf("len = %d, want 2", len(accounts))
	}

	// Only one default
	defaultCount := 0
	for _, a := range accounts {
		if a.IsDefault {
			defaultCount++
		}
	}
	if defaultCount != 1 {
		t.Errorf("default count = %d, want 1", defaultCount)
	}
}

func TestAddAccountOverwrites(t *testing.T) {
	s := openTestStore(t)

	s.AddAccount("key1", "pt1", "user1", true, "")
	s.AddAccount("key1", "pt1-updated", "user1-new", false, "GLM-5.1")

	accounts, _ := s.ListAccounts()
	if len(accounts) != 1 {
		t.Fatalf("len = %d, want 1 (upsert)", len(accounts))
	}

	a, _ := s.GetAccount("key1")
	if a.PtKey != "pt1-updated" {
		t.Errorf("PtKey = %q, want %q", a.PtKey, "pt1-updated")
	}
}

func TestGetAccount(t *testing.T) {
	s := openTestStore(t)

	s.AddAccount("key1", "secret-pt-key", "user1", true, "JoyAI-Code")

	a, err := s.GetAccount("key1")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if a == nil {
		t.Fatal("expected account, got nil")
	}
	if a.PtKey != "secret-pt-key" {
		t.Errorf("PtKey = %q, want %q", a.PtKey, "secret-pt-key")
	}
	if a.UserID != "key1" {
		t.Errorf("UserID = %q, want %q", a.UserID, "key1")
	}
}

func TestGetAccountNotFound(t *testing.T) {
	s := openTestStore(t)

	a, err := s.GetAccount("nonexistent")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if a != nil {
		t.Error("expected nil for nonexistent key")
	}
}

func TestRemoveAccount(t *testing.T) {
	s := openTestStore(t)

	s.AddAccount("key1", "pt1", "user1", true, "")
	s.RemoveAccount("key1")

	accounts, _ := s.ListAccounts()
	if len(accounts) != 0 {
		t.Errorf("len = %d, want 0 after remove", len(accounts))
	}
}

func TestRemoveNonexistent(t *testing.T) {
	s := openTestStore(t)

	err := s.RemoveAccount("nonexistent")
	if err != nil {
		t.Errorf("remove nonexistent should not error: %v", err)
	}
}

func TestSetDefault(t *testing.T) {
	s := openTestStore(t)

	s.AddAccount("key1", "pt1", "user1", true, "")
	s.AddAccount("key2", "pt2", "user2", false, "")

	s.SetDefault("key2")

	accounts, _ := s.ListAccounts()
	for _, a := range accounts {
		if a.UserID == "key1" && a.IsDefault {
			t.Error("key1 should no longer be default")
		}
		if a.UserID == "key2" && !a.IsDefault {
			t.Error("key2 should be default")
		}
	}
}

func TestUpdateAccountModel(t *testing.T) {
	s := openTestStore(t)

	s.AddAccount("key1", "pt1", "user1", true, "JoyAI-Code")
	s.UpdateAccountModel("key1", "GLM-5.1")

	a, _ := s.GetAccount("key1")
	if a.DefaultModel != "GLM-5.1" {
		t.Errorf("DefaultModel = %q, want %q", a.DefaultModel, "GLM-5.1")
	}
}

func TestGetDefaultAccount(t *testing.T) {
	s := openTestStore(t)

	s.AddAccount("key1", "pt1", "user1", false, "")
	s.AddAccount("key2", "pt2", "user2", true, "JoyAI-Code")

	a, err := s.GetDefaultAccount()
	if err != nil {
		t.Fatalf("get default: %v", err)
	}
	if a == nil {
		t.Fatal("expected default account, got nil")
	}
	if a.UserID != "key2" {
		t.Errorf("default UserID = %q, want %q", a.UserID, "key2")
	}
}

func TestGetDefaultAccountNone(t *testing.T) {
	s := openTestStore(t)

	a, err := s.GetDefaultAccount()
	if err != nil {
		t.Fatalf("get default: %v", err)
	}
	if a != nil {
		t.Error("expected nil when no default account")
	}
}

// --- Settings ---

func TestGetSettingsEmpty(t *testing.T) {
	s := openTestStore(t)

	settings, err := s.GetSettings()
	if err != nil {
		t.Fatalf("get settings: %v", err)
	}
	if len(settings) != 0 {
		t.Errorf("expected empty settings, got %v", settings)
	}
}

func TestSetAndGetSetting(t *testing.T) {
	s := openTestStore(t)

	s.SetSetting("key1", "value1")
	s.SetSetting("key2", "value2")

	settings, _ := s.GetSettings()
	if settings["key1"] != "value1" {
		t.Errorf("key1 = %q, want %q", settings["key1"], "value1")
	}
	if settings["key2"] != "value2" {
		t.Errorf("key2 = %q, want %q", settings["key2"], "value2")
	}
}

func TestSetSettingsBatch(t *testing.T) {
	s := openTestStore(t)

	err := s.SetSettings(map[string]string{
		"a": "1",
		"b": "2",
	})
	if err != nil {
		t.Fatalf("set settings batch: %v", err)
	}

	settings, _ := s.GetSettings()
	if len(settings) != 2 {
		t.Errorf("len = %d, want 2", len(settings))
	}
}

func TestSetSettingOverwrite(t *testing.T) {
	s := openTestStore(t)

	s.SetSetting("key", "old")
	s.SetSetting("key", "new")

	settings, _ := s.GetSettings()
	if settings["key"] != "new" {
		t.Errorf("key = %q, want %q", settings["key"], "new")
	}
}

// --- Request Logging & Stats ---

func TestLogRequestAndGetStats(t *testing.T) {
	s := openTestStore(t)

	s.LogRequest("key1", "JoyAI-Code", "/v1/chat/completions", true, 200, 500, "", 0, 0)
	s.LogRequest("key1", "GLM-5.1", "/v1/chat/completions", false, 200, 300, "", 0, 0)
	s.LogRequest("key2", "JoyAI-Code", "/v1/messages", true, 200, 400, "", 0, 0)

	stats, err := s.GetStats()
	if err != nil {
		t.Fatalf("get stats: %v", err)
	}
	if stats.TotalRequests != 3 {
		t.Errorf("TotalRequests = %d, want 3", stats.TotalRequests)
	}
	if stats.AccountsCount != 0 {
		t.Errorf("AccountsCount = %d, want 0 (no accounts added)", stats.AccountsCount)
	}
	if len(stats.ByModel) != 2 {
		t.Errorf("ByModel len = %d, want 2", len(stats.ByModel))
	}
	if len(stats.ByAccount) != 1 || stats.ByAccount[0].UserID != "其他" {
		t.Errorf("ByAccount = %v, want [{其他 3}]", stats.ByAccount)
	}
}

func TestGetStatsEmpty(t *testing.T) {
	s := openTestStore(t)

	stats, err := s.GetStats()
	if err != nil {
		t.Fatalf("get stats: %v", err)
	}
	if stats.TotalRequests != 0 {
		t.Errorf("TotalRequests = %d, want 0", stats.TotalRequests)
	}
}

func TestGetAccountStats(t *testing.T) {
	s := openTestStore(t)

	s.LogRequest("key1", "JoyAI-Code", "/v1/chat/completions", true, 200, 500, "", 0, 0)
	s.LogRequest("key1", "GLM-5.1", "/v1/messages", false, 200, 300, "", 0, 0)
	s.LogRequest("key1", "JoyAI-Code", "/v1/chat/completions", true, 500, 100, "", 0, 0)

	stats, err := s.GetAccountStats("key1")
	if err != nil {
		t.Fatalf("get account stats: %v", err)
	}
	if stats.TotalRequests != 3 {
		t.Errorf("TotalRequests = %d, want 3", stats.TotalRequests)
	}
	if stats.StreamCount != 2 {
		t.Errorf("StreamCount = %d, want 2", stats.StreamCount)
	}
	if stats.ErrorCount != 1 {
		t.Errorf("ErrorCount = %d, want 1", stats.ErrorCount)
	}
	if len(stats.ByModel) != 2 {
		t.Errorf("ByModel len = %d, want 2", len(stats.ByModel))
	}
	if len(stats.ByEndpoint) != 2 {
		t.Errorf("ByEndpoint len = %d, want 2", len(stats.ByEndpoint))
	}
}

func TestGetAccountStatsEmpty(t *testing.T) {
	s := openTestStore(t)

	stats, err := s.GetAccountStats("nonexistent")
	if err != nil {
		t.Fatalf("get account stats: %v", err)
	}
	if stats.TotalRequests != 0 {
		t.Errorf("TotalRequests = %d, want 0", stats.TotalRequests)
	}
}

func TestGetRecentLogs(t *testing.T) {
	s := openTestStore(t)

	s.LogRequest("key1", "model1", "/v1/test", true, 200, 100, "", 0, 0)
	s.LogRequest("key2", "model2", "/v1/test", false, 200, 200, "", 0, 0)
	s.LogRequest("key1", "model3", "/v1/test", true, 200, 300, "", 0, 0)

	logs, err := s.GetRecentLogs(2)
	if err != nil {
		t.Fatalf("get recent logs: %v", err)
	}
	if len(logs) != 2 {
		t.Errorf("len = %d, want 2", len(logs))
	}
	// Most recent first
	if logs[0].Model != "model3" {
		t.Errorf("first log model = %q, want %q", logs[0].Model, "model3")
	}
}

func TestGetRecentLogsDefault(t *testing.T) {
	s := openTestStore(t)

	s.LogRequest("key1", "m1", "/v1", true, 200, 100, "", 0, 0)
	s.LogRequest("key1", "m2", "/v1", true, 200, 100, "", 0, 0)

	logs, err := s.GetRecentLogs(0)
	if err != nil {
		t.Fatalf("get recent logs: %v", err)
	}
	// Default limit = 100
	if len(logs) != 2 {
		t.Errorf("len = %d, want 2", len(logs))
	}
}

// --- Encryption key persistence ---

func TestEncryptionKeyReused(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	// Open twice with same dir — should reuse same encryption key
	s1, _ := Open(dbPath)
	enc1, _ := s1.encrypt("hello")
	s1.Close()

	s2, _ := Open(dbPath)
	dec2, err := s2.decrypt(enc1)
	s2.Close()

	if err != nil {
		t.Fatalf("decrypt with reopened store: %v", err)
	}
	if dec2 != "hello" {
		t.Errorf("decrypted = %q, want %q", dec2, "hello")
	}
}

// TestGetHourlyStatsUsesLocalHour guards against the double-localtime bug:
// created_at is stored as local time, so the hourly bucket must be
// strftime(created_at) — applying 'localtime' again shifts the hour label by
// the UTC offset (visible on non-UTC servers; a no-op exactly at UTC).
func TestGetHourlyStatsUsesLocalHour(t *testing.T) {
	s := openTestStore(t)
	if err := s.LogRequest("k1", "GLM-5.1", "/v1/chat", true, 200, 100, "", 1, 2); err != nil {
		t.Fatalf("log request: %v", err)
	}

	// Expected bucket: the stored (local) timestamp formatted as-is.
	var want string
	if err := s.db.QueryRow("SELECT strftime('%m-%d %H', 'now', 'localtime')").Scan(&want); err != nil {
		t.Fatalf("compute expected hour: %v", err)
	}

	hourly, err := s.GetHourlyStats()
	if err != nil {
		t.Fatalf("hourly stats: %v", err)
	}
	if len(hourly) != 1 {
		t.Fatalf("hourly rows = %d, want 1: %+v", len(hourly), hourly)
	}
	if hourly[0].Hour != want {
		t.Errorf("hour = %q, want %q (double localtime conversion?)", hourly[0].Hour, want)
	}
	if hourly[0].Count != 1 {
		t.Errorf("count = %d, want 1", hourly[0].Count)
	}
}

func TestGetAccountStats_UsesCalendarDayAndRollingHourlyWindow(t *testing.T) {
	s := openTestStore(t)
	loc := time.FixedZone("UTC+8", 8*60*60)
	now := time.Date(2026, time.August, 11, 11, 0, 0, 0, loc)

	insert := func(createdAt, model, endpoint string, stream, status, latency, input, output int) {
		t.Helper()
		_, err := s.db.Exec(`
			INSERT INTO request_logs
				(api_key, model, endpoint, stream, status_code, latency_ms, error_message, input_tokens, output_tokens, created_at)
			VALUES (?, ?, ?, ?, ?, ?, '', ?, ?, ?)`,
			"key1", model, endpoint, stream, status, latency, input, output, createdAt)
		if err != nil {
			t.Fatal(err)
		}
	}

	insert("2026-08-10 12:00:00", "Yesterday", "/yesterday", 0, 200, 900, 100, 10)
	insert("2026-08-11 00:00:00", "Today-A", "/today-a", 1, 200, 100, 20, 2)
	insert("2026-08-11 10:30:00", "Today-B", "/today-b", 0, 500, 300, 30, 3)
	insert("2026-08-11 23:59:59", "Today-A", "/today-a", 1, 200, 200, 40, 4)
	insert("2026-08-12 00:00:00", "Tomorrow", "/tomorrow", 0, 200, 700, 50, 5)
	insert("2026-08-10 10:59:59", "Old", "/old", 0, 200, 800, 60, 6)

	stats, err := s.getAccountStatsAt("key1", now)
	if err != nil {
		t.Fatal(err)
	}
	if stats.TotalRequests != 3 || stats.SuccessCount != 2 || stats.ErrorCount != 1 || stats.StreamCount != 2 {
		t.Errorf("daily counts = total %d success %d error %d stream %d, want 3/2/1/2",
			stats.TotalRequests, stats.SuccessCount, stats.ErrorCount, stats.StreamCount)
	}
	if stats.TotalInputTk != 90 || stats.TotalOutputTk != 9 {
		t.Errorf("daily tokens = %d/%d, want 90/9", stats.TotalInputTk, stats.TotalOutputTk)
	}
	if stats.AvgLatencyMs != 200 {
		t.Errorf("daily avg latency = %v, want 200", stats.AvgLatencyMs)
	}

	models := map[string]int{}
	for _, item := range stats.ByModel {
		models[item.Model] = item.Count
	}
	if len(models) != 2 || models["Today-A"] != 2 || models["Today-B"] != 1 {
		t.Errorf("daily models = %#v", models)
	}
	endpoints := map[string]int{}
	for _, item := range stats.ByEndpoint {
		endpoints[item.Endpoint] = item.Count
	}
	if len(endpoints) != 2 || endpoints["/today-a"] != 2 || endpoints["/today-b"] != 1 {
		t.Errorf("daily endpoints = %#v", endpoints)
	}

	if stats.AllTime.TotalRequests != 6 || stats.AllTime.TotalInputTk != 300 || stats.AllTime.TotalOutputTk != 30 || stats.AllTime.ErrorCount != 1 {
		t.Errorf("all-time = %+v, want all six records", stats.AllTime)
	}

	hourly := map[string]int{}
	for _, item := range stats.Hourly {
		hourly[item.Hour] = item.Count
	}
	if hourly["08-10 12"] != 1 || hourly["08-11 00"] != 1 || hourly["08-11 10"] != 1 {
		t.Errorf("rolling hourly = %#v, want yesterday 12:00 and today's elapsed hours", hourly)
	}
	if _, exists := hourly["08-10 10"]; exists {
		t.Errorf("rolling hourly includes record older than 24h: %#v", hourly)
	}
	if _, exists := hourly["08-11 23"]; exists {
		t.Errorf("rolling hourly includes future same-day record: %#v", hourly)
	}
	if _, exists := hourly["08-12 00"]; exists {
		t.Errorf("rolling hourly includes tomorrow record: %#v", hourly)
	}
}

func TestEncryptionKey_InvalidExistingFileIsNotOverwritten(t *testing.T) {
	dir := t.TempDir()
	keyPath := filepath.Join(dir, encKeyFile)
	original := []byte("corrupt-key")
	if err := os.WriteFile(keyPath, original, 0600); err != nil {
		t.Fatal(err)
	}
	s := &Store{}
	if _, err := s.loadOrCreateEncKey(dir); err == nil {
		t.Fatal("expected invalid existing key to fail")
	}
	got, _ := os.ReadFile(keyPath)
	if !bytes.Equal(got, original) {
		t.Fatalf("invalid key was overwritten: %q", got)
	}
}

func TestEncryptionKey_RejectsSymlink(t *testing.T) {
	dir := t.TempDir()
	target := filepath.Join(dir, "target")
	if err := os.WriteFile(target, []byte(strings.Repeat("0", 64)), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, filepath.Join(dir, encKeyFile)); err != nil {
		t.Skipf("symlink unavailable: %v", err)
	}
	if _, err := (&Store{}).loadOrCreateEncKey(dir); err == nil {
		t.Fatal("expected symlink key path to be rejected")
	}
}

func TestEncryptionKey_ConcurrentCreationUsesSingleKey(t *testing.T) {
	dir := t.TempDir()
	const workers = 16
	results := make(chan []byte, workers)
	errs := make(chan error, workers)
	for i := 0; i < workers; i++ {
		go func() {
			key, err := (&Store{}).loadOrCreateEncKey(dir)
			if err != nil {
				errs <- err
				return
			}
			results <- key
		}()
	}
	var first []byte
	for i := 0; i < workers; i++ {
		select {
		case err := <-errs:
			t.Fatal(err)
		case key := <-results:
			if first == nil {
				first = key
			} else if !bytes.Equal(first, key) {
				t.Fatal("concurrent creators returned different keys")
			}
		}
	}
	info, err := os.Stat(filepath.Join(dir, encKeyFile))
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm() != 0600 {
		t.Errorf("key permissions = %o, want 600", info.Mode().Perm())
	}
}

func TestEncryptionKey_RestrictsExistingPermissions(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX permissions")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, encKeyFile)
	if err := os.WriteFile(path, []byte(strings.Repeat("0", 64)), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := (&Store{}).loadOrCreateEncKey(dir); err != nil {
		t.Fatal(err)
	}
	info, _ := os.Stat(path)
	if info.Mode().Perm() != 0600 {
		t.Errorf("permissions = %o", info.Mode().Perm())
	}
}

func TestSchemaMigrationBaselineIsIdempotent(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "proxy.db")
	s, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	var count int
	if err := s.db.QueryRow("SELECT COUNT(*) FROM schema_migrations WHERE version=1 AND name='legacy_localtime_baseline'").Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Fatalf("migration count=%d", count)
	}
	s.Close()
	s, err = Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	if err := s.db.QueryRow("SELECT COUNT(*) FROM schema_migrations WHERE version=1").Scan(&count); err != nil {
		t.Fatal(err)
	}
	if count != 1 {
		t.Fatalf("migration repeated, count=%d", count)
	}
}

func TestStoreRejectsNewerSchemaVersion(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "proxy.db")
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatal(err)
	}
	_, err = db.Exec("CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT); INSERT INTO schema_migrations(version,name) VALUES (99,'future')")
	db.Close()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Open(path); err == nil || !strings.Contains(err.Error(), "newer than supported") {
		t.Fatalf("Open error=%v", err)
	}
}
