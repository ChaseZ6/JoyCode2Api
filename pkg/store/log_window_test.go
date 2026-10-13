package store

import (
	"testing"
	"time"
)

func insertUsageLog(t *testing.T, s *Store, at time.Time, userID string, status, input int) {
	t.Helper()
	_, err := s.db.Exec(`INSERT INTO request_logs
		(api_key, model, endpoint, stream, status_code, latency_ms, input_tokens, output_tokens, created_at)
		VALUES (?, 'model', '/v1/responses', 1, ?, 100, ?, 10, ?)`,
		userID, status, input, at.Format("2006-01-02 15:04:05"))
	if err != nil {
		t.Fatal(err)
	}
}

func TestMonthToDateRetentionBoundaries(t *testing.T) {
	loc := time.FixedZone("UTC+8", 8*60*60)
	for _, now := range []time.Time{
		time.Date(2026, time.January, 1, 0, 0, 0, 0, loc),
		time.Date(2028, time.March, 1, 0, 30, 0, 0, loc),
		time.Date(2026, time.October, 31, 23, 59, 59, 0, loc),
	} {
		t.Run(now.Format("2006-01-02"), func(t *testing.T) {
			s := openTestStore(t)
			start := time.Date(now.Year(), now.Month(), 1, 0, 0, 0, 0, loc)
			insertUsageLog(t, s, start.Add(-time.Second), "account", 500, 1000)
			insertUsageLog(t, s, start, "account", 200, 100)
			insertUsageLog(t, s, now, "account", 500, 200)
			insertUsageLog(t, s, now, "other", 200, 300)
			insertUsageLog(t, s, now, "", 200, 400)
			insertUsageLog(t, s, now.Add(time.Second), "account", 200, 2000)

			// Query before cleanup: stale rows must never inflate this month.
			totals, err := s.getMonthToDateTotalsAt(now)
			if err != nil {
				t.Fatal(err)
			}
			if *totals != (UsageTotals{TotalRequests: 4, TotalInputTk: 1000, TotalOutputTk: 40, ErrorCount: 1}) {
				t.Fatalf("global month-to-date = %+v", totals)
			}
			summary, err := s.getMonthToDateSummaryAt(now)
			if err != nil {
				t.Fatal(err)
			}
			if summary.UsageTotals != *totals || !summary.MonthStart.Equal(start) ||
				!summary.MonthEnd.Equal(start.AddDate(0, 1, 0)) || !summary.AsOf.Equal(now) {
				t.Fatalf("month summary uses a different usage window: %+v", summary)
			}
			if summary.MonthStart.Location() != loc || summary.MonthEnd.Location() != loc || summary.AsOf.Location() != loc {
				t.Fatalf("month summary lost the service timezone: %+v", summary)
			}
			account, err := s.getAccountStatsAt("account", now)
			if err != nil {
				t.Fatal(err)
			}
			if *account.MonthToDate != (UsageTotals{TotalRequests: 2, TotalInputTk: 300, TotalOutputTk: 20, ErrorCount: 1}) {
				t.Fatalf("account month-to-date = %+v", account.MonthToDate)
			}
			var hourlyCount int
			for _, hour := range account.Hourly {
				hourlyCount += hour.Count
			}
			wantHourly := 1
			if now.Sub(start) < 24*time.Hour {
				wantHourly = 2
			}
			if hourlyCount != wantHourly {
				t.Fatalf("hourly count = %d, want %d", hourlyCount, wantHourly)
			}
			unassigned, err := s.getAccountStatsAt("", now)
			if err != nil || unassigned.MonthToDate.TotalRequests != 1 {
				t.Fatalf("unassigned account = %+v, error = %v", unassigned, err)
			}

			deleted, err := s.cleanupPreviousMonthLogsAt(now)
			if err != nil || deleted != 1 {
				t.Fatalf("cleanup deleted = %d, error = %v; want only last month's row", deleted, err)
			}
			after, err := s.getMonthToDateTotalsAt(now)
			if err != nil || *after != *totals {
				t.Fatalf("cleanup changed current-month totals: %+v, error = %v", after, err)
			}
			if deleted, err = s.cleanupPreviousMonthLogsAt(now); err != nil || deleted != 0 {
				t.Fatalf("repeat cleanup deleted = %d, error = %v", deleted, err)
			}
			// Advance to next month while the same store remains open.
			nextMonth := start.AddDate(0, 1, 0)
			nextTotals, err := s.getMonthToDateTotalsAt(nextMonth)
			if err != nil {
				t.Fatal(err)
			}
			wantNext := UsageTotals{}
			if now.Add(time.Second).Equal(nextMonth) {
				wantNext = UsageTotals{TotalRequests: 1, TotalInputTk: 2000, TotalOutputTk: 10}
			}
			if *nextTotals != wantNext {
				t.Fatalf("month rollover = %+v, want %+v", nextTotals, wantNext)
			}
			if _, err = s.cleanupPreviousMonthLogsAt(nextMonth); err != nil {
				t.Fatal(err)
			}
			var oldRows int
			if err := s.db.QueryRow("SELECT COUNT(*) FROM request_logs WHERE created_at < ?", nextMonth.Format("2006-01-02 15:04:05")).Scan(&oldRows); err != nil || oldRows != 0 {
				t.Fatalf("rows from previous month = %d, error = %v", oldRows, err)
			}
		})
	}
}

func TestMonthToDateQueriesBeforeCleanup(t *testing.T) {
	s := openTestStore(t)
	now := time.Now()
	start := time.Date(now.Year(), now.Month(), 1, 0, 0, 0, 0, now.Location())
	insertUsageLog(t, s, start.Add(-time.Second), "account", 500, 1000)
	insertUsageLog(t, s, start, "account", 200, 100)
	insertUsageLog(t, s, now, "account", 500, 200)
	insertUsageLog(t, s, start.AddDate(0, 1, 0), "account", 500, 2000)

	totals, err := s.GetMonthToDateTotals()
	if err != nil || totals.TotalRequests != 2 || totals.TotalInputTk != 300 {
		t.Fatalf("month-to-date = %+v, error = %v", totals, err)
	}
	accounts := []AccountInfo{{UserID: "account"}}
	s.FillAccountStats(accounts)
	if accounts[0].TotalRequests != 2 || accounts[0].TotalTokens != 320 {
		t.Fatalf("account list = %+v", accounts[0])
	}
	logs, total, err := s.GetAccountLogsPage("account", LogQuery{Limit: 1})
	if err != nil || total != 2 || len(logs) != 1 || logs[0].InputTokens != 200 {
		t.Fatalf("account page = %+v, total = %d, error = %v", logs, total, err)
	}
	logs, total, err = s.GetAccountLogsPage("account", LogQuery{Limit: 1, Offset: 1})
	if err != nil || total != 2 || len(logs) != 1 || logs[0].InputTokens != 100 {
		t.Fatalf("second account page = %+v, total = %d, error = %v", logs, total, err)
	}
	logs, total, err = s.GetAccountLogsPage("account", LogQuery{Kind: "error", Q: "responses"})
	if err != nil || total != 1 || len(logs) != 1 || logs[0].InputTokens != 200 {
		t.Fatalf("filtered account page = %+v, total = %d, error = %v", logs, total, err)
	}
	logs, err = s.GetRecentLogs(100)
	if err != nil || len(logs) != 2 {
		t.Fatalf("recent logs = %+v, error = %v", logs, err)
	}
	logs, err = s.GetRecentErrors(100)
	if err != nil || len(logs) != 1 || logs[0].InputTokens != 200 {
		t.Fatalf("recent errors = %+v, error = %v", logs, err)
	}
	hourly, err := s.GetHourlyStats()
	if err != nil {
		t.Fatal(err)
	}
	var input int
	for _, hour := range hourly {
		input += hour.InputTokens
	}
	if input != 200 && input != 300 {
		t.Fatalf("hourly includes previous-month or future usage: %+v", hourly)
	}
	if deleted, err := s.CleanupPreviousMonthLogs(); err != nil || deleted != 1 {
		t.Fatalf("cleanup deleted = %d, error = %v", deleted, err)
	}
}
