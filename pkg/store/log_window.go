package store

import "time"

const monthLogFilter = "created_at >= ? AND created_at <= ?"

func startOfMonth(now time.Time) time.Time {
	return time.Date(now.Year(), now.Month(), 1, 0, 0, 0, 0, now.Location())
}

// Log timestamps are stored in local wall time. Do not convert them to UTC or
// apply SQLite's localtime modifier again when comparing them with this window.
func monthLogWindow(now time.Time) (string, string) {
	const layout = "2006-01-02 15:04:05"
	return startOfMonth(now).Format(layout), now.Format(layout)
}
