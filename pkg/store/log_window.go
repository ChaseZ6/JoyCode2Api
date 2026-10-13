package store

import "time"

const monthLogFilter = "created_at >= ? AND created_at <= ?"

type MonthToDateSummary struct {
	UsageTotals
	MonthStart time.Time `json:"month_start"`
	MonthEnd   time.Time `json:"month_end"`
	AsOf       time.Time `json:"as_of"`
}

// GetMonthToDateSummary keeps usage and calendar progress on the same service-local clock.
func (s *Store) GetMonthToDateSummary() (*MonthToDateSummary, error) {
	return s.getMonthToDateSummaryAt(time.Now())
}

func (s *Store) getMonthToDateSummaryAt(now time.Time) (*MonthToDateSummary, error) {
	totals, err := s.getMonthToDateTotalsAt(now)
	if err != nil {
		return nil, err
	}
	start := startOfMonth(now)
	return &MonthToDateSummary{
		UsageTotals: *totals,
		MonthStart:  start,
		MonthEnd:    start.AddDate(0, 1, 0),
		AsOf:        now,
	}, nil
}

func startOfMonth(now time.Time) time.Time {
	return time.Date(now.Year(), now.Month(), 1, 0, 0, 0, 0, now.Location())
}

// Log timestamps are stored in local wall time. Do not convert them to UTC or
// apply SQLite's localtime modifier again when comparing them with this window.
func monthLogWindow(now time.Time) (string, string) {
	const layout = "2006-01-02 15:04:05"
	return startOfMonth(now).Format(layout), now.Format(layout)
}
