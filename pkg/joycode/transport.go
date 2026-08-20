package joycode

import (
	"net/http"
	"sync/atomic"
	"time"
)

// TransportManager is a stable RoundTripper whose underlying immutable
// transport can be atomically replaced when connection limits change.
type TransportManager struct {
	current atomic.Pointer[http.Transport]
}

func NewTransportManager(maxConns int) *TransportManager {
	m := &TransportManager{}
	m.Update(maxConns)
	return m
}

func newManagedTransport(maxConns int) *http.Transport {
	if maxConns < 1 {
		maxConns = 1
	}
	idle := maxConns / 2
	if idle < 2 {
		idle = 2
	}
	return &http.Transport{
		Proxy:               http.ProxyFromEnvironment,
		MaxIdleConns:        maxConns * 4,
		MaxIdleConnsPerHost: idle,
		MaxConnsPerHost:     maxConns,
		IdleConnTimeout:     90 * time.Second,
	}
}

func (m *TransportManager) RoundTrip(req *http.Request) (*http.Response, error) {
	return m.current.Load().RoundTrip(req)
}

func (m *TransportManager) Update(maxConns int) {
	next := newManagedTransport(maxConns)
	previous := m.current.Swap(next)
	if previous != nil {
		previous.CloseIdleConnections()
	}
}

func (m *TransportManager) CloseIdleConnections() {
	if transport := m.current.Load(); transport != nil {
		transport.CloseIdleConnections()
	}
}
