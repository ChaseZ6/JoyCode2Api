package joycode

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestTransportManagerUpdate(t *testing.T) {
	manager := NewTransportManager(20)
	first := manager.current.Load()
	if first.MaxConnsPerHost != 20 || first.MaxIdleConnsPerHost != 10 {
		t.Fatalf("first transport = %+v", first)
	}
	manager.Update(8)
	second := manager.current.Load()
	if first == second || second.MaxConnsPerHost != 8 || second.MaxIdleConnsPerHost != 4 {
		t.Fatalf("second transport = %+v", second)
	}
}

func TestTransportManagerConcurrentUpdateAndRoundTrip(t *testing.T) {
	manager := NewTransportManager(10)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(204) }))
	defer server.Close()
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 1; i <= 100; i++ {
			manager.Update(i)
		}
	}()
	for i := 0; i < 100; i++ {
		req, _ := http.NewRequest(http.MethodGet, server.URL, nil)
		resp, err := manager.RoundTrip(req)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
	}
	<-done
}
