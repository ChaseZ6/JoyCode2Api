package common

import (
	"errors"
	"testing"
)

func TestIsTimeoutErrorRecognizesColorGateway504(t *testing.T) {
	for _, message := range []string{
		`upstream error: {"code":"COLOR_FORWARD_EXCEPTION","message":"HttpStatus=504"}`,
		`invalid JSON response: 504 Gateway Time-out`,
		`API error 504: gateway unavailable`,
	} {
		if !IsTimeoutError(errors.New(message)) {
			t.Errorf("IsTimeoutError(%q) = false, want true", message)
		}
	}
}

func TestIsTimeoutErrorDoesNotMatchCancellation(t *testing.T) {
	if IsTimeoutError(errors.New("context canceled")) {
		t.Fatal("context cancellation is not an upstream timeout")
	}
}
