package joycode

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// ToolArguments accepts the two shapes emitted by upstream chat gateways:
// the protocol's usual JSON string and gateways that send the arguments object
// directly. String values may be partial JSON because streaming chunks are
// accumulated by the caller; object values are complete JSON documents.
type ToolArguments string

func (a *ToolArguments) UnmarshalJSON(data []byte) error {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		*a = ""
		return nil
	}
	if trimmed[0] == '"' {
		var s string
		if err := json.Unmarshal(trimmed, &s); err != nil {
			return err
		}
		*a = ToolArguments(s)
		return nil
	}
	if trimmed[0] != '{' || !json.Valid(trimmed) {
		return fmt.Errorf("tool arguments must be a JSON string or object")
	}
	var compact bytes.Buffer
	if err := json.Compact(&compact, trimmed); err != nil {
		return err
	}
	*a = ToolArguments(compact.String())
	return nil
}

// NormalizeToolArguments converts a decoded non-streaming arguments value to
// the string form required by OpenAI Responses and Anthropic tool blocks.
func NormalizeToolArguments(value interface{}) (string, bool) {
	switch v := value.(type) {
	case string:
		return v, true
	case map[string]interface{}:
		b, err := json.Marshal(v)
		if err != nil {
			return "", false
		}
		return string(b), true
	case json.RawMessage:
		var args ToolArguments
		if err := args.UnmarshalJSON(v); err != nil {
			return "", false
		}
		return string(args), true
	default:
		return "", false
	}
}
