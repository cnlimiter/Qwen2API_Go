package openai

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"qwen2api/internal/logging"
	"qwen2api/internal/metrics"
	"qwen2api/internal/toolcall"
)

func TestNormalizeResponsesRequest(t *testing.T) {
	t.Run("string input", func(t *testing.T) {
		normalized, err := normalizeResponsesRequest(responsesRequest{
			Model: "qwen3",
			Input: json.RawMessage(`"hello"`),
		})
		if err != nil {
			t.Fatalf("normalizeResponsesRequest() error = %v", err)
		}
		want := []map[string]any{{"role": "user", "content": "hello"}}
		if !reflect.DeepEqual(normalized.Messages, want) {
			t.Fatalf("messages = %#v, want %#v", normalized.Messages, want)
		}
	})

	t.Run("item input and flat function tool", func(t *testing.T) {
		instructions := "follow policy"
		normalized, err := normalizeResponsesRequest(responsesRequest{
			Model:        "qwen3",
			Instructions: &instructions,
			Input: json.RawMessage(`[
				{"type":"message","role":"developer","content":[{"type":"input_text","text":"be concise"}]},
				{"type":"message","role":"user","content":"weather"},
				{"type":"function_call","call_id":"call_1","name":"weather","arguments":"{\"city\":\"Hangzhou\"}"},
				{"type":"function_call_output","call_id":"call_1","output":"sunny"}
			]`),
			Tools: []responseFunctionTool{{
				Type: "function",
				Name: "weather",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"city": map[string]any{"type": "string"},
					},
				},
			}},
			ToolChoice: map[string]any{"type": "function", "name": "weather"},
		})
		if err != nil {
			t.Fatalf("normalizeResponsesRequest() error = %v", err)
		}
		if len(normalized.Messages) != 5 {
			t.Fatalf("messages len = %d, want 5", len(normalized.Messages))
		}
		if normalized.Messages[0]["role"] != "system" || normalized.Messages[1]["role"] != "system" {
			t.Fatalf("instruction/developer roles = %#v", normalized.Messages[:2])
		}
		assistant := normalized.Messages[3]
		toolCalls, _ := assistant["tool_calls"].([]any)
		call, _ := toolCalls[0].(map[string]any)
		function, _ := call["function"].(map[string]any)
		if call["id"] != "call_1" || function["name"] != "weather" || function["arguments"] != `{"city":"Hangzhou"}` {
			t.Fatalf("assistant function call = %#v", assistant)
		}
		toolResult := normalized.Messages[4]
		if toolResult["role"] != "tool" || toolResult["tool_call_id"] != "call_1" || toolResult["name"] != "weather" {
			t.Fatalf("tool result = %#v", toolResult)
		}

		chatTool := normalized.ChatTools[0].(map[string]any)
		chatFunction := chatTool["function"].(map[string]any)
		if chatFunction["name"] != "weather" || chatFunction["parameters"] == nil {
			t.Fatalf("chat tool lost name or parameters: %#v", chatTool)
		}
		chatChoice := normalized.ChatToolChoice.(map[string]any)
		choiceFunction := chatChoice["function"].(map[string]any)
		if choiceFunction["name"] != "weather" {
			t.Fatalf("chat tool choice = %#v", chatChoice)
		}
	})
}

func TestNormalizeResponseInputRejectsInvalidInput(t *testing.T) {
	invalid := []json.RawMessage{
		json.RawMessage(`null`),
		json.RawMessage(`[]`),
		json.RawMessage(`[{"type":"message","role":"user","content":[{"type":"input_audio"}]}]`),
		json.RawMessage(`[{"type":"function_call","call_id":"call_1","name":"weather","arguments":"not-json"}]`),
		json.RawMessage(`[{"type":"function_call_output","call_id":"call_1","output":"sunny"}]`),
		json.RawMessage(`[
			{"type":"function_call","call_id":"call_1","name":"weather","arguments":"{}"},
			{"type":"function_call","call_id":"call_1","name":"weather","arguments":"{}"}
		]`),
	}
	for _, input := range invalid {
		if _, err := normalizeResponseInput(input); err == nil {
			t.Fatalf("normalizeResponseInput(%s) error = nil, want error", input)
		}
	}
}

func TestHandleResponseNonStreamText(t *testing.T) {
	handler := responseTestHandler()
	recorder := httptest.NewRecorder()
	upstream := `{"choices":[{"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`

	handler.handleResponseNonStream(recorder, strings.NewReader(upstream), responseContext{
		ID: "resp_test", CreatedAt: 123, Model: "qwen3", Metadata: map[string]any{}, ParallelToolCalls: true, ToolChoice: "auto", Tools: []map[string]any{},
	}, "qwen3", nil, nil, 1)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}
	var response map[string]any
	decodeJSONForTest(t, recorder.Body.String(), &response)
	if response["id"] != "resp_test" || response["object"] != "response" || response["status"] != "completed" {
		t.Fatalf("response identity/status = %#v", response)
	}
	output := response["output"].([]any)
	message := output[0].(map[string]any)
	content := message["content"].([]any)
	part := content[0].(map[string]any)
	if message["type"] != "message" || part["type"] != "output_text" || part["text"] != "hello" {
		t.Fatalf("message output = %#v", message)
	}
	usage := response["usage"].(map[string]any)
	if usage["input_tokens"] != float64(3) || usage["output_tokens"] != float64(2) || usage["total_tokens"] != float64(5) {
		t.Fatalf("usage = %#v", usage)
	}
}

func TestHandleResponseNonStreamFunctionCall(t *testing.T) {
	handler := responseTestHandler()
	recorder := httptest.NewRecorder()
	upstream := `{"choices":[{"message":{"role":"assistant","content":"<tool_calls><tool_call><tool_name>weather</tool_name><parameters><count><![CDATA[3]]></count></parameters></tool_call></tool_calls>"}}]}`
	schemas := []toolcall.ToolSchema{{
		Name: "weather",
		Parameters: map[string]any{
			"type":       "object",
			"properties": map[string]any{"count": map[string]any{"type": "integer"}},
		},
	}}

	handler.handleResponseNonStream(recorder, strings.NewReader(upstream), responseContext{
		ID: "resp_tool", CreatedAt: 123, Model: "qwen3", Metadata: map[string]any{}, ParallelToolCalls: true, ToolChoice: "auto", Tools: []map[string]any{},
	}, "qwen3", []string{"weather"}, schemas, 1)

	var response map[string]any
	decodeJSONForTest(t, recorder.Body.String(), &response)
	output := response["output"].([]any)
	if len(output) != 1 {
		t.Fatalf("output len = %d, want 1", len(output))
	}
	call := output[0].(map[string]any)
	if call["type"] != "function_call" || call["name"] != "weather" || call["arguments"] != `{"count":3}` {
		t.Fatalf("function output = %#v", call)
	}
	if !strings.HasPrefix(call["call_id"].(string), "call_") || !strings.HasPrefix(call["id"].(string), "fc_") {
		t.Fatalf("function identifiers = %#v", call)
	}
}

func TestHandleResponseStreamTextEventSequence(t *testing.T) {
	handler := responseTestHandler()
	recorder := httptest.NewRecorder()
	upstream := strings.Join([]string{
		`data: {"choices":[{"delta":{"role":"assistant","content":"hel"}}]}`,
		"",
		`data: {"choices":[{"delta":{"role":"assistant","content":"lo"}}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`,
		"",
		`data: [DONE]`,
		"",
	}, "\n")

	handler.handleResponseStream(recorder, strings.NewReader(upstream), responseContext{
		ID: "resp_stream", CreatedAt: 123, Model: "qwen3", Metadata: map[string]any{}, ParallelToolCalls: true, ToolChoice: "auto", Tools: []map[string]any{},
	}, "qwen3", nil, nil, 1)

	events := decodeResponseEvents(t, recorder.Body.String())
	wantTypes := []string{
		"response.created",
		"response.in_progress",
		"response.output_item.added",
		"response.content_part.added",
		"response.output_text.delta",
		"response.output_text.delta",
		"response.output_text.done",
		"response.content_part.done",
		"response.output_item.done",
		"response.completed",
	}
	assertResponseEventTypes(t, events, wantTypes)
	if events[4]["delta"] != "hel" || events[5]["delta"] != "lo" || events[6]["text"] != "hello" {
		t.Fatalf("text events = %#v", events[4:7])
	}
	completed := events[len(events)-1]["response"].(map[string]any)
	if completed["status"] != "completed" {
		t.Fatalf("completed response = %#v", completed)
	}
}

func TestHandleResponseStreamFunctionCallEvents(t *testing.T) {
	handler := responseTestHandler()
	recorder := httptest.NewRecorder()
	upstream := strings.Join([]string{
		`data: {"choices":[{"delta":{"role":"assistant","content":"<tool_calls><tool_call><tool_name>weather</tool_name><parameters><city><![CDATA[Hangzhou]]></city></parameters></tool_call></tool_calls>"}}]}`,
		"",
		`data: [DONE]`,
		"",
	}, "\n")

	handler.handleResponseStream(recorder, strings.NewReader(upstream), responseContext{
		ID: "resp_stream_tool", CreatedAt: 123, Model: "qwen3", Metadata: map[string]any{}, ParallelToolCalls: true, ToolChoice: "auto", Tools: []map[string]any{},
	}, "qwen3", []string{"weather"}, []toolcall.ToolSchema{{Name: "weather", Parameters: map[string]any{"type": "object"}}}, 1)

	events := decodeResponseEvents(t, recorder.Body.String())
	wantTypes := []string{
		"response.created",
		"response.in_progress",
		"response.output_item.added",
		"response.function_call_arguments.delta",
		"response.function_call_arguments.done",
		"response.output_item.done",
		"response.completed",
	}
	assertResponseEventTypes(t, events, wantTypes)
	added := events[2]["item"].(map[string]any)
	delta := events[3]
	done := events[4]
	if added["type"] != "function_call" || added["status"] != "in_progress" || added["arguments"] != "" {
		t.Fatalf("function added event = %#v", added)
	}
	if delta["item_id"] != added["id"] || delta["delta"] != `{"city":"Hangzhou"}` {
		t.Fatalf("function delta event = %#v", delta)
	}
	if done["name"] != "weather" || done["arguments"] != `{"city":"Hangzhou"}` {
		t.Fatalf("function done event = %#v", done)
	}
}

func TestHandleResponsesInvalidInputErrorEnvelope(t *testing.T) {
	handler := &Handler{}
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"qwen3","input":[]}`))

	handler.HandleResponses(recorder, request)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadRequest)
	}
	var payload map[string]any
	decodeJSONForTest(t, recorder.Body.String(), &payload)
	errorObject := payload["error"].(map[string]any)
	if errorObject["type"] != "invalid_request_error" || errorObject["param"] != "input" || errorObject["code"] != "invalid_value" {
		t.Fatalf("error envelope = %#v", errorObject)
	}
}

func responseTestHandler() *Handler {
	return &Handler{
		logger:  logging.New(false),
		metrics: metrics.NewDashboardStats(),
	}
}

func decodeJSONForTest(t *testing.T, raw string, target any) {
	t.Helper()
	if err := json.Unmarshal([]byte(raw), target); err != nil {
		t.Fatalf("json.Unmarshal() error = %v, raw=%s", err, raw)
	}
}

func decodeResponseEvents(t *testing.T, body string) []map[string]any {
	t.Helper()
	blocks := strings.Split(strings.TrimSpace(body), "\n\n")
	events := make([]map[string]any, 0, len(blocks))
	for _, block := range blocks {
		for _, line := range strings.Split(block, "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			var event map[string]any
			decodeJSONForTest(t, strings.TrimPrefix(line, "data: "), &event)
			events = append(events, event)
		}
	}
	return events
}

func assertResponseEventTypes(t *testing.T, events []map[string]any, want []string) {
	t.Helper()
	got := make([]string, 0, len(events))
	for index, event := range events {
		got = append(got, stringValue(event["type"]))
		if event["sequence_number"] != float64(index) {
			t.Fatalf("event[%d].sequence_number = %v, want %d", index, event["sequence_number"], index)
		}
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("event types = %#v, want %#v", got, want)
	}
}
