package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"qwen2api/internal/account"
	"qwen2api/internal/config"
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
			Tools: responseRawTools(t, map[string]any{
				"type": "function",
				"name": "weather",
				"parameters": map[string]any{
					"type": "object",
					"properties": map[string]any{
						"city": map[string]any{"type": "string"},
					},
				},
			}),
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
		ToolDefinitions: []responseToolDefinition{{Type: responseToolFunction, Name: "weather", InternalName: "weather"}},
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
		ToolDefinitions: []responseToolDefinition{{Type: responseToolFunction, Name: "weather", InternalName: "weather"}},
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

func TestNormalizeResponseToolsSupportsCodexLocalToolSet(t *testing.T) {
	normalized := normalizeMixedResponseTools(t)
	if len(normalized.ChatTools) != 5 || len(normalized.ResponseTools) != 5 || len(normalized.ToolDefinitions) != 5 {
		t.Fatalf("normalized tools lengths = chat:%d response:%d definitions:%d", len(normalized.ChatTools), len(normalized.ResponseTools), len(normalized.ToolDefinitions))
	}
	seenInternal := make(map[string]struct{}, len(normalized.ToolDefinitions))
	for _, definition := range normalized.ToolDefinitions {
		if _, exists := seenInternal[definition.InternalName]; exists {
			t.Fatalf("duplicate internal name %q", definition.InternalName)
		}
		seenInternal[definition.InternalName] = struct{}{}
		if definition.Type != responseToolFunction && definition.InternalName == definition.Name {
			t.Fatalf("%s internal name was not isolated: %#v", definition.Type, definition)
		}
	}

	customEcho := normalized.ResponseTools[1]
	format := customEcho["format"].(map[string]any)
	if customEcho["name"] != "freeform" || format["type"] != "grammar" || format["syntax"] != "regex" || format["definition"] != `^[a-z]+$` {
		t.Fatalf("custom echo lost caller fields: %#v", customEcho)
	}
	for _, echoed := range normalized.ResponseTools {
		if _, exists := echoed["function"]; exists {
			t.Fatalf("response tool leaked synthetic schema: %#v", echoed)
		}
	}
}

func TestNormalizeResponseToolsAvoidsSyntheticNameCollisions(t *testing.T) {
	_, _, registry, err := normalizeResponseTools(responseRawTools(t,
		map[string]any{"type": "function", "name": "freeform", "parameters": map[string]any{"type": "object"}},
		map[string]any{"type": "custom", "name": "freeform", "format": map[string]any{"type": "text"}},
		map[string]any{"type": "function", "name": "__responses_custom_1", "parameters": map[string]any{"type": "object"}},
	))
	if err != nil {
		t.Fatalf("normalizeResponseTools() error = %v", err)
	}
	publicFunctionName := registry.internalName(responseToolFunction, "", "freeform")
	collisionFunctionName := registry.internalName(responseToolFunction, "", "__responses_custom_1")
	customName := registry.internalName(responseToolCustom, "", "freeform")
	if publicFunctionName != "freeform" || customName != "__responses_custom_1_1" || collisionFunctionName == customName {
		t.Fatalf("collision mapping = public function:%q collision function:%q custom:%q", publicFunctionName, collisionFunctionName, customName)
	}
}

func TestNormalizeResponseInputSupportsLocalToolRoundTrips(t *testing.T) {
	input := json.RawMessage(`[
		{"type":"function_call","call_id":"call_f","name":"weather","arguments":"{\"city\":\"Hangzhou\"}"},
		{"type":"function_call_output","call_id":"call_f","output":"sunny"},
		{"type":"custom_tool_call","call_id":"call_c","name":"freeform","input":"alpha"},
		{"type":"custom_tool_call_output","call_id":"call_c","output":"accepted"},
		{"type":"local_shell_call","call_id":"call_l","action":{"type":"exec","command":["git","status"],"timeout_ms":1000,"working_directory":"repo","env":{"A":"B"},"user":"codex"}},
		{"type":"local_shell_call_output","call_id":"call_l","output":"{\"stdout\":\"clean\"}"},
		{"type":"shell_call","call_id":"call_s","action":{"commands":["pwd","git status"],"timeout_ms":1000,"max_output_length":4096}},
		{"type":"shell_call_output","call_id":"call_s","output":[{"stdout":"repo","stderr":"","outcome":{"type":"exit","exit_code":0}}]},
		{"type":"apply_patch_call","call_id":"call_a","operation":{"type":"update_file","path":"a.txt","diff":"@@ -1 +1 @@\n-old\n+new"}},
		{"type":"apply_patch_call_output","call_id":"call_a","status":"completed","output":"Done"}
	]`)
	tools := mixedResponseRawTools(t)
	normalized, err := normalizeResponsesRequest(responsesRequest{Model: "qwen3", Input: input, Tools: tools})
	if err != nil {
		t.Fatalf("normalizeResponsesRequest() error = %v", err)
	}
	if len(normalized.Messages) != 10 {
		t.Fatalf("messages len = %d, want 10", len(normalized.Messages))
	}
	for index := 0; index < len(normalized.Messages); index += 2 {
		assistant := normalized.Messages[index]
		call := assistant["tool_calls"].([]any)[0].(map[string]any)
		function := call["function"].(map[string]any)
		result := normalized.Messages[index+1]
		if result["role"] != "tool" || result["tool_call_id"] != call["id"] || result["name"] != function["name"] {
			t.Fatalf("round-trip messages %d/%d are not linked: assistant=%#v result=%#v", index, index+1, assistant, result)
		}
	}
}

func TestNormalizeResponseInputRejectsToolCallAssociationErrors(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{
			name:  "unknown call id",
			input: `[{"type":"custom_tool_call_output","call_id":"missing","output":"x"}]`,
		},
		{
			name:  "mismatched output type",
			input: `[{"type":"custom_tool_call","call_id":"call_1","name":"freeform","input":"x"},{"type":"function_call_output","call_id":"call_1","output":"x"}]`,
		},
		{
			name:  "duplicate output",
			input: `[{"type":"custom_tool_call","call_id":"call_1","name":"freeform","input":"x"},{"type":"custom_tool_call_output","call_id":"call_1","output":"x"},{"type":"custom_tool_call_output","call_id":"call_1","output":"x"}]`,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, err := normalizeResponsesRequest(responsesRequest{Model: "qwen3", Input: json.RawMessage(tt.input), Tools: mixedResponseRawTools(t)}); err == nil {
				t.Fatal("normalizeResponsesRequest() error = nil, want association error")
			}
		})
	}
}

func TestHandleResponseNonStreamNativeToolCalls(t *testing.T) {
	normalized := normalizeMixedResponseTools(t)
	tests := []struct {
		name       string
		toolType   string
		parameters string
		assert     func(*testing.T, map[string]any)
	}{
		{
			name:       "custom",
			toolType:   responseToolCustom,
			parameters: `<input><![CDATA[alpha]]></input>`,
			assert: func(t *testing.T, item map[string]any) {
				if item["type"] != "custom_tool_call" || item["name"] != "freeform" || item["input"] != "alpha" {
					t.Fatalf("custom item = %#v", item)
				}
			},
		},
		{
			name:       "local shell",
			toolType:   responseToolLocalShell,
			parameters: `<command><![CDATA[["git","status"]]]></command><timeout_ms><![CDATA[1000]]></timeout_ms><env><![CDATA[{"A":"B"}]]></env>`,
			assert: func(t *testing.T, item map[string]any) {
				action := item["action"].(map[string]any)
				if item["type"] != "local_shell_call" || action["type"] != "exec" || !reflect.DeepEqual(action["command"], []any{"git", "status"}) || action["timeout_ms"] != float64(1000) {
					t.Fatalf("local shell item = %#v", item)
				}
			},
		},
		{
			name:       "shell",
			toolType:   responseToolShell,
			parameters: `<commands><![CDATA[["pwd","git status"]]]></commands><max_output_length><![CDATA[4096]]></max_output_length>`,
			assert: func(t *testing.T, item map[string]any) {
				action := item["action"].(map[string]any)
				if item["type"] != "shell_call" || !reflect.DeepEqual(action["commands"], []any{"pwd", "git status"}) || action["max_output_length"] != float64(4096) {
					t.Fatalf("shell item = %#v", item)
				}
			},
		},
		{
			name:     "apply patch",
			toolType: responseToolApplyPatch,
			parameters: `<operation><![CDATA[update_file]]></operation><path><![CDATA[a.txt]]></path><diff><![CDATA[@@ -1 +1 @@
-old
+new]]></diff>`,
			assert: func(t *testing.T, item map[string]any) {
				operation := item["operation"].(map[string]any)
				if item["type"] != "apply_patch_call" || operation["type"] != "update_file" || operation["path"] != "a.txt" {
					t.Fatalf("apply patch item = %#v", item)
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			definition := responseDefinition(t, normalized.ToolDefinitions, tt.toolType)
			content := fmt.Sprintf("<tool_calls><tool_call><tool_name>%s</tool_name><parameters>%s</parameters></tool_call></tool_calls>", definition.InternalName, tt.parameters)
			rawUpstream, err := json.Marshal(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": content}}}})
			if err != nil {
				t.Fatalf("json.Marshal(upstream) error = %v", err)
			}
			recorder := httptest.NewRecorder()
			responseTestHandler().handleResponseNonStream(recorder, strings.NewReader(string(rawUpstream)), responseContext{
				ID: "resp_native", CreatedAt: 123, Model: "qwen3", Metadata: map[string]any{}, ParallelToolCalls: true,
				ToolChoice: "auto", Tools: normalized.ResponseTools, ToolDefinitions: normalized.ToolDefinitions,
			}, "qwen3", responseToolNames(normalized.ToolDefinitions), responseSchemas(normalized.ChatTools), 1)
			if recorder.Code != http.StatusOK {
				t.Fatalf("status = %d body=%s", recorder.Code, recorder.Body.String())
			}
			var response map[string]any
			decodeJSONForTest(t, recorder.Body.String(), &response)
			output := response["output"].([]any)
			if len(output) != 1 {
				t.Fatalf("output len = %d, want 1: %#v", len(output), output)
			}
			item := output[0].(map[string]any)
			if item["status"] != "completed" || !strings.HasPrefix(item["call_id"].(string), "call_") {
				t.Fatalf("native item identifiers/status = %#v", item)
			}
			tt.assert(t, item)
		})
	}
}

func TestHandleResponseStreamCustomToolCallEvents(t *testing.T) {
	normalized := normalizeMixedResponseTools(t)
	definition := responseDefinition(t, normalized.ToolDefinitions, responseToolCustom)
	upstream := fmt.Sprintf("data: {\"choices\":[{\"delta\":{\"content\":\"<tool_calls><tool_call><tool_name>%s</tool_name><parameters><input><![CDATA[alpha]]></input></parameters></tool_call></tool_calls>\"}}]}\n\ndata: [DONE]\n\n", definition.InternalName)
	recorder := httptest.NewRecorder()
	responseTestHandler().handleResponseStream(recorder, strings.NewReader(upstream), responseContext{
		ID: "resp_custom_stream", CreatedAt: 123, Model: "qwen3", Metadata: map[string]any{}, ParallelToolCalls: true,
		ToolChoice: "auto", Tools: normalized.ResponseTools, ToolDefinitions: normalized.ToolDefinitions,
	}, "qwen3", responseToolNames(normalized.ToolDefinitions), responseSchemas(normalized.ChatTools), 1)

	events := decodeResponseEvents(t, recorder.Body.String())
	assertResponseEventTypes(t, events, []string{
		"response.created", "response.in_progress", "response.output_item.added",
		"response.custom_tool_call_input.delta", "response.custom_tool_call_input.done",
		"response.output_item.done", "response.completed",
	})
	added := events[2]["item"].(map[string]any)
	doneItem := events[5]["item"].(map[string]any)
	completed := events[6]["response"].(map[string]any)
	completedItem := completed["output"].([]any)[0].(map[string]any)
	if added["type"] != "custom_tool_call" || added["status"] != "in_progress" || added["input"] != "" {
		t.Fatalf("custom added item = %#v", added)
	}
	if events[3]["delta"] != "alpha" || events[4]["input"] != "alpha" || !reflect.DeepEqual(doneItem, completedItem) {
		t.Fatalf("custom stream events/items mismatch: %#v", events[3:])
	}
}

func TestHandleResponseStreamNativeToolCallItems(t *testing.T) {
	normalized := normalizeMixedResponseTools(t)
	tests := []struct {
		toolType   string
		parameters string
		itemType   string
	}{
		{responseToolLocalShell, `<command><![CDATA[["pwd"]]]></command>`, "local_shell_call"},
		{responseToolShell, `<commands><![CDATA[["pwd"]]]></commands>`, "shell_call"},
		{responseToolApplyPatch, `<operation><![CDATA[delete_file]]></operation><path><![CDATA[old.txt]]></path>`, "apply_patch_call"},
	}
	for _, tt := range tests {
		t.Run(tt.toolType, func(t *testing.T) {
			definition := responseDefinition(t, normalized.ToolDefinitions, tt.toolType)
			content := fmt.Sprintf("<tool_calls><tool_call><tool_name>%s</tool_name><parameters>%s</parameters></tool_call></tool_calls>", definition.InternalName, tt.parameters)
			delta, err := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"content": content}}}})
			if err != nil {
				t.Fatalf("json.Marshal(delta) error = %v", err)
			}
			upstream := fmt.Sprintf("data: %s\n\ndata: [DONE]\n\n", delta)
			recorder := httptest.NewRecorder()
			responseTestHandler().handleResponseStream(recorder, strings.NewReader(upstream), responseContext{
				ID: "resp_native_stream", CreatedAt: 123, Model: "qwen3", Metadata: map[string]any{}, ParallelToolCalls: true,
				ToolChoice: "auto", Tools: normalized.ResponseTools, ToolDefinitions: normalized.ToolDefinitions,
			}, "qwen3", responseToolNames(normalized.ToolDefinitions), responseSchemas(normalized.ChatTools), 1)
			events := decodeResponseEvents(t, recorder.Body.String())
			assertResponseEventTypes(t, events, []string{
				"response.created", "response.in_progress", "response.output_item.added", "response.output_item.done", "response.completed",
			})
			added := events[2]["item"].(map[string]any)
			done := events[3]["item"].(map[string]any)
			completed := events[4]["response"].(map[string]any)["output"].([]any)[0].(map[string]any)
			if added["type"] != tt.itemType || added["status"] != "in_progress" || done["status"] != "completed" || !reflect.DeepEqual(done, completed) {
				t.Fatalf("native stream item mismatch: added=%#v done=%#v completed=%#v", added, done, completed)
			}
		})
	}
}

func TestNormalizeResponseToolChoiceSupportsLocalToolTypes(t *testing.T) {
	normalized := normalizeMixedResponseTools(t)
	registry := registryFromDefinitions(normalized.ToolDefinitions)
	tests := []map[string]any{
		{"type": "function", "name": "weather"},
		{"type": "custom", "name": "freeform"},
		{"type": "local_shell"},
		{"type": "shell"},
		{"type": "apply_patch"},
	}
	for _, choice := range tests {
		choice := choice
		t.Run(stringValue(choice["type"]), func(t *testing.T) {
			chatChoice, responseChoice, err := normalizeResponseToolChoice(choice, registry)
			if err != nil {
				t.Fatalf("normalizeResponseToolChoice() error = %v", err)
			}
			internal := chatChoice.(map[string]any)["function"].(map[string]any)["name"]
			if internal == "" || !reflect.DeepEqual(responseChoice, choice) {
				t.Fatalf("tool choice mapping = chat:%#v response:%#v", chatChoice, responseChoice)
			}
		})
	}
	if _, _, err := normalizeResponseToolChoice(map[string]any{"type": "custom", "name": "weather"}, registry); err == nil {
		t.Fatal("type-mismatched custom choice error = nil")
	}
}

func TestNormalizeResponseToolsRejectsHostedExecutionTools(t *testing.T) {
	for _, toolType := range []string{"file_search", "web_search_preview", "computer_use_preview", "code_interpreter", "image_generation", "mcp"} {
		t.Run(toolType, func(t *testing.T) {
			_, _, _, err := normalizeResponseTools(responseRawTools(t, map[string]any{"type": toolType}))
			if err == nil || !strings.Contains(err.Error(), "hosted execution") {
				t.Fatalf("normalizeResponseTools() error = %v, want hosted execution error", err)
			}
		})
	}
}

func TestNormalizeResponseToolsValidatesCustomFormats(t *testing.T) {
	tests := []struct {
		name    string
		format  map[string]any
		wantErr bool
	}{
		{name: "text", format: map[string]any{"type": "text"}},
		{name: "lark grammar", format: map[string]any{"type": "grammar", "syntax": "lark", "definition": `start: WORD`}},
		{name: "invalid grammar syntax", format: map[string]any{"type": "grammar", "syntax": "peg", "definition": `word`}, wantErr: true},
		{name: "missing grammar definition", format: map[string]any{"type": "grammar", "syntax": "regex"}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, _, err := normalizeResponseTools(responseRawTools(t, map[string]any{"type": "custom", "name": "freeform", "format": tt.format}))
			if (err != nil) != tt.wantErr {
				t.Fatalf("normalizeResponseTools() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestNormalizeResponseNamespaceSupportsFunctionAndCustomMembers(t *testing.T) {
	rawNamespace := responseNamespaceRawTool(t)
	chatTools, responseTools, registry, err := normalizeResponseTools([]json.RawMessage{rawNamespace})
	if err != nil {
		t.Fatalf("normalizeResponseTools() error = %v", err)
	}
	if len(chatTools) != 2 || len(responseTools) != 1 || len(registry.definitions) != 2 {
		t.Fatalf("namespace normalization lengths = chat:%d response:%d definitions:%d", len(chatTools), len(responseTools), len(registry.definitions))
	}
	function := registry.byTypeName[responseToolKey(responseToolFunction, "github", "get_issue")]
	custom := registry.byTypeName[responseToolKey(responseToolCustom, "github", "query")]
	if function.Namespace != "github" || function.Name != "get_issue" || function.InternalName == "get_issue" {
		t.Fatalf("function namespace mapping = %#v", function)
	}
	if custom.Namespace != "github" || custom.Name != "query" || custom.InternalName == "query" || custom.InternalName == function.InternalName {
		t.Fatalf("custom namespace mapping = %#v", custom)
	}
	echoedMembers := responseTools[0]["tools"].([]any)
	if responseTools[0]["type"] != "namespace" || responseTools[0]["name"] != "github" || len(echoedMembers) != 2 {
		t.Fatalf("namespace echo = %#v", responseTools[0])
	}
	if echoedMembers[0].(map[string]any)["defer_loading"] != true || echoedMembers[1].(map[string]any)["format"] == nil {
		t.Fatalf("namespace member echo lost semantic fields: %#v", echoedMembers)
	}
}

func TestNormalizeResponseNamespaceRejectsMalformedDefinitions(t *testing.T) {
	tests := []struct {
		name string
		tool map[string]any
	}{
		{name: "missing namespace name", tool: map[string]any{"type": "namespace", "description": "tools", "tools": []any{map[string]any{"type": "function", "name": "f", "parameters": map[string]any{"type": "object"}}}}},
		{name: "empty members", tool: map[string]any{"type": "namespace", "name": "apps", "description": "tools", "tools": []any{}}},
		{name: "unsupported member", tool: map[string]any{"type": "namespace", "name": "apps", "description": "tools", "tools": []any{map[string]any{"type": "shell"}}}},
		{name: "function parameters not object", tool: map[string]any{"type": "namespace", "name": "apps", "description": "tools", "tools": []any{map[string]any{"type": "function", "name": "f", "parameters": "object"}}}},
		{name: "custom format invalid", tool: map[string]any{"type": "namespace", "name": "apps", "description": "tools", "tools": []any{map[string]any{"type": "custom", "name": "c", "format": map[string]any{"type": "grammar", "syntax": "peg", "definition": "x"}}}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if _, _, _, err := normalizeResponseTools(responseRawTools(t, tt.tool)); err == nil {
				t.Fatal("normalizeResponseTools() error = nil, want malformed namespace error")
			}
		})
	}
}

func TestResponseNamespaceMembersRoundTrip(t *testing.T) {
	_, _, registry, err := normalizeResponseTools([]json.RawMessage{responseNamespaceRawTool(t)})
	if err != nil {
		t.Fatalf("normalizeResponseTools() error = %v", err)
	}
	input := json.RawMessage(`[
		{"type":"function_call","call_id":"call_f","namespace":"github","name":"get_issue","arguments":"{\"number\":42}"},
		{"type":"function_call_output","call_id":"call_f","output":"issue"},
		{"type":"custom_tool_call","call_id":"call_c","namespace":"github","name":"query","input":"is:open"},
		{"type":"custom_tool_call_output","call_id":"call_c","output":"results"}
	]`)
	messages, err := normalizeResponseInputWithTools(input, registry)
	if err != nil {
		t.Fatalf("normalizeResponseInputWithTools() error = %v", err)
	}
	if len(messages) != 4 {
		t.Fatalf("messages len = %d, want 4", len(messages))
	}
	for index := 0; index < len(messages); index += 2 {
		call := messages[index]["tool_calls"].([]any)[0].(map[string]any)
		function := call["function"].(map[string]any)
		internalName := function["name"]
		if messages[index+1]["name"] != internalName || messages[index+1]["tool_call_id"] != call["id"] {
			t.Fatalf("namespace round trip pair %d mismatch: call=%#v output=%#v", index/2, messages[index], messages[index+1])
		}
		if index == 2 && function["arguments"] != `{"input":"is:open"}` {
			t.Fatalf("namespace custom input arguments = %q", function["arguments"])
		}
	}

	functionDefinition := registry.byTypeName[responseToolKey(responseToolFunction, "github", "get_issue")]
	customDefinition := registry.byTypeName[responseToolKey(responseToolCustom, "github", "query")]
	outputs, err := responseToolOutputs([]toolcall.ToolCall{
		{Name: functionDefinition.InternalName, Input: map[string]any{"number": 42}},
		{Name: customDefinition.InternalName, Input: map[string]any{"input": "is:open"}},
	}, []toolcall.ToolSchema{
		{Name: functionDefinition.InternalName, Parameters: map[string]any{"type": "object", "properties": map[string]any{"number": map[string]any{"type": "integer"}}}},
		{Name: customDefinition.InternalName, Parameters: responseCustomParameters()},
	}, registry.definitions)
	if err != nil {
		t.Fatalf("responseToolOutputs() error = %v", err)
	}
	if outputs[0]["type"] != "function_call" || outputs[0]["namespace"] != "github" || outputs[0]["name"] != "get_issue" {
		t.Fatalf("namespace function output = %#v", outputs[0])
	}
	if outputs[1]["type"] != "custom_tool_call" || outputs[1]["namespace"] != "github" || outputs[1]["name"] != "query" {
		t.Fatalf("namespace custom output = %#v", outputs[1])
	}
}

func TestResponseNamespaceStreamPreservesNamespace(t *testing.T) {
	chatTools, responseTools, registry, err := normalizeResponseTools([]json.RawMessage{responseNamespaceRawTool(t)})
	if err != nil {
		t.Fatalf("normalizeResponseTools() error = %v", err)
	}
	definition := registry.byTypeName[responseToolKey(responseToolFunction, "github", "get_issue")]
	content := fmt.Sprintf("<ml_tool_calls><ml_tool_call><ml_tool_name>%s</ml_tool_name><ml_parameters><number><![CDATA[42]]></number></ml_parameters></ml_tool_call></ml_tool_calls>", definition.InternalName)
	delta, _ := json.Marshal(map[string]any{"choices": []any{map[string]any{"delta": map[string]any{"content": content}}}})
	recorder := httptest.NewRecorder()
	responseTestHandler().handleResponseStream(recorder, strings.NewReader(fmt.Sprintf("data: %s\n\ndata: [DONE]\n\n", delta)), responseContext{
		ID: "resp_namespace", CreatedAt: 123, Model: "qwen3", Metadata: map[string]any{}, ParallelToolCalls: true,
		ToolChoice: "auto", Tools: responseTools, ToolDefinitions: registry.definitions,
	}, "qwen3", responseToolNames(registry.definitions), responseSchemas(chatTools), 1)
	events := decodeResponseEvents(t, recorder.Body.String())
	assertResponseEventTypes(t, events, []string{
		"response.created", "response.in_progress", "response.output_item.added", "response.function_call_arguments.delta",
		"response.function_call_arguments.done", "response.output_item.done", "response.completed",
	})
	done := events[5]["item"].(map[string]any)
	completed := events[6]["response"].(map[string]any)["output"].([]any)[0].(map[string]any)
	if done["namespace"] != "github" || done["name"] != "get_issue" || !reflect.DeepEqual(done, completed) {
		t.Fatalf("namespace stream output mismatch: done=%#v completed=%#v", done, completed)
	}
}

func TestForcedFunctionChoiceSurvivesIncrementalPreparation(t *testing.T) {
	tools := responseRawTools(t, map[string]any{
		"type": "function", "name": "weather", "description": "Get weather",
		"parameters": map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}, "required": []any{"city"}},
	})
	for name, toolChoice := range map[string]any{
		"required": "required",
		"explicit": map[string]any{"type": "function", "name": "weather"},
	} {
		t.Run(name, func(t *testing.T) {
			request := responsesRequest{Model: "qwen3", Input: json.RawMessage(`"weather in Hangzhou"`), Tools: tools, ToolChoice: toolChoice}
			normalized, err := normalizeResponsesRequest(request)
			if err != nil {
				t.Fatalf("normalizeResponsesRequest() error = %v", err)
			}
			logger := logging.New(false)
			accounts := account.NewService(config.Config{}, nil, nil, nil, logger)
			handler := &Handler{accounts: accounts, logger: logger}
			prepared := handler.prepareChatRequest(context.Background(), executedChatRequest{
				Model: "qwen3", Messages: normalized.Messages, Tools: normalized.ChatTools, ToolChoice: normalized.ChatToolChoice,
			})
			if len(prepared.LastUpstreamMessages) != 1 {
				t.Fatalf("last upstream messages len = %d, want 1", len(prepared.LastUpstreamMessages))
			}
			lastContent := extractText(prepared.LastUpstreamMessages[0]["content"])
			if !strings.Contains(lastContent, "[ml_tool reminder]") || !strings.Contains(lastContent, "weather") || !strings.Contains(lastContent, "must call") {
				t.Fatalf("forced tool reminder did not reach incremental message: %q", lastContent)
			}

			upstream := `{"choices":[{"message":{"role":"assistant","content":"<ml_tool_calls><ml_tool_call><ml_tool_name>weather</ml_tool_name><ml_parameters><city><![CDATA[Hangzhou]]></city></ml_parameters></ml_tool_call></ml_tool_calls>"}}]}`
			recorder := httptest.NewRecorder()
			handler.handleResponseNonStream(recorder, strings.NewReader(upstream), newResponseContext(request, normalized, "qwen3"),
				"qwen3", responseToolNames(normalized.ToolDefinitions), responseSchemas(normalized.ChatTools), 1)
			var response map[string]any
			decodeJSONForTest(t, recorder.Body.String(), &response)
			output := response["output"].([]any)
			if len(output) != 1 {
				t.Fatalf("output len = %d, want exactly one function call", len(output))
			}
			call := output[0].(map[string]any)
			if call["type"] != "function_call" || call["name"] != "weather" || call["arguments"] != `{"city":"Hangzhou"}` {
				t.Fatalf("forced function output = %#v", call)
			}
		})
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

func TestHandleResponsesRejectsHostedToolWithExplicitError(t *testing.T) {
	handler := &Handler{}
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{"model":"qwen3","input":"search","tools":[{"type":"web_search_preview"}]}`))

	handler.HandleResponses(recorder, request)

	if recorder.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadRequest)
	}
	var payload map[string]any
	decodeJSONForTest(t, recorder.Body.String(), &payload)
	errorObject := payload["error"].(map[string]any)
	if errorObject["type"] != "invalid_request_error" || errorObject["param"] != "tools" || !strings.Contains(stringValue(errorObject["message"]), "hosted execution") {
		t.Fatalf("hosted tool error envelope = %#v", errorObject)
	}
}

func responseTestHandler() *Handler {
	return &Handler{
		logger:  logging.New(false),
		metrics: metrics.NewDashboardStats(),
	}
}

func mixedResponseRawTools(t *testing.T) []json.RawMessage {
	t.Helper()
	return responseRawTools(t,
		map[string]any{
			"type":        "function",
			"name":        "weather",
			"description": "Get weather",
			"parameters":  map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}},
		},
		map[string]any{
			"type":        "custom",
			"name":        "freeform",
			"description": "Accept a word",
			"format":      map[string]any{"type": "grammar", "syntax": "regex", "definition": `^[a-z]+$`},
		},
		map[string]any{"type": "local_shell"},
		map[string]any{"type": "shell", "environment": map[string]any{"type": "local"}},
		map[string]any{"type": "apply_patch"},
	)
}

func responseNamespaceRawTool(t *testing.T) json.RawMessage {
	t.Helper()
	return responseRawTools(t, map[string]any{
		"type": "namespace", "name": "github", "description": "GitHub tools",
		"tools": []any{
			map[string]any{
				"type": "function", "name": "get_issue", "description": "Get an issue", "strict": false, "defer_loading": true,
				"parameters": map[string]any{"type": "object", "properties": map[string]any{"number": map[string]any{"type": "integer"}}, "required": []any{"number"}},
			},
			map[string]any{
				"type": "custom", "name": "query", "description": "Query issues", "defer_loading": false,
				"format": map[string]any{"type": "grammar", "syntax": "regex", "definition": `^.+$`},
			},
		},
	})[0]
}

func normalizeMixedResponseTools(t *testing.T) normalizedResponsesRequest {
	t.Helper()
	normalized, err := normalizeResponsesRequest(responsesRequest{
		Model: "qwen3",
		Input: json.RawMessage(`"use a tool"`),
		Tools: mixedResponseRawTools(t),
	})
	if err != nil {
		t.Fatalf("normalizeResponsesRequest() error = %v", err)
	}
	return normalized
}

func responseDefinition(t *testing.T, definitions []responseToolDefinition, toolType string) responseToolDefinition {
	t.Helper()
	for _, definition := range definitions {
		if definition.Type == toolType {
			return definition
		}
	}
	t.Fatalf("missing %s definition in %#v", toolType, definitions)
	return responseToolDefinition{}
}

func responseToolNames(definitions []responseToolDefinition) []string {
	result := make([]string, len(definitions))
	for index, definition := range definitions {
		result[index] = definition.InternalName
	}
	return result
}

func responseSchemas(chatTools []any) []toolcall.ToolSchema {
	result := make([]toolcall.ToolSchema, 0, len(chatTools))
	for _, raw := range chatTools {
		tool := raw.(map[string]any)
		function := tool["function"].(map[string]any)
		result = append(result, toolcall.ToolSchema{
			Name:        stringValue(function["name"]),
			Description: stringValue(function["description"]),
			Parameters:  function["parameters"].(map[string]any),
		})
	}
	return result
}

func registryFromDefinitions(definitions []responseToolDefinition) responseToolRegistry {
	registry := responseToolRegistry{
		definitions: append([]responseToolDefinition(nil), definitions...),
		byTypeName:  make(map[string]responseToolDefinition, len(definitions)),
	}
	for _, definition := range definitions {
		registry.byTypeName[responseToolKey(definition.Type, definition.Namespace, definition.Name)] = definition
	}
	return registry
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

func responseRawTools(t *testing.T, tools ...map[string]any) []json.RawMessage {
	t.Helper()
	result := make([]json.RawMessage, len(tools))
	for index, tool := range tools {
		raw, err := json.Marshal(tool)
		if err != nil {
			t.Fatalf("json.Marshal(tool) error = %v", err)
		}
		result[index] = raw
	}
	return result
}
