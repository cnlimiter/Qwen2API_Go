package openai

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"

	"qwen2api/internal/toolcall"
)

type responsesRequest struct {
	Model             string            `json:"model"`
	Input             json.RawMessage   `json:"input"`
	Instructions      *string           `json:"instructions"`
	Stream            bool              `json:"stream"`
	Tools             []json.RawMessage `json:"tools"`
	ToolChoice        any               `json:"tool_choice"`
	ParallelToolCalls *bool             `json:"parallel_tool_calls"`
	Metadata          map[string]any    `json:"metadata"`
	Reasoning         *chatReasoning    `json:"reasoning"`
}

const (
	responseToolFunction   = "function"
	responseToolCustom     = "custom"
	responseToolLocalShell = "local_shell"
	responseToolShell      = "shell"
	responseToolApplyPatch = "apply_patch"
	responseToolNamespace  = "namespace"
)

type responseToolDefinition struct {
	Type         string
	Name         string
	Namespace    string
	InternalName string
}

type responseToolRegistry struct {
	definitions []responseToolDefinition
	byTypeName  map[string]responseToolDefinition
}

type normalizedResponsesRequest struct {
	Messages        []map[string]any
	ChatTools       []any
	ChatToolChoice  any
	ResponseTools   []map[string]any
	ToolDefinitions []responseToolDefinition
	ToolChoice      any
	Parallel        bool
}

type responseContext struct {
	ID                string
	CreatedAt         int64
	Model             string
	Instructions      any
	Metadata          map[string]any
	ParallelToolCalls bool
	ToolChoice        any
	Tools             []map[string]any
	ToolDefinitions   []responseToolDefinition
}

type responseOutputBuilder struct {
	response       responseContext
	output         []map[string]any
	messageID      string
	messageIndex   int
	messageStarted bool
	messageText    strings.Builder
	toolCalls      []toolcall.ToolCall
	toolCount      int
}

type responseStreamWriter struct {
	w        http.ResponseWriter
	flusher  http.Flusher
	sequence int
}

// HandleResponses implements the OpenAI Responses API on top of the existing chat execution path.
func (h *Handler) HandleResponses(w http.ResponseWriter, r *http.Request) {
	var payload responsesRequest
	decoder := json.NewDecoder(r.Body)
	if err := decoder.Decode(&payload); err != nil {
		writeResponseError(w, http.StatusBadRequest, "Invalid JSON request body", "invalid_request_error", nil, "invalid_json")
		return
	}
	if err := ensureJSONEOF(decoder); err != nil {
		writeResponseError(w, http.StatusBadRequest, "Invalid JSON request body", "invalid_request_error", nil, "invalid_json")
		return
	}

	normalized, err := normalizeResponsesRequest(payload)
	if err != nil {
		writeResponseError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", responseErrorParam(err), "invalid_value")
		return
	}

	estimatedPromptTokens := estimateOpenAIInputTokens(normalized.Messages, normalized.ChatTools, normalized.ChatToolChoice)
	executed, status, err := h.executeChatRequest(r.Context(), executedChatRequest{
		Model:    payload.Model,
		Messages: normalized.Messages,
		ReasoningEffort: func() any {
			if payload.Reasoning == nil {
				return nil
			}
			return payload.Reasoning.Effort
		}(),
		Tools:      normalized.ChatTools,
		ToolChoice: normalized.ChatToolChoice,
	})
	if err != nil {
		writeResponseError(w, status, err.Error(), "api_error", nil, nil)
		return
	}
	defer executed.Stream.Close()

	response := newResponseContext(payload, normalized, executed.Model)
	statsModel := statsModelName(executed.RequestedModel, executed.Model)
	if payload.Stream {
		h.handleResponseStream(w, executed.Stream, response, statsModel, executed.ToolNames, executed.ToolSchemas, estimatedPromptTokens)
		return
	}
	h.handleResponseNonStream(w, executed.Stream, response, statsModel, executed.ToolNames, executed.ToolSchemas, estimatedPromptTokens)
}

func ensureJSONEOF(decoder *json.Decoder) error {
	var extra any
	err := decoder.Decode(&extra)
	if errors.Is(err, io.EOF) {
		return nil
	}
	if err == nil {
		return errors.New("multiple JSON values")
	}
	return err
}

type responseRequestError struct {
	param   string
	message string
}

func (e *responseRequestError) Error() string {
	return e.message
}

func responseInputError(message string) error {
	return &responseRequestError{param: "input", message: message}
}

func responseToolError(message string) error {
	return &responseRequestError{param: "tools", message: message}
}

func responseToolChoiceError(message string) error {
	return &responseRequestError{param: "tool_choice", message: message}
}

func responseErrorParam(err error) any {
	var requestErr *responseRequestError
	if errors.As(err, &requestErr) {
		return requestErr.param
	}
	return nil
}

func normalizeResponsesRequest(payload responsesRequest) (normalizedResponsesRequest, error) {
	if strings.TrimSpace(payload.Model) == "" {
		return normalizedResponsesRequest{}, &responseRequestError{param: "model", message: "model is required"}
	}
	chatTools, responseTools, registry, err := normalizeResponseTools(payload.Tools)
	if err != nil {
		return normalizedResponsesRequest{}, err
	}
	messages, err := normalizeResponseInputWithTools(payload.Input, registry)
	if err != nil {
		return normalizedResponsesRequest{}, err
	}
	if payload.Instructions != nil {
		messages = append([]map[string]any{{"role": "system", "content": *payload.Instructions}}, messages...)
	}

	chatToolChoice, responseToolChoice, err := normalizeResponseToolChoice(payload.ToolChoice, registry)
	if err != nil {
		return normalizedResponsesRequest{}, err
	}

	parallel := true
	if payload.ParallelToolCalls != nil {
		parallel = *payload.ParallelToolCalls
	}
	return normalizedResponsesRequest{
		Messages:        messages,
		ChatTools:       chatTools,
		ChatToolChoice:  chatToolChoice,
		ResponseTools:   responseTools,
		ToolDefinitions: registry.definitions,
		ToolChoice:      responseToolChoice,
		Parallel:        parallel,
	}, nil
}

func normalizeResponseInput(raw json.RawMessage) ([]map[string]any, error) {
	return normalizeResponseInputWithTools(raw, responseToolRegistry{})
}

func normalizeResponseInputWithTools(raw json.RawMessage, registry responseToolRegistry) ([]map[string]any, error) {
	trimmed := strings.TrimSpace(string(raw))
	if trimmed == "" || trimmed == "null" {
		return nil, responseInputError("input is required")
	}
	if strings.HasPrefix(trimmed, `"`) {
		var input string
		if err := json.Unmarshal(raw, &input); err != nil {
			return nil, responseInputError("input must be a string or an array of input items")
		}
		if strings.TrimSpace(input) == "" {
			return nil, responseInputError("input must not be empty")
		}
		return []map[string]any{{"role": "user", "content": input}}, nil
	}

	var items []map[string]any
	if err := json.Unmarshal(raw, &items); err != nil || len(items) == 0 {
		return nil, responseInputError("input must be a non-empty string or array of input items")
	}

	messages := make([]map[string]any, 0, len(items))
	calls := make(map[string]responseInputToolCall)
	outputs := make(map[string]struct{})
	for index, item := range items {
		itemType := strings.ToLower(strings.TrimSpace(stringValue(item["type"])))
		switch itemType {
		case "function_call":
			callID := strings.TrimSpace(stringValue(item["call_id"]))
			name := strings.TrimSpace(stringValue(item["name"]))
			namespace, err := optionalResponseString(item, "namespace")
			if err != nil {
				return nil, responseInputError(fmt.Sprintf("input[%d].namespace must be a string", index))
			}
			arguments, ok := item["arguments"].(string)
			var decodedArguments map[string]any
			if callID == "" || name == "" || !ok || json.Unmarshal([]byte(arguments), &decodedArguments) != nil {
				return nil, responseInputError(fmt.Sprintf("input[%d] has an invalid function_call", index))
			}
			if _, exists := calls[callID]; exists {
				return nil, responseInputError(fmt.Sprintf("input[%d].call_id is duplicated", index))
			}
			internalName := registry.internalName(responseToolFunction, namespace, name)
			if namespace != "" && internalName == "" {
				return nil, responseInputError(fmt.Sprintf("input[%d] does not reference a provided function namespace member", index))
			}
			if internalName == "" {
				internalName = name
			}
			calls[callID] = responseInputToolCall{Type: responseToolFunction, InternalName: internalName}
			messages = append(messages, responseAssistantToolMessage(callID, internalName, arguments))
		case "custom_tool_call":
			callID, callIDErr := requiredResponseString(item["call_id"], "call_id")
			name, nameErr := requiredResponseString(item["name"], "name")
			namespace, namespaceErr := optionalResponseString(item, "namespace")
			input, ok := item["input"].(string)
			if callIDErr != nil || nameErr != nil || namespaceErr != nil || !ok {
				return nil, responseInputError(fmt.Sprintf("input[%d] has an invalid custom_tool_call", index))
			}
			if _, exists := calls[callID]; exists {
				return nil, responseInputError(fmt.Sprintf("input[%d].call_id is duplicated", index))
			}
			internalName := registry.internalName(responseToolCustom, namespace, name)
			if internalName == "" {
				return nil, responseInputError(fmt.Sprintf("input[%d].name does not reference a provided custom tool", index))
			}
			calls[callID] = responseInputToolCall{Type: responseToolCustom, InternalName: internalName}
			arguments, err := json.Marshal(map[string]any{"input": input})
			if err != nil {
				return nil, responseInputError(fmt.Sprintf("input[%d] has an invalid custom_tool_call", index))
			}
			messages = append(messages, responseAssistantToolMessage(callID, internalName, string(arguments)))
		case "local_shell_call", "shell_call", "apply_patch_call":
			toolType := strings.TrimSuffix(itemType, "_call")
			call, arguments, err := normalizeResponseNativeInputCall(index, item, toolType, registry)
			if err != nil {
				return nil, err
			}
			if _, exists := calls[call.CallID]; exists {
				return nil, responseInputError(fmt.Sprintf("input[%d].call_id is duplicated", index))
			}
			calls[call.CallID] = call.responseInputToolCall
			messages = append(messages, responseAssistantToolMessage(call.CallID, call.InternalName, arguments))
		case "function_call_output", "custom_tool_call_output", "local_shell_call_output", "shell_call_output", "apply_patch_call_output":
			toolType := strings.TrimSuffix(itemType, "_call_output")
			if toolType == "custom_tool" {
				toolType = responseToolCustom
			}
			message, err := normalizeResponseToolOutput(index, item, toolType, calls, outputs)
			if err != nil {
				return nil, err
			}
			messages = append(messages, message)
		case "", "message":
			role := strings.ToLower(strings.TrimSpace(stringValue(item["role"])))
			switch role {
			case "user", "system", "developer", "assistant":
			default:
				return nil, responseInputError(fmt.Sprintf("input[%d].role is invalid", index))
			}
			content, err := normalizeResponseTextContent(item["content"])
			if err != nil {
				return nil, responseInputError(fmt.Sprintf("input[%d].content must contain text", index))
			}
			if role == "developer" {
				role = "system"
			}
			messages = append(messages, map[string]any{"role": role, "content": content})
		default:
			return nil, responseInputError(fmt.Sprintf("input[%d].type %q is not supported", index, itemType))
		}
	}
	return messages, nil
}

type responseInputToolCall struct {
	Type         string
	InternalName string
}

type responseNativeInputCall struct {
	responseInputToolCall
	CallID string
}

func responseAssistantToolMessage(callID, name, arguments string) map[string]any {
	return map[string]any{
		"role":    "assistant",
		"content": nil,
		"tool_calls": []any{map[string]any{
			"id":   callID,
			"type": "function",
			"function": map[string]any{
				"name":      name,
				"arguments": arguments,
			},
		}},
	}
}

func normalizeResponseNativeInputCall(index int, item map[string]any, toolType string, registry responseToolRegistry) (responseNativeInputCall, string, error) {
	callID, err := requiredResponseString(item["call_id"], "call_id")
	if err != nil {
		return responseNativeInputCall{}, "", responseInputError(fmt.Sprintf("input[%d].call_id is required", index))
	}
	internalName := registry.internalName(toolType, "", toolType)
	if internalName == "" {
		return responseNativeInputCall{}, "", responseInputError(fmt.Sprintf("input[%d] references a %s tool that was not provided", index, toolType))
	}

	var arguments map[string]any
	switch toolType {
	case responseToolLocalShell:
		arguments, err = normalizeResponseLocalShellAction(item["action"])
	case responseToolShell:
		arguments, err = normalizeResponseShellAction(item["action"])
	case responseToolApplyPatch:
		arguments, err = normalizeResponseApplyPatchOperation(item["operation"])
	default:
		err = fmt.Errorf("unsupported native tool type %q", toolType)
	}
	if err != nil {
		return responseNativeInputCall{}, "", responseInputError(fmt.Sprintf("input[%d] has an invalid %s_call: %v", index, toolType, err))
	}
	rawArguments, err := json.Marshal(arguments)
	if err != nil {
		return responseNativeInputCall{}, "", responseInputError(fmt.Sprintf("input[%d] has an invalid %s_call", index, toolType))
	}
	return responseNativeInputCall{
		responseInputToolCall: responseInputToolCall{Type: toolType, InternalName: internalName},
		CallID:                callID,
	}, string(rawArguments), nil
}

func normalizeResponseToolOutput(index int, item map[string]any, expectedType string, calls map[string]responseInputToolCall, outputs map[string]struct{}) (map[string]any, error) {
	callID, err := requiredResponseString(item["call_id"], "call_id")
	if err != nil {
		return nil, responseInputError(fmt.Sprintf("input[%d].call_id is required", index))
	}
	call, exists := calls[callID]
	if !exists {
		return nil, responseInputError(fmt.Sprintf("input[%d].call_id does not match a previous %s_call", index, expectedType))
	}
	if call.Type != expectedType {
		return nil, responseInputError(fmt.Sprintf("input[%d].call_id references %s_call, not %s_call", index, call.Type, expectedType))
	}
	if _, exists := outputs[callID]; exists {
		return nil, responseInputError(fmt.Sprintf("input[%d].call_id output is duplicated", index))
	}

	var content any
	switch expectedType {
	case responseToolFunction, responseToolCustom:
		content, err = normalizeResponseTextContent(item["output"])
	case responseToolLocalShell:
		content, err = requiredResponseString(item["output"], "output")
	case responseToolShell:
		content, err = normalizeResponseShellOutput(item["output"])
	case responseToolApplyPatch:
		content, err = normalizeResponseApplyPatchOutput(item)
	}
	if err != nil {
		return nil, responseInputError(fmt.Sprintf("input[%d] has an invalid %s_call_output: %v", index, expectedType, err))
	}
	outputs[callID] = struct{}{}
	return map[string]any{
		"role":         "tool",
		"tool_call_id": callID,
		"name":         call.InternalName,
		"content":      content,
	}, nil
}

func normalizeResponseLocalShellAction(raw any) (map[string]any, error) {
	action, ok := raw.(map[string]any)
	if !ok {
		return nil, errors.New("action must be an object")
	}
	if strings.TrimSpace(stringValue(action["type"])) != "exec" {
		return nil, errors.New("action.type must be exec")
	}
	if err := validateResponseObjectKeys(action, "type", "command", "timeout_ms", "working_directory", "env", "user"); err != nil {
		return nil, err
	}
	command, err := responseStringArray(action["command"], "action.command")
	if err != nil {
		return nil, err
	}
	result := map[string]any{"command": command}
	if value, exists := action["timeout_ms"]; exists {
		result["timeout_ms"], err = responsePositiveInteger(value, "action.timeout_ms")
		if err != nil {
			return nil, err
		}
	}
	for _, field := range []string{"working_directory", "user"} {
		if value, exists := action[field]; exists {
			result[field], err = requiredResponseString(value, "action."+field)
			if err != nil {
				return nil, err
			}
		}
	}
	if value, exists := action["env"]; exists {
		env, ok := value.(map[string]any)
		if !ok {
			return nil, errors.New("action.env must be an object of strings")
		}
		normalized := make(map[string]any, len(env))
		for key, rawValue := range env {
			stringValue, ok := rawValue.(string)
			if strings.TrimSpace(key) == "" || !ok {
				return nil, errors.New("action.env must be an object of strings")
			}
			normalized[key] = stringValue
		}
		result["env"] = normalized
	}
	return result, nil
}

func normalizeResponseShellAction(raw any) (map[string]any, error) {
	action, ok := raw.(map[string]any)
	if !ok {
		return nil, errors.New("action must be an object")
	}
	if err := validateResponseObjectKeys(action, "commands", "timeout_ms", "max_output_length"); err != nil {
		return nil, err
	}
	commands, err := responseStringArray(action["commands"], "action.commands")
	if err != nil {
		return nil, err
	}
	result := map[string]any{"commands": commands}
	for _, field := range []string{"timeout_ms", "max_output_length"} {
		if value, exists := action[field]; exists {
			result[field], err = responsePositiveInteger(value, "action."+field)
			if err != nil {
				return nil, err
			}
		}
	}
	return result, nil
}

func normalizeResponseApplyPatchOperation(raw any) (map[string]any, error) {
	operation, ok := raw.(map[string]any)
	if !ok {
		return nil, errors.New("operation must be an object")
	}
	if err := validateResponseObjectKeys(operation, "type", "path", "diff"); err != nil {
		return nil, err
	}
	operationType := strings.TrimSpace(stringValue(operation["type"]))
	if operationType != "create_file" && operationType != "update_file" && operationType != "delete_file" {
		return nil, errors.New("operation.type must be create_file, update_file, or delete_file")
	}
	path, err := requiredResponseString(operation["path"], "operation.path")
	if err != nil {
		return nil, err
	}
	result := map[string]any{"operation": operationType, "path": path}
	if operationType != "delete_file" {
		diff, err := requiredResponseString(operation["diff"], "operation.diff")
		if err != nil {
			return nil, err
		}
		result["diff"] = diff
	} else if _, exists := operation["diff"]; exists {
		return nil, errors.New("operation.diff is not valid for delete_file")
	}
	return result, nil
}

func normalizeResponseShellOutput(raw any) (string, error) {
	items, ok := raw.([]any)
	if !ok {
		return "", errors.New("output must be an array")
	}
	for index, rawItem := range items {
		item, ok := rawItem.(map[string]any)
		if !ok {
			return "", fmt.Errorf("output[%d] must be an object", index)
		}
		if err := validateResponseObjectKeys(item, "stdout", "stderr", "outcome"); err != nil {
			return "", fmt.Errorf("output[%d]: %w", index, err)
		}
		if _, err := requiredResponseStringAllowEmpty(item["stdout"], "stdout"); err != nil {
			return "", fmt.Errorf("output[%d]: %w", index, err)
		}
		if _, err := requiredResponseStringAllowEmpty(item["stderr"], "stderr"); err != nil {
			return "", fmt.Errorf("output[%d]: %w", index, err)
		}
		outcome, ok := item["outcome"].(map[string]any)
		if !ok {
			return "", fmt.Errorf("output[%d].outcome must be an object", index)
		}
		switch strings.TrimSpace(stringValue(outcome["type"])) {
		case "timeout":
			if err := validateResponseObjectKeys(outcome, "type"); err != nil {
				return "", fmt.Errorf("output[%d].outcome: %w", index, err)
			}
		case "exit":
			if err := validateResponseObjectKeys(outcome, "type", "exit_code"); err != nil {
				return "", fmt.Errorf("output[%d].outcome: %w", index, err)
			}
			if _, err := responseInteger(outcome["exit_code"], "exit_code"); err != nil {
				return "", fmt.Errorf("output[%d].outcome: %w", index, err)
			}
		default:
			return "", fmt.Errorf("output[%d].outcome.type must be exit or timeout", index)
		}
	}
	encoded, err := json.Marshal(items)
	if err != nil {
		return "", errors.New("output is not JSON serializable")
	}
	return string(encoded), nil
}

func normalizeResponseApplyPatchOutput(item map[string]any) (string, error) {
	status := strings.TrimSpace(stringValue(item["status"]))
	if status != "completed" && status != "failed" {
		return "", errors.New("status must be completed or failed")
	}
	output := ""
	if value, exists := item["output"]; exists {
		var err error
		output, err = requiredResponseStringAllowEmpty(value, "output")
		if err != nil {
			return "", err
		}
	}
	encoded, err := json.Marshal(map[string]any{"status": status, "output": output})
	if err != nil {
		return "", errors.New("output is not JSON serializable")
	}
	return string(encoded), nil
}

func responseStringArray(raw any, field string) ([]any, error) {
	items, ok := raw.([]any)
	if !ok || len(items) == 0 {
		return nil, fmt.Errorf("%s must be a non-empty array of strings", field)
	}
	result := make([]any, len(items))
	for index, rawItem := range items {
		item, ok := rawItem.(string)
		if !ok || strings.TrimSpace(item) == "" {
			return nil, fmt.Errorf("%s[%d] must be a non-empty string", field, index)
		}
		result[index] = item
	}
	return result, nil
}

func responseInteger(raw any, field string) (int, error) {
	value, ok := raw.(float64)
	if !ok || math.Trunc(value) != value || value < math.MinInt32 || value > math.MaxInt32 {
		return 0, fmt.Errorf("%s must be an integer", field)
	}
	return int(value), nil
}

func responsePositiveInteger(raw any, field string) (int, error) {
	value, err := responseInteger(raw, field)
	if err != nil || value <= 0 {
		return 0, fmt.Errorf("%s must be a positive integer", field)
	}
	return value, nil
}

func requiredResponseString(raw any, field string) (string, error) {
	value, err := requiredResponseStringAllowEmpty(raw, field)
	if err != nil || strings.TrimSpace(value) == "" {
		return "", fmt.Errorf("%s must be a non-empty string", field)
	}
	return value, nil
}

func requiredResponseStringAllowEmpty(raw any, field string) (string, error) {
	value, ok := raw.(string)
	if !ok {
		return "", fmt.Errorf("%s must be a string", field)
	}
	return value, nil
}

func optionalResponseString(object map[string]any, field string) (string, error) {
	raw, exists := object[field]
	if !exists || raw == nil {
		return "", nil
	}
	value, ok := raw.(string)
	if !ok {
		return "", fmt.Errorf("%s must be a string", field)
	}
	return strings.TrimSpace(value), nil
}

func validateResponseObjectKeys(object map[string]any, allowed ...string) error {
	allowedKeys := make(map[string]struct{}, len(allowed))
	for _, key := range allowed {
		allowedKeys[key] = struct{}{}
	}
	for key := range object {
		if _, ok := allowedKeys[key]; !ok {
			return fmt.Errorf("field %q is not supported", key)
		}
	}
	return nil
}

func normalizeResponseTextContent(raw any) (any, error) {
	switch value := raw.(type) {
	case string:
		return value, nil
	case []any:
		content := make([]any, 0, len(value))
		for _, rawPart := range value {
			part, ok := rawPart.(map[string]any)
			if !ok {
				return nil, errors.New("invalid content part")
			}
			partType := strings.ToLower(strings.TrimSpace(stringValue(part["type"])))
			switch partType {
			case "input_text", "output_text", "text":
				text, ok := part["text"].(string)
				if !ok {
					return nil, errors.New("text is required")
				}
				content = append(content, map[string]any{"type": "text", "text": text})
			default:
				return nil, fmt.Errorf("unsupported content type %q", partType)
			}
		}
		if len(content) == 0 {
			return nil, errors.New("content must not be empty")
		}
		return content, nil
	default:
		return nil, errors.New("content must be text")
	}
}

func normalizeResponseTools(tools []json.RawMessage) ([]any, []map[string]any, responseToolRegistry, error) {
	registry := responseToolRegistry{
		byTypeName: make(map[string]responseToolDefinition),
	}
	parsed := make([]map[string]any, len(tools))
	usedInternalNames := make(map[string]struct{}, len(tools))
	for index, raw := range tools {
		var tool map[string]any
		if err := json.Unmarshal(raw, &tool); err != nil || tool == nil {
			return nil, nil, responseToolRegistry{}, responseToolError(fmt.Sprintf("tools[%d] must be an object", index))
		}
		parsed[index] = tool
		rawType, _ := tool["type"].(string)
		if strings.ToLower(strings.TrimSpace(rawType)) == responseToolFunction {
			name, _ := tool["name"].(string)
			name = strings.TrimSpace(name)
			if name != "" {
				usedInternalNames[name] = struct{}{}
			}
		}
	}

	chatTools := make([]any, 0, len(parsed))
	responseTools := make([]map[string]any, 0, len(parsed))
	for index, tool := range parsed {
		rawType, ok := tool["type"].(string)
		toolType := strings.ToLower(strings.TrimSpace(rawType))
		if !ok || toolType == "" {
			return nil, nil, responseToolRegistry{}, responseToolError(fmt.Sprintf("tools[%d].type must be a non-empty string", index))
		}
		if isHostedResponseTool(toolType) {
			return nil, nil, responseToolRegistry{}, responseToolError(fmt.Sprintf("tools[%d].type %q requires OpenAI-hosted execution, which this backend cannot provide", index, toolType))
		}
		if toolType == responseToolNamespace {
			namespaceTools, err := normalizeResponseNamespace(tool, index, usedInternalNames, &registry)
			if err != nil {
				return nil, nil, responseToolRegistry{}, err
			}
			chatTools = append(chatTools, namespaceTools...)
			echo := cloneMap(tool)
			echo["type"] = toolType
			responseTools = append(responseTools, echo)
			continue
		}

		name := toolType
		internalName := ""
		var description string
		var parameters map[string]any
		var err error
		switch toolType {
		case responseToolFunction:
			name, err = requiredResponseString(tool["name"], "name")
			if err != nil {
				return nil, nil, responseToolRegistry{}, responseToolError(fmt.Sprintf("tools[%d].name must be a non-empty string", index))
			}
			parameters, _ = tool["parameters"].(map[string]any)
			if parameters == nil {
				return nil, nil, responseToolRegistry{}, responseToolError(fmt.Sprintf("tools[%d].parameters is required", index))
			}
			if rawDescription, exists := tool["description"]; exists {
				description, ok = rawDescription.(string)
				if !ok {
					return nil, nil, responseToolRegistry{}, responseToolError(fmt.Sprintf("tools[%d].description must be a string", index))
				}
			}
			if rawStrict, exists := tool["strict"]; exists {
				if _, ok := rawStrict.(bool); !ok {
					return nil, nil, responseToolRegistry{}, responseToolError(fmt.Sprintf("tools[%d].strict must be a boolean", index))
				}
			}
			internalName = name
		case responseToolCustom:
			name, err = requiredResponseString(tool["name"], "name")
			if err != nil {
				return nil, nil, responseToolRegistry{}, responseToolError(fmt.Sprintf("tools[%d].name must be a non-empty string", index))
			}
			if rawDescription, exists := tool["description"]; exists {
				description, ok = rawDescription.(string)
				if !ok {
					return nil, nil, responseToolRegistry{}, responseToolError(fmt.Sprintf("tools[%d].description must be a string", index))
				}
			}
			formatDescription, err := validateResponseCustomFormat(tool["format"])
			if err != nil {
				return nil, nil, responseToolRegistry{}, responseToolError(fmt.Sprintf("tools[%d].format %v", index, err))
			}
			description = strings.TrimSpace(strings.Join([]string{description, formatDescription}, "\n"))
			parameters = responseCustomParameters()
			internalName = nextResponseInternalName(toolType, index, usedInternalNames)
		case responseToolLocalShell:
			description = "Execute a local command. Supply command as a JSON array of executable and arguments; optional timeout_ms, working_directory, env, and user must use their documented types."
			parameters = responseLocalShellParameters()
			internalName = nextResponseInternalName(toolType, index, usedInternalNames)
		case responseToolShell:
			description = "Execute ordered shell commands. Supply commands as a JSON array of strings and optional positive timeout_ms and max_output_length integers."
			parameters = responseShellParameters()
			internalName = nextResponseInternalName(toolType, index, usedInternalNames)
		case responseToolApplyPatch:
			description = "Apply one file patch operation. Supply operation as create_file, update_file, or delete_file; path is required, and diff is required except for delete_file."
			parameters = responseApplyPatchParameters()
			internalName = nextResponseInternalName(toolType, index, usedInternalNames)
		default:
			return nil, nil, responseToolRegistry{}, responseToolError(fmt.Sprintf("tools[%d].type %q is not supported", index, toolType))
		}

		key := responseToolKey(toolType, "", name)
		if _, exists := registry.byTypeName[key]; exists {
			return nil, nil, responseToolRegistry{}, responseToolError(fmt.Sprintf("tools[%d] duplicates %s tool %q", index, toolType, name))
		}
		definition := responseToolDefinition{Type: toolType, Name: name, InternalName: internalName}
		registry.definitions = append(registry.definitions, definition)
		registry.byTypeName[key] = definition
		usedInternalNames[internalName] = struct{}{}

		chatTools = append(chatTools, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        internalName,
				"description": description,
				"parameters":  parameters,
			},
		})
		echo := cloneMap(tool)
		echo["type"] = toolType
		responseTools = append(responseTools, echo)
	}
	return chatTools, responseTools, registry, nil
}

func normalizeResponseNamespace(tool map[string]any, toolIndex int, usedInternalNames map[string]struct{}, registry *responseToolRegistry) ([]any, error) {
	if err := validateResponseObjectKeys(tool, "type", "name", "description", "tools"); err != nil {
		return nil, responseToolError(fmt.Sprintf("tools[%d] %v", toolIndex, err))
	}
	namespace, err := requiredResponseString(tool["name"], "name")
	if err != nil {
		return nil, responseToolError(fmt.Sprintf("tools[%d].name must be a non-empty string", toolIndex))
	}
	description, err := requiredResponseStringAllowEmpty(tool["description"], "description")
	if err != nil {
		return nil, responseToolError(fmt.Sprintf("tools[%d].description must be a string", toolIndex))
	}
	rawMembers, ok := tool["tools"].([]any)
	if !ok || len(rawMembers) == 0 {
		return nil, responseToolError(fmt.Sprintf("tools[%d].tools must be a non-empty array", toolIndex))
	}

	chatTools := make([]any, 0, len(rawMembers))
	for memberIndex, rawMember := range rawMembers {
		member, ok := rawMember.(map[string]any)
		if !ok {
			return nil, responseToolError(fmt.Sprintf("tools[%d].tools[%d] must be an object", toolIndex, memberIndex))
		}
		rawType, ok := member["type"].(string)
		memberType := strings.ToLower(strings.TrimSpace(rawType))
		if !ok || (memberType != responseToolFunction && memberType != responseToolCustom) {
			return nil, responseToolError(fmt.Sprintf("tools[%d].tools[%d].type must be function or custom", toolIndex, memberIndex))
		}
		name, err := requiredResponseString(member["name"], "name")
		if err != nil {
			return nil, responseToolError(fmt.Sprintf("tools[%d].tools[%d].name must be a non-empty string", toolIndex, memberIndex))
		}
		key := responseToolKey(memberType, namespace, name)
		if _, exists := registry.byTypeName[key]; exists {
			return nil, responseToolError(fmt.Sprintf("tools[%d].tools[%d] duplicates %s member %q in namespace %q", toolIndex, memberIndex, memberType, name, namespace))
		}

		memberDescription := ""
		if rawDescription, exists := member["description"]; exists {
			memberDescription, ok = rawDescription.(string)
			if !ok {
				return nil, responseToolError(fmt.Sprintf("tools[%d].tools[%d].description must be a string", toolIndex, memberIndex))
			}
		}
		var parameters map[string]any
		switch memberType {
		case responseToolFunction:
			if err := validateResponseNamespaceFunction(member); err != nil {
				return nil, responseToolError(fmt.Sprintf("tools[%d].tools[%d] %v", toolIndex, memberIndex, err))
			}
			parameters, _ = member["parameters"].(map[string]any)
		case responseToolCustom:
			formatDescription, err := validateResponseNamespaceCustom(member)
			if err != nil {
				return nil, responseToolError(fmt.Sprintf("tools[%d].tools[%d] %v", toolIndex, memberIndex, err))
			}
			memberDescription = strings.TrimSpace(strings.Join([]string{memberDescription, formatDescription}, "\n"))
			parameters = responseCustomParameters()
		}

		internalName := nextResponseInternalName(fmt.Sprintf("namespace_%d_%s", toolIndex, memberType), memberIndex, usedInternalNames)
		definition := responseToolDefinition{Type: memberType, Name: name, Namespace: namespace, InternalName: internalName}
		registry.definitions = append(registry.definitions, definition)
		registry.byTypeName[key] = definition
		usedInternalNames[internalName] = struct{}{}
		chatTools = append(chatTools, map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":        internalName,
				"description": fmt.Sprintf("Namespace %s: %s\n%s", namespace, description, memberDescription),
				"parameters":  parameters,
			},
		})
	}
	return chatTools, nil
}

func validateResponseNamespaceFunction(member map[string]any) error {
	if err := validateResponseObjectKeys(member, "type", "name", "description", "strict", "defer_loading", "parameters", "output_schema", "allowed_callers"); err != nil {
		return err
	}
	parameters, ok := member["parameters"].(map[string]any)
	if !ok || parameters == nil {
		return errors.New("parameters must be an object")
	}
	for _, field := range []string{"strict", "defer_loading"} {
		if raw, exists := member[field]; exists {
			if _, ok := raw.(bool); !ok {
				return fmt.Errorf("%s must be a boolean", field)
			}
		}
	}
	if raw, exists := member["output_schema"]; exists {
		if _, ok := raw.(map[string]any); !ok {
			return errors.New("output_schema must be an object")
		}
	}
	return validateResponseAllowedCallers(member["allowed_callers"])
}

func validateResponseNamespaceCustom(member map[string]any) (string, error) {
	if err := validateResponseObjectKeys(member, "type", "name", "description", "format", "defer_loading", "allowed_callers"); err != nil {
		return "", err
	}
	if raw, exists := member["defer_loading"]; exists {
		if _, ok := raw.(bool); !ok {
			return "", errors.New("defer_loading must be a boolean")
		}
	}
	formatDescription, err := validateResponseCustomFormat(member["format"])
	if err != nil {
		return "", fmt.Errorf("format %v", err)
	}
	if err := validateResponseAllowedCallers(member["allowed_callers"]); err != nil {
		return "", err
	}
	return formatDescription, nil
}

func validateResponseAllowedCallers(raw any) error {
	if raw == nil {
		return nil
	}
	callers, ok := raw.([]any)
	if !ok || len(callers) == 0 {
		return errors.New("allowed_callers must be a non-empty array")
	}
	for _, rawCaller := range callers {
		caller, ok := rawCaller.(string)
		if !ok || (caller != "direct" && caller != "programmatic") {
			return errors.New("allowed_callers entries must be direct or programmatic")
		}
	}
	return nil
}

func normalizeResponseToolChoice(raw any, registry responseToolRegistry) (any, any, error) {
	if raw == nil {
		return "auto", "auto", nil
	}
	switch value := raw.(type) {
	case string:
		choice := strings.ToLower(strings.TrimSpace(value))
		switch choice {
		case "none", "auto":
			return choice, choice, nil
		case "required":
			if len(registry.definitions) == 0 {
				return nil, nil, responseToolChoiceError("tool_choice required needs at least one tool")
			}
			return choice, choice, nil
		default:
			return nil, nil, responseToolChoiceError("tool_choice must be none, auto, required, or a supported tool selector")
		}
	case map[string]any:
		rawType, ok := value["type"].(string)
		toolType := strings.ToLower(strings.TrimSpace(rawType))
		if !ok || toolType == "" {
			return nil, nil, responseToolChoiceError("tool_choice.type must be a non-empty string")
		}
		if isHostedResponseTool(toolType) {
			return nil, nil, responseToolChoiceError(fmt.Sprintf("tool_choice.type %q requires OpenAI-hosted execution, which this backend cannot provide", toolType))
		}
		name := toolType
		if toolType == responseToolFunction || toolType == responseToolCustom {
			if err := validateResponseObjectKeys(value, "type", "name"); err != nil {
				return nil, nil, responseToolChoiceError(fmt.Sprintf("tool_choice %v", err))
			}
			var err error
			name, err = requiredResponseString(value["name"], "name")
			if err != nil {
				return nil, nil, responseToolChoiceError("tool_choice.name is required for function and custom tools")
			}
		} else if err := validateResponseObjectKeys(value, "type"); err != nil {
			return nil, nil, responseToolChoiceError(fmt.Sprintf("tool_choice %v", err))
		}
		definition, exists := registry.byTypeName[responseToolKey(toolType, "", name)]
		if !exists {
			return nil, nil, responseToolChoiceError(fmt.Sprintf("tool_choice does not reference a provided %s tool", toolType))
		}
		responseChoice := map[string]any{"type": toolType}
		if toolType == responseToolFunction || toolType == responseToolCustom {
			responseChoice["name"] = name
		}
		return map[string]any{
			"type":     "function",
			"function": map[string]any{"name": definition.InternalName},
		}, responseChoice, nil
	default:
		return nil, nil, responseToolChoiceError("tool_choice has an invalid type")
	}
}

func (registry responseToolRegistry) internalName(toolType, namespace, name string) string {
	return registry.byTypeName[responseToolKey(toolType, namespace, name)].InternalName
}

func responseToolKey(toolType, namespace, name string) string {
	return toolType + "\x00" + namespace + "\x00" + name
}

func nextResponseInternalName(toolType string, index int, used map[string]struct{}) string {
	base := fmt.Sprintf("__responses_%s_%d", toolType, index)
	name := base
	for suffix := 1; ; suffix++ {
		if _, exists := used[name]; !exists {
			return name
		}
		name = fmt.Sprintf("%s_%d", base, suffix)
	}
}

func validateResponseCustomFormat(raw any) (string, error) {
	if raw == nil {
		return "Input format: unconstrained text.", nil
	}
	format, ok := raw.(map[string]any)
	if !ok {
		return "", errors.New("must be an object")
	}
	formatType := strings.ToLower(strings.TrimSpace(stringValue(format["type"])))
	switch formatType {
	case "text":
		if err := validateResponseObjectKeys(format, "type"); err != nil {
			return "", err
		}
		return "Input format: unconstrained text.", nil
	case "grammar":
		if err := validateResponseObjectKeys(format, "type", "syntax", "definition"); err != nil {
			return "", err
		}
		syntax := strings.ToLower(strings.TrimSpace(stringValue(format["syntax"])))
		if syntax != "lark" && syntax != "regex" {
			return "", errors.New("syntax must be lark or regex")
		}
		definition, err := requiredResponseString(format["definition"], "definition")
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("Input must conform to this %s grammar:\n%s", syntax, definition), nil
	default:
		return "", errors.New("type must be text or grammar")
	}
}

func responseCustomParameters() map[string]any {
	return map[string]any{
		"type":                 "object",
		"properties":           map[string]any{"input": map[string]any{"type": "string", "description": "Complete free-form tool input."}},
		"required":             []any{"input"},
		"additionalProperties": false,
	}
}

func responseLocalShellParameters() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"command":           map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
			"timeout_ms":        map[string]any{"type": "integer", "minimum": 1},
			"working_directory": map[string]any{"type": "string"},
			"env":               map[string]any{"type": "object", "additionalProperties": map[string]any{"type": "string"}},
			"user":              map[string]any{"type": "string"},
		},
		"required":             []any{"command"},
		"additionalProperties": false,
	}
}

func responseShellParameters() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"commands":          map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
			"timeout_ms":        map[string]any{"type": "integer", "minimum": 1},
			"max_output_length": map[string]any{"type": "integer", "minimum": 1},
		},
		"required":             []any{"commands"},
		"additionalProperties": false,
	}
}

func responseApplyPatchParameters() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"operation": map[string]any{"type": "string", "enum": []any{"create_file", "update_file", "delete_file"}},
			"path":      map[string]any{"type": "string"},
			"diff":      map[string]any{"type": "string"},
		},
		"required":             []any{"operation", "path"},
		"additionalProperties": false,
	}
}

func isHostedResponseTool(toolType string) bool {
	switch toolType {
	case "file_search", "code_interpreter", "image_generation", "mcp", "tool_search", "programmatic_tool_calling":
		return true
	}
	return strings.HasPrefix(toolType, "web_search") || strings.HasPrefix(toolType, "computer")
}

func newResponseContext(payload responsesRequest, normalized normalizedResponsesRequest, model string) responseContext {
	metadata := payload.Metadata
	if metadata == nil {
		metadata = map[string]any{}
	}
	return responseContext{
		ID:                fmt.Sprintf("resp_%d", time.Now().UnixNano()),
		CreatedAt:         time.Now().Unix(),
		Model:             model,
		Instructions:      optionalStringValue(payload.Instructions),
		Metadata:          metadata,
		ParallelToolCalls: normalized.Parallel,
		ToolChoice:        normalized.ToolChoice,
		Tools:             normalized.ResponseTools,
		ToolDefinitions:   normalized.ToolDefinitions,
	}
}

func optionalStringValue(value *string) any {
	if value == nil {
		return nil
	}
	return *value
}

func (response responseContext) payload(status string, output []map[string]any, usage any, responseError any) map[string]any {
	return map[string]any{
		"id":                  response.ID,
		"object":              "response",
		"created_at":          response.CreatedAt,
		"status":              status,
		"error":               responseError,
		"incomplete_details":  nil,
		"instructions":        response.Instructions,
		"metadata":            response.Metadata,
		"model":               response.Model,
		"output":              output,
		"parallel_tool_calls": response.ParallelToolCalls,
		"tool_choice":         response.ToolChoice,
		"tools":               response.Tools,
		"usage":               usage,
	}
}

func (h *Handler) handleResponseNonStream(w http.ResponseWriter, body io.Reader, response responseContext, statsModel string, toolNames []string, toolSchemas []toolcall.ToolSchema, estimatedPromptTokens int) {
	result, upstreamErr, err := h.readCompletedChat(body, response.Model, toolNames)
	if err != nil {
		writeResponseError(w, http.StatusBadGateway, "Failed to read upstream response", "api_error", nil, nil)
		return
	}
	if upstreamErr != nil {
		writeResponseError(w, normalizeUpstreamStatus(upstreamErr.StatusCode), upstreamErr.Error(), "api_error", nil, nil)
		return
	}

	if !response.ParallelToolCalls && len(result.ToolCalls) > 1 {
		result.ToolCalls = result.ToolCalls[:1]
	}
	output := make([]map[string]any, 0, 1+len(result.ToolCalls))
	if result.Content != "" || len(result.ToolCalls) == 0 {
		output = append(output, responseMessageOutput(fmt.Sprintf("msg_%d", time.Now().UnixNano()), result.Content, "completed"))
	}
	toolOutputs, err := responseToolOutputs(result.ToolCalls, toolSchemas, response.ToolDefinitions)
	if err != nil {
		writeResponseError(w, http.StatusBadGateway, fmt.Sprintf("Failed to format upstream tool call: %v", err), "api_error", nil, nil)
		return
	}
	output = append(output, toolOutputs...)
	result.PromptTokens, result.CompletionTokens, result.TotalTokens = applyUsageFallback(
		result.PromptTokens,
		result.CompletionTokens,
		result.TotalTokens,
		estimatedPromptTokens,
		estimateOpenAIOutputTokens(result.Content, result.ToolCalls),
	)
	usage := responseUsage(result.PromptTokens, result.CompletionTokens, result.TotalTokens)
	if h.metrics != nil {
		h.metrics.RecordModelUsage(statsModel, result.PromptTokens, result.CompletionTokens, result.TotalTokens)
	}
	writeJSON(w, http.StatusOK, response.payload("completed", output, usage, nil))
}

func responseMessageOutput(id, text, status string) map[string]any {
	return map[string]any{
		"id":     id,
		"type":   "message",
		"status": status,
		"role":   "assistant",
		"content": []map[string]any{{
			"type":        "output_text",
			"text":        text,
			"annotations": []any{},
		}},
	}
}

func responseToolOutputs(calls []toolcall.ToolCall, schemas []toolcall.ToolSchema, definitions []responseToolDefinition) ([]map[string]any, error) {
	definitionByInternal := make(map[string]responseToolDefinition, len(definitions))
	for _, definition := range definitions {
		definitionByInternal[definition.InternalName] = definition
	}
	formatted := toolcall.FormatOpenAIToolCallsWithSchemas(calls, schemas)
	output := make([]map[string]any, 0, len(formatted))
	for index, call := range formatted {
		function, _ := call["function"].(map[string]any)
		internalName := strings.TrimSpace(stringValue(function["name"]))
		definition, exists := definitionByInternal[internalName]
		if !exists {
			return nil, fmt.Errorf("tool %q is not in the Responses tool registry", internalName)
		}
		callID := strings.TrimSpace(stringValue(call["id"]))
		if callID == "" {
			callID = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), index)
		}
		arguments := stringValue(function["arguments"])
		var decoded map[string]any
		if err := json.Unmarshal([]byte(arguments), &decoded); err != nil {
			return nil, fmt.Errorf("tool %q arguments are invalid JSON: %w", definition.Name, err)
		}
		item, err := responseNativeToolOutput(definition, callID, arguments, decoded, index)
		if err != nil {
			return nil, err
		}
		output = append(output, item)
	}
	return output, nil
}

func responseNativeToolOutput(definition responseToolDefinition, callID, arguments string, decoded map[string]any, index int) (map[string]any, error) {
	idPrefix := map[string]string{
		responseToolFunction:   "fc",
		responseToolCustom:     "ctc",
		responseToolLocalShell: "lsc",
		responseToolShell:      "sc",
		responseToolApplyPatch: "apc",
	}[definition.Type]
	base := map[string]any{
		"id":      fmt.Sprintf("%s_%d_%d", idPrefix, time.Now().UnixNano(), index),
		"status":  "completed",
		"call_id": callID,
	}
	switch definition.Type {
	case responseToolFunction:
		base["type"] = "function_call"
		base["name"] = definition.Name
		if definition.Namespace != "" {
			base["namespace"] = definition.Namespace
		}
		base["arguments"] = arguments
	case responseToolCustom:
		if err := validateResponseObjectKeys(decoded, "input"); err != nil {
			return nil, fmt.Errorf("custom tool %q: %w", definition.Name, err)
		}
		input, err := requiredResponseStringAllowEmpty(decoded["input"], "input")
		if err != nil {
			return nil, fmt.Errorf("custom tool %q: %w", definition.Name, err)
		}
		base["type"] = "custom_tool_call"
		base["name"] = definition.Name
		if definition.Namespace != "" {
			base["namespace"] = definition.Namespace
		}
		base["input"] = input
	case responseToolLocalShell:
		if err := validateResponseObjectKeys(decoded, "command", "timeout_ms", "working_directory", "env", "user"); err != nil {
			return nil, fmt.Errorf("local_shell tool call: %w", err)
		}
		action := cloneMap(decoded)
		action["type"] = "exec"
		if _, err := normalizeResponseLocalShellAction(action); err != nil {
			return nil, fmt.Errorf("local_shell tool call: %w", err)
		}
		base["type"] = "local_shell_call"
		base["action"] = action
	case responseToolShell:
		if _, err := normalizeResponseShellAction(decoded); err != nil {
			return nil, fmt.Errorf("shell tool call: %w", err)
		}
		base["type"] = "shell_call"
		base["action"] = decoded
	case responseToolApplyPatch:
		if err := validateResponseObjectKeys(decoded, "operation", "path", "diff"); err != nil {
			return nil, fmt.Errorf("apply_patch tool call: %w", err)
		}
		operation := map[string]any{"type": decoded["operation"], "path": decoded["path"]}
		if diff, exists := decoded["diff"]; exists {
			operation["diff"] = diff
		}
		if _, err := normalizeResponseApplyPatchOperation(operation); err != nil {
			return nil, fmt.Errorf("apply_patch tool call: %w", err)
		}
		base["type"] = "apply_patch_call"
		base["operation"] = operation
	default:
		return nil, fmt.Errorf("tool %q has unsupported Responses type %q", definition.Name, definition.Type)
	}
	return base, nil
}

func responseUsage(inputTokens, outputTokens, totalTokens int) map[string]any {
	return map[string]any{
		"input_tokens":  inputTokens,
		"output_tokens": outputTokens,
		"total_tokens":  totalTokens,
	}
}

func (h *Handler) handleResponseStream(w http.ResponseWriter, body io.Reader, response responseContext, statsModel string, toolNames []string, toolSchemas []toolcall.ToolSchema, estimatedPromptTokens int) {
	setSSEHeaders(w)
	writer := newResponseStreamWriter(w)
	builder := &responseOutputBuilder{response: response, messageIndex: -1}
	writer.emit("response.created", map[string]any{"response": response.payload("in_progress", []map[string]any{}, nil, nil)})
	writer.emit("response.in_progress", map[string]any{"response": response.payload("in_progress", []map[string]any{}, nil, nil)})

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	promptTokens, completionTokens, totalTokens := 0, 0, 0
	streamState := toolcall.NewStreamState()

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}
		var raw map[string]any
		if err := json.Unmarshal([]byte(payload), &raw); err != nil {
			continue
		}
		if upstreamErr := parseAssetError([]byte(payload)); upstreamErr != nil {
			writer.fail(response, upstreamErr.Error())
			return
		}
		promptTokens, completionTokens, totalTokens = extractUsage(raw, promptTokens, completionTokens, totalTokens)

		choices, _ := raw["choices"].([]any)
		if len(choices) == 0 {
			continue
		}
		choice, _ := choices[0].(map[string]any)
		delta, _ := choice["delta"].(map[string]any)
		if delta == nil || isThinkingPhase(stringValue(delta["phase"])) {
			continue
		}
		content := extractDeltaContent(delta)
		if content == "" {
			continue
		}

		if len(toolNames) > 0 {
			chunk := toolcall.ProcessStreamChunk(streamState, content)
			builder.addText(writer, toolcall.CleanVisibleChunk(chunk.Content))
			if err := builder.addToolCalls(writer, chunk.ToolCalls, toolSchemas); err != nil {
				writer.fail(response, fmt.Sprintf("Failed to format upstream tool call: %v", err))
				return
			}
			continue
		}
		builder.addText(writer, content)
	}

	if err := scanner.Err(); err != nil {
		writer.fail(response, err.Error())
		return
	}
	if len(toolNames) > 0 {
		final := toolcall.FinalizeStream(streamState)
		builder.addText(writer, toolcall.CleanVisibleChunk(final.Content))
		if err := builder.addToolCalls(writer, final.ToolCalls, toolSchemas); err != nil {
			writer.fail(response, fmt.Sprintf("Failed to format upstream tool call: %v", err))
			return
		}
	}
	if !builder.messageStarted && len(builder.output) == 0 {
		builder.startMessage(writer)
	}
	builder.finishMessage(writer)

	promptTokens, completionTokens, totalTokens = applyUsageFallback(
		promptTokens,
		completionTokens,
		totalTokens,
		estimatedPromptTokens,
		estimateOpenAIOutputTokens(builder.messageText.String(), builder.toolCalls),
	)
	usage := responseUsage(promptTokens, completionTokens, totalTokens)
	writer.emit("response.completed", map[string]any{
		"response": response.payload("completed", builder.output, usage, nil),
	})
	if h.metrics != nil {
		h.metrics.RecordModelUsage(statsModel, promptTokens, completionTokens, totalTokens)
	}
}

func newResponseStreamWriter(w http.ResponseWriter) *responseStreamWriter {
	flusher, _ := w.(http.Flusher)
	return &responseStreamWriter{w: w, flusher: flusher}
}

func (writer *responseStreamWriter) emit(eventType string, event map[string]any) {
	event["type"] = eventType
	event["sequence_number"] = writer.sequence
	writer.sequence++
	raw, err := json.Marshal(event)
	if err != nil {
		return
	}
	_, _ = fmt.Fprintf(writer.w, "event: %s\ndata: %s\n\n", eventType, raw)
	if writer.flusher != nil {
		writer.flusher.Flush()
	}
}

func (writer *responseStreamWriter) fail(response responseContext, message string) {
	errorObject := map[string]any{"type": "api_error", "message": message, "code": nil}
	writer.emit("response.failed", map[string]any{
		"response": response.payload("failed", []map[string]any{}, nil, errorObject),
	})
}

func (builder *responseOutputBuilder) startMessage(writer *responseStreamWriter) {
	if builder.messageStarted {
		return
	}
	builder.messageStarted = true
	builder.messageID = fmt.Sprintf("msg_%d", time.Now().UnixNano())
	builder.messageIndex = len(builder.output)
	builder.output = append(builder.output, nil)
	writer.emit("response.output_item.added", map[string]any{
		"output_index": builder.messageIndex,
		"item": map[string]any{
			"id": builder.messageID, "type": "message", "status": "in_progress", "role": "assistant", "content": []any{},
		},
	})
	writer.emit("response.content_part.added", map[string]any{
		"item_id": builder.messageID, "output_index": builder.messageIndex, "content_index": 0,
		"part": map[string]any{"type": "output_text", "text": "", "annotations": []any{}},
	})
}

func (builder *responseOutputBuilder) addText(writer *responseStreamWriter, text string) {
	if text == "" {
		return
	}
	builder.startMessage(writer)
	builder.messageText.WriteString(text)
	writer.emit("response.output_text.delta", map[string]any{
		"item_id": builder.messageID, "output_index": builder.messageIndex, "content_index": 0, "delta": text,
	})
}

func (builder *responseOutputBuilder) finishMessage(writer *responseStreamWriter) {
	if !builder.messageStarted {
		return
	}
	text := builder.messageText.String()
	part := map[string]any{"type": "output_text", "text": text, "annotations": []any{}}
	message := responseMessageOutput(builder.messageID, text, "completed")
	writer.emit("response.output_text.done", map[string]any{
		"item_id": builder.messageID, "output_index": builder.messageIndex, "content_index": 0, "text": text,
	})
	writer.emit("response.content_part.done", map[string]any{
		"item_id": builder.messageID, "output_index": builder.messageIndex, "content_index": 0, "part": part,
	})
	writer.emit("response.output_item.done", map[string]any{"output_index": builder.messageIndex, "item": message})
	builder.output[builder.messageIndex] = message
}

func (builder *responseOutputBuilder) addToolCalls(writer *responseStreamWriter, calls []toolcall.ToolCall, schemas []toolcall.ToolSchema) error {
	if len(calls) == 0 {
		return nil
	}
	if !builder.response.ParallelToolCalls {
		if builder.toolCount > 0 {
			return nil
		}
		calls = calls[:1]
	}
	items, err := responseToolOutputs(calls, schemas, builder.response.ToolDefinitions)
	if err != nil {
		return err
	}
	for index, item := range items {
		builder.toolCount++
		builder.toolCalls = append(builder.toolCalls, calls[index])
		outputIndex := len(builder.output)
		added := cloneMap(item)
		added["status"] = "in_progress"
		switch item["type"] {
		case "function_call":
			added["arguments"] = ""
		case "custom_tool_call":
			added["input"] = ""
		}
		writer.emit("response.output_item.added", map[string]any{"output_index": outputIndex, "item": added})
		switch item["type"] {
		case "function_call":
			writer.emit("response.function_call_arguments.delta", map[string]any{
				"item_id": item["id"], "output_index": outputIndex, "delta": item["arguments"],
			})
			writer.emit("response.function_call_arguments.done", map[string]any{
				"item_id": item["id"], "output_index": outputIndex, "name": item["name"], "arguments": item["arguments"],
			})
		case "custom_tool_call":
			writer.emit("response.custom_tool_call_input.delta", map[string]any{
				"item_id": item["id"], "output_index": outputIndex, "delta": item["input"],
			})
			writer.emit("response.custom_tool_call_input.done", map[string]any{
				"item_id": item["id"], "output_index": outputIndex, "input": item["input"],
			})
		}
		writer.emit("response.output_item.done", map[string]any{"output_index": outputIndex, "item": item})
		builder.output = append(builder.output, item)
	}
	return nil
}

func writeResponseError(w http.ResponseWriter, status int, message, errorType string, param, code any) {
	if status < 400 || status > 599 {
		status = http.StatusBadGateway
	}
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    errorType,
			"param":   param,
			"code":    code,
		},
	})
}
