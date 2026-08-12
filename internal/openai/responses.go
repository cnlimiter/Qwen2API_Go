package openai

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"qwen2api/internal/toolcall"
)

type responsesRequest struct {
	Model             string                 `json:"model"`
	Input             json.RawMessage        `json:"input"`
	Instructions      *string                `json:"instructions"`
	Stream            bool                   `json:"stream"`
	Tools             []responseFunctionTool `json:"tools"`
	ToolChoice        any                    `json:"tool_choice"`
	ParallelToolCalls *bool                  `json:"parallel_tool_calls"`
	Metadata          map[string]any         `json:"metadata"`
	Reasoning         *chatReasoning         `json:"reasoning"`
}

type responseFunctionTool struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters"`
	Strict      *bool          `json:"strict,omitempty"`
}

type normalizedResponsesRequest struct {
	Messages       []map[string]any
	ChatTools      []any
	ChatToolChoice any
	ResponseTools  []map[string]any
	ToolChoice     any
	Parallel       bool
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
	messages, err := normalizeResponseInput(payload.Input)
	if err != nil {
		return normalizedResponsesRequest{}, err
	}
	if payload.Instructions != nil {
		messages = append([]map[string]any{{"role": "system", "content": *payload.Instructions}}, messages...)
	}

	chatTools, responseTools, toolNames, err := normalizeResponseTools(payload.Tools)
	if err != nil {
		return normalizedResponsesRequest{}, err
	}
	chatToolChoice, responseToolChoice, err := normalizeResponseToolChoice(payload.ToolChoice, toolNames)
	if err != nil {
		return normalizedResponsesRequest{}, err
	}

	parallel := true
	if payload.ParallelToolCalls != nil {
		parallel = *payload.ParallelToolCalls
	}
	return normalizedResponsesRequest{
		Messages:       messages,
		ChatTools:      chatTools,
		ChatToolChoice: chatToolChoice,
		ResponseTools:  responseTools,
		ToolChoice:     responseToolChoice,
		Parallel:       parallel,
	}, nil
}

func normalizeResponseInput(raw json.RawMessage) ([]map[string]any, error) {
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
	functionNames := make(map[string]string)
	for index, item := range items {
		itemType := strings.ToLower(strings.TrimSpace(stringValue(item["type"])))
		switch itemType {
		case "function_call":
			callID := strings.TrimSpace(stringValue(item["call_id"]))
			name := strings.TrimSpace(stringValue(item["name"]))
			arguments, ok := item["arguments"].(string)
			var decodedArguments map[string]any
			if callID == "" || name == "" || !ok || json.Unmarshal([]byte(arguments), &decodedArguments) != nil {
				return nil, responseInputError(fmt.Sprintf("input[%d] has an invalid function_call", index))
			}
			if _, exists := functionNames[callID]; exists {
				return nil, responseInputError(fmt.Sprintf("input[%d].call_id is duplicated", index))
			}
			functionNames[callID] = name
			messages = append(messages, map[string]any{
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
			})
		case "function_call_output":
			callID := strings.TrimSpace(stringValue(item["call_id"]))
			if callID == "" {
				return nil, responseInputError(fmt.Sprintf("input[%d].call_id is required", index))
			}
			name, exists := functionNames[callID]
			if !exists {
				return nil, responseInputError(fmt.Sprintf("input[%d].call_id does not match a previous function_call", index))
			}
			output, err := normalizeResponseTextContent(item["output"])
			if err != nil {
				return nil, responseInputError(fmt.Sprintf("input[%d].output must contain text", index))
			}
			messages = append(messages, map[string]any{
				"role":         "tool",
				"tool_call_id": callID,
				"name":         name,
				"content":      output,
			})
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

func normalizeResponseTools(tools []responseFunctionTool) ([]any, []map[string]any, []string, error) {
	chatTools := make([]any, 0, len(tools))
	responseTools := make([]map[string]any, 0, len(tools))
	toolNames := make([]string, 0, len(tools))
	for index, tool := range tools {
		if !strings.EqualFold(strings.TrimSpace(tool.Type), "function") {
			return nil, nil, nil, responseToolError(fmt.Sprintf("tools[%d].type must be function", index))
		}
		name := strings.TrimSpace(tool.Name)
		if name == "" {
			return nil, nil, nil, responseToolError(fmt.Sprintf("tools[%d].name is required", index))
		}
		if tool.Parameters == nil {
			return nil, nil, nil, responseToolError(fmt.Sprintf("tools[%d].parameters is required", index))
		}
		function := map[string]any{
			"name":        name,
			"description": tool.Description,
			"parameters":  tool.Parameters,
		}
		chatTools = append(chatTools, map[string]any{"type": "function", "function": function})
		echo := map[string]any{
			"type":        "function",
			"name":        name,
			"description": tool.Description,
			"parameters":  tool.Parameters,
		}
		if tool.Strict != nil {
			echo["strict"] = *tool.Strict
		}
		responseTools = append(responseTools, echo)
		toolNames = append(toolNames, name)
	}
	return chatTools, responseTools, toolNames, nil
}

func normalizeResponseToolChoice(raw any, toolNames []string) (any, any, error) {
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
			if len(toolNames) == 0 {
				return nil, nil, responseToolChoiceError("tool_choice required needs at least one function tool")
			}
			return choice, choice, nil
		default:
			return nil, nil, responseToolChoiceError("tool_choice must be none, auto, required, or a named function")
		}
	case map[string]any:
		if !strings.EqualFold(strings.TrimSpace(stringValue(value["type"])), "function") {
			return nil, nil, responseToolChoiceError("tool_choice.type must be function")
		}
		name := strings.TrimSpace(stringValue(value["name"]))
		if name == "" || !containsString(toolNames, name) {
			return nil, nil, responseToolChoiceError("tool_choice.name must reference a provided function tool")
		}
		return map[string]any{
			"type":     "function",
			"function": map[string]any{"name": name},
		}, map[string]any{"type": "function", "name": name}, nil
	default:
		return nil, nil, responseToolChoiceError("tool_choice has an invalid type")
	}
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
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
	output = append(output, responseFunctionOutputs(result.ToolCalls, toolSchemas)...)
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

func responseFunctionOutputs(calls []toolcall.ToolCall, schemas []toolcall.ToolSchema) []map[string]any {
	formatted := toolcall.FormatOpenAIToolCallsWithSchemas(calls, schemas)
	output := make([]map[string]any, 0, len(formatted))
	for index, call := range formatted {
		function, _ := call["function"].(map[string]any)
		callID := strings.TrimSpace(stringValue(call["id"]))
		if callID == "" {
			callID = fmt.Sprintf("call_%d_%d", time.Now().UnixNano(), index)
		}
		output = append(output, map[string]any{
			"id":        fmt.Sprintf("fc_%d_%d", time.Now().UnixNano(), index),
			"type":      "function_call",
			"status":    "completed",
			"call_id":   callID,
			"name":      stringValue(function["name"]),
			"arguments": stringValue(function["arguments"]),
		})
	}
	return output
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
			builder.addToolCalls(writer, chunk.ToolCalls, toolSchemas)
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
		builder.addToolCalls(writer, final.ToolCalls, toolSchemas)
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

func (builder *responseOutputBuilder) addToolCalls(writer *responseStreamWriter, calls []toolcall.ToolCall, schemas []toolcall.ToolSchema) {
	if len(calls) == 0 {
		return
	}
	if !builder.response.ParallelToolCalls {
		if builder.toolCount > 0 {
			return
		}
		calls = calls[:1]
	}
	for index, item := range responseFunctionOutputs(calls, schemas) {
		builder.toolCount++
		builder.toolCalls = append(builder.toolCalls, calls[index])
		outputIndex := len(builder.output)
		added := cloneMap(item)
		added["status"] = "in_progress"
		added["arguments"] = ""
		writer.emit("response.output_item.added", map[string]any{"output_index": outputIndex, "item": added})
		writer.emit("response.function_call_arguments.delta", map[string]any{
			"item_id": item["id"], "output_index": outputIndex, "delta": item["arguments"],
		})
		writer.emit("response.function_call_arguments.done", map[string]any{
			"item_id": item["id"], "output_index": outputIndex, "name": item["name"], "arguments": item["arguments"],
		})
		writer.emit("response.output_item.done", map[string]any{"output_index": outputIndex, "item": item})
		builder.output = append(builder.output, item)
	}
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
