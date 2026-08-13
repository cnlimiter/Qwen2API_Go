package openai

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"

	"qwen2api/internal/prompts"
	"qwen2api/internal/qwen"
	"qwen2api/internal/storage"
	"qwen2api/internal/toolcall"
)

type executedChatRequest struct {
	Model                 string
	Messages              []map[string]any
	EnableThinking        any
	ReasoningEffort       any
	NestedReasoningEffort any
	Tools                 any
	ToolChoice            any
	ChatType              string
	Size                  string
}

type executedChat struct {
	Model          string
	RequestedModel string
	ToolNames      []string
	ToolSchemas    []toolcall.ToolSchema
	Stream         io.ReadCloser
}

type completedChat struct {
	Content          string
	ReasoningContent string
	ToolCalls        []toolcall.ToolCall
	FinishReason     string
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

func (h *Handler) executeChatRequest(ctx context.Context, payload executedChatRequest) (*executedChat, int, error) {
	prepared := h.prepareChatRequest(ctx, payload)
	maxAttempts := len(h.accounts.Accounts())
	if maxAttempts <= 0 {
		maxAttempts = 1
	}

	attempted := map[string]struct{}{}

	if prepared.ContextHash != "" {
		if mapped, ok := h.sessions.Get(prepared.ContextHash); ok && mapped.Model == prepared.Model && mapped.ChatType == prepared.ChatType {
			if session, err := h.accounts.GetAccountSessionByEmail(mapped.AccountEmail); err == nil {
				sessionCtx := bindAccountContext(ctx, session)
				h.logger.DebugModule("OPENAI", "reuse mapped chat model=%s hash=%s account=%s chat_id=%s", prepared.Model, prepared.ContextHash, mapped.AccountEmail, mapped.ChatID)
				executed, status, err := h.sendChatWithSession(sessionCtx, prepared, session, mapped.ChatID, true)
				if err == nil {
					h.sessions.Save(prepared.ContextHash, session.Email, mapped.ChatID, prepared.Model, prepared.ChatType)
					h.recordChatUsage(session.Email, mapped.ChatID)
					return executed, status, nil
				}
				if upstreamErr, ok := err.(*qwen.UpstreamError); ok {
					if shouldInvalidateConversationMapping(upstreamErr) {
						h.sessions.Delete(prepared.ContextHash)
						h.logger.WarnModule("OPENAI", "invalidate mapped chat model=%s hash=%s account=%s chat_id=%s err=%v", prepared.Model, prepared.ContextHash, session.Email, mapped.ChatID, upstreamErr)
					} else if upstreamErr.Retryable {
						h.accounts.RecordFailureAndRefresh(sessionCtx, session.Email)
						attempted[session.Email] = struct{}{}
						h.logger.WarnModule("OPENAI", "mapped chat retryable error model=%s hash=%s account=%s chat_id=%s status=%d err=%v", prepared.Model, prepared.ContextHash, session.Email, mapped.ChatID, upstreamErr.StatusCode, upstreamErr)
					} else {
						return nil, normalizeUpstreamStatus(upstreamErr.StatusCode), upstreamErr
					}
				} else {
					h.accounts.RecordFailureAndRefresh(sessionCtx, session.Email)
					attempted[session.Email] = struct{}{}
				}
			}
		}
	}

	var lastErr error
	lastStatus := http.StatusBadGateway
	for len(attempted) < maxAttempts {
		session, err := h.accounts.GetAccountSessionExcluding(attempted)
		if err != nil {
			break
		}
		attempted[session.Email] = struct{}{}

		sessionCtx := bindAccountContext(ctx, session)
		executed, status, err := h.sendChatWithSession(sessionCtx, prepared, session, "", false)
		if err == nil {
			if prepared.ContextHash != "" {
				if chatID := chatIDFromStream(executed.Stream); chatID != "" {
					h.sessions.Save(prepared.ContextHash, session.Email, chatID, prepared.Model, prepared.ChatType)
					h.recordChatUsage(session.Email, chatID)
				}
			}
			return executed, status, nil
		}

		lastErr = err
		lastStatus = status
		if upstreamErr, ok := err.(*qwen.UpstreamError); ok {
			h.accounts.RecordFailureAndRefresh(sessionCtx, session.Email)
			if shouldInvalidateConversationMapping(upstreamErr) && prepared.ContextHash != "" {
				h.sessions.Delete(prepared.ContextHash)
			}
			if upstreamErr.Retryable {
				continue
			}
		} else {
			h.accounts.RecordFailureAndRefresh(sessionCtx, session.Email)
		}
		return nil, status, err
	}

	if lastErr == nil {
		lastErr = fmt.Errorf("上游聊天请求失败")
	}
	return nil, lastStatus, lastErr
}

func (h *Handler) prepareChatRequest(ctx context.Context, payload executedChatRequest) preparedChatRequest {
	chatType := chatTypeForModel(payload.Model)
	if strings.TrimSpace(payload.ChatType) != "" {
		chatType = strings.TrimSpace(payload.ChatType)
	}
	model, _ := h.ResolveModel(ctx, payload.Model, chatType)
	thinkingMode := resolveThinkingMode(payload.Model, payload.ReasoningEffort, payload.NestedReasoningEffort, payload.EnableThinking)
	messages := injectQwenWeb2ControlPrompt(payload.Messages, h.qwenWeb2ControlPrompt())
	injection := toolcall.InjectPromptWithOverrides(messages, payload.Tools, payload.ToolChoice, h.promptOverrides())
	expandedMessages := cloneMessageList(injection.Messages)
	fullUpstreamMessages := normalizeMessages(cloneMessageList(expandedMessages), chatType, thinkingMode)

	lastUpstreamMessages := fullUpstreamMessages
	if len(expandedMessages) > 1 {
		lastRaw := selectIncrementalTailMessages(expandedMessages)
		lastExpanded := toolcall.NormalizeToolMessagesForExecution(lastRaw)
		lastUpstreamMessages = normalizeMessages(lastExpanded, chatType, thinkingMode)
	}

	return preparedChatRequest{
		RequestedModel:       strings.TrimSpace(payload.Model),
		Model:                model,
		ChatType:             chatType,
		ThinkingMode:         thinkingMode,
		ExpandedMessages:     expandedMessages,
		FullUpstreamMessages: fullUpstreamMessages,
		LastUpstreamMessages: lastUpstreamMessages,
		ContextHash:          computeContextHash(model, chatType, injection.ToolNames, expandedMessages),
		ToolNames:            injection.ToolNames,
		ToolSchemas:          injection.ToolSchemas,
	}
}

func (h *Handler) qwenWeb2ControlPrompt() string {
	return h.promptValue(prompts.IDQwenWeb2Control)
}

func (h *Handler) promptValue(id string) string {
	if h == nil {
		return ""
	}
	if h.runtime != nil {
		return prompts.Resolve(h.runtime.Snapshot().PromptOverrides, id)
	}
	return prompts.Resolve(h.cfg.PromptOverrides, id)
}

func (h *Handler) promptOverrides() map[string]string {
	if h == nil {
		return nil
	}
	if h.runtime != nil {
		return h.runtime.Snapshot().PromptOverrides
	}
	return prompts.CloneOverrides(h.cfg.PromptOverrides)
}

func injectQwenWeb2ControlPrompt(messages []map[string]any, prompt string) []map[string]any {
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return messages
	}
	injected := make([]map[string]any, 0, len(messages)+1)
	injected = append(injected, map[string]any{
		"role":    "system",
		"content": prompt,
	})
	injected = append(injected, messages...)
	return injected
}

func selectIncrementalTailMessages(messages []map[string]any) []map[string]any {
	if len(messages) == 0 {
		return nil
	}

	start := len(messages) - 1
	for start > 0 && strings.EqualFold(strings.TrimSpace(fmt.Sprint(messages[start-1]["role"])), "tool") {
		start--
	}
	return cloneMessageList(messages[start:])
}

func (h *Handler) sendChatWithSession(ctx context.Context, prepared preparedChatRequest, session storage.Account, existingChatID string, incremental bool) (*executedChat, int, error) {
	ctx = bindAccountContext(ctx, session)
	return h.sendChatWithSessionAttempt(ctx, prepared, session, existingChatID, incremental, true)
}

func (h *Handler) sendChatWithSessionAttempt(ctx context.Context, prepared preparedChatRequest, session storage.Account, existingChatID string, incremental bool, allowGuestRefresh bool) (*executedChat, int, error) {
	chatID := strings.TrimSpace(existingChatID)
	if chatID == "" {
		var err error
		chatID, err = h.qwen.NewChat(ctx, session.Token, prepared.Model, prepared.ChatType)
		if err != nil {
			if allowGuestRefresh && session.IsGuest() && h.refreshGuestSession(ctx, session, err) == nil {
				return h.sendChatWithSessionAttempt(ctx, prepared, session, "", false, false)
			}
			if upstreamErr, ok := err.(*qwen.UpstreamError); ok {
				return nil, normalizeUpstreamStatus(upstreamErr.StatusCode), err
			}
			return nil, http.StatusBadGateway, err
		}
	}

	baseMessages := prepared.FullUpstreamMessages
	if incremental && len(prepared.LastUpstreamMessages) > 0 {
		baseMessages = prepared.LastUpstreamMessages
	}

	upstreamMessages, err := h.uploadInlineMedia(ctx, session.Token, cloneMessageList(baseMessages))
	if err != nil {
		return nil, http.StatusBadGateway, err
	}

	body := buildChatRequestBody(session, prepared.Model, chatID, prepared.ChatType, upstreamMessages)

	resp, err := h.qwen.ChatCompletions(ctx, session.Token, chatID, body)
	if err != nil {
		if allowGuestRefresh && session.IsGuest() && h.refreshGuestSession(ctx, session, err) == nil {
			return h.sendChatWithSessionAttempt(ctx, prepared, session, "", false, false)
		}
		if upstreamErr, ok := err.(*qwen.UpstreamError); ok {
			return nil, normalizeUpstreamStatus(upstreamErr.StatusCode), err
		}
		return nil, http.StatusBadGateway, err
	}
	inspected, err := qwen.InspectUpstreamStream(ctx, resp.Body)
	if err != nil {
		if allowGuestRefresh && session.IsGuest() && h.refreshGuestSession(ctx, session, err) == nil {
			return h.sendChatWithSessionAttempt(ctx, prepared, session, "", false, false)
		}
		return nil, http.StatusBadGateway, err
	}
	if inspected.UpstreamError != nil {
		if allowGuestRefresh && session.IsGuest() && h.refreshGuestSession(ctx, session, inspected.UpstreamError) == nil {
			return h.sendChatWithSessionAttempt(ctx, prepared, session, "", false, false)
		}
		return nil, normalizeUpstreamStatus(inspected.UpstreamError.StatusCode), inspected.UpstreamError
	}
	h.accounts.ResetFailure(session.Email)
	stream := withChatID(inspected.Stream, chatID)
	return &executedChat{
		Model:          prepared.Model,
		RequestedModel: prepared.RequestedModel,
		ToolNames:      prepared.ToolNames,
		Stream:         stream,
		ToolSchemas:    prepared.ToolSchemas,
	}, http.StatusOK, nil
}

func (h *Handler) refreshGuestSession(ctx context.Context, session storage.Account, cause error) error {
	if !session.IsGuest() {
		return nil
	}
	h.logger.WarnModule("OPENAI", "guest session failed, refreshing anonymous cookies account=%s err=%v", session.Email, cause)
	return h.accounts.RefreshAccount(ctx, session.Email)
}

func normalizeUpstreamStatus(status int) int {
	if status <= 0 {
		return http.StatusBadGateway
	}
	return status
}

func shouldInvalidateConversationMapping(err *qwen.UpstreamError) bool {
	if err == nil {
		return false
	}
	haystack := strings.ToLower(strings.TrimSpace(err.Error()))
	return strings.Contains(haystack, "chat_id") ||
		strings.Contains(haystack, "not found") ||
		strings.Contains(haystack, "permission") ||
		strings.Contains(haystack, "unauthorized")
}

type streamWithChatID struct {
	io.ReadCloser
	chatID string
}

func withChatID(stream io.ReadCloser, chatID string) io.ReadCloser {
	return &streamWithChatID{ReadCloser: stream, chatID: chatID}
}

func chatIDFromStream(stream io.ReadCloser) string {
	if wrapped, ok := stream.(*streamWithChatID); ok {
		return wrapped.chatID
	}
	return ""
}

func (h *Handler) readCompletedChat(body io.Reader, model string, toolNames []string) (completedChat, *qwen.UpstreamError, error) {
	rawBody, err := io.ReadAll(body)
	if err != nil {
		return completedChat{}, nil, err
	}
	h.logger.DebugModule("OPENAI", "non-stream upstream raw response model=%s body=%s", model, string(rawBody))
	if upstreamErr := parseAssetError(rawBody); upstreamErr != nil {
		return completedChat{}, upstreamErr, nil
	}

	fullContent, reasoningContent, promptTokens, completionTokens, totalTokens := parseChatCompletionContent(rawBody, h.shouldExposeThinking())
	h.logger.DebugModule("OPENAI", "non-stream parsed model=%s full_content=%q usage=%s", model, fullContent, debugJSON(map[string]any{
		"prompt_tokens":     promptTokens,
		"completion_tokens": completionTokens,
		"total_tokens":      totalTokens,
	}))
	parsedCalls := []toolcall.ToolCall(nil)
	normalizedContent := fullContent
	if len(toolNames) > 0 {
		parsedCalls = toolcall.ParseCalls(fullContent)
		if len(parsedCalls) > 0 {
			normalizedContent = toolcall.CleanVisibleText(fullContent)
		}
	}
	normalizedContent = toolcall.CleanVisibleText(normalizedContent)
	h.logger.DebugModule("OPENAI", "non-stream normalized model=%s normalized_content=%q parsed_tool_calls=%s", model, normalizedContent, debugJSON(parsedCalls))

	finishReason := "stop"
	if len(parsedCalls) > 0 {
		finishReason = "tool_calls"
	}

	return completedChat{
		Content:          strings.TrimSpace(normalizedContent),
		ReasoningContent: strings.TrimSpace(reasoningContent),
		ToolCalls:        parsedCalls,
		FinishReason:     finishReason,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      totalTokens,
	}, nil, nil
}

func (h *Handler) recordChatUsage(accountEmail, chatID string) {
	if h.chatTracker == nil || strings.TrimSpace(accountEmail) == "" || strings.TrimSpace(chatID) == "" {
		return
	}
	if err := h.chatTracker.RecordChatUsage(accountEmail, chatID); err != nil && h.logger != nil {
		h.logger.WarnModule("OPENAI", "记录对话使用失败 account=%s chat_id=%s err=%v", accountEmail, chatID, err)
	}
}
