package client

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"

	"github.com/lopesgabriel/ollama-to-adk/internal/converter"
	"github.com/lopesgabriel/ollama-to-adk/internal/ollamaapi"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

type Client struct {
	baseURL string
}

func NewClient(baseURL string) *Client {
	return &Client{baseURL: baseURL}
}

func (c *Client) Generate(ctx context.Context, modelName string, contents []*genai.Content, config *genai.GenerateContentConfig) (*model.LLMResponse, error) {
	httpResponse, err := c.sendChatRequest(ctx, modelName, contents, config, false)
	if err != nil {
		return nil, err
	}
	defer httpResponse.Body.Close()

	rawBody, err := io.ReadAll(httpResponse.Body)
	if err != nil {
		return nil, fmt.Errorf("read chat response: %w", err)
	}

	var chatResponse ollamaapi.ChatResponse
	if err := json.Unmarshal(rawBody, &chatResponse); err != nil {
		return nil, fmt.Errorf("decode chat response: %w", err)
	}

	return mapChatResponse(&chatResponse, httpResponse.Header, rawBody), nil
}

func (c *Client) GenerateStream(ctx context.Context, modelName string, contents []*genai.Content, config *genai.GenerateContentConfig) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		httpResponse, err := c.sendChatRequest(ctx, modelName, contents, config, true)
		if err != nil {
			yield(nil, err)
			return
		}
		defer httpResponse.Body.Close()

		decoder := json.NewDecoder(httpResponse.Body)
		for {
			var event ollamaapi.ChatResponse
			if err := decoder.Decode(&event); err != nil {
				if errors.Is(err, io.EOF) {
					return
				}
				yield(nil, fmt.Errorf("decode chat stream event: %w", err))
				return
			}

			response := mapChatStreamResponse(&event, httpResponse.Header)
			if !yield(response, nil) {
				return
			}
		}
	}
}

func (c *Client) sendChatRequest(ctx context.Context, modelName string, contents []*genai.Content, config *genai.GenerateContentConfig, stream bool) (*http.Response, error) {
	requestBody, err := converter.BuildChatRequest(modelName, contents, config)
	if err != nil {
		return nil, fmt.Errorf("build chat request: %w", err)
	}

	requestBody.Stream = &stream

	bodyBytes, err := marshalChatRequestBody(requestBody, config)
	if err != nil {
		return nil, fmt.Errorf("marshal chat request: %w", err)
	}

	endpoint := strings.TrimRight(c.resolveBaseURL(config), "/") + "/api/chat"
	httpRequest, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, fmt.Errorf("create chat request: %w", err)
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	applyHeaders(httpRequest.Header, config)

	httpClient := &http.Client{}
	if config != nil && config.HTTPOptions != nil && config.HTTPOptions.Timeout != nil {
		httpClient.Timeout = *config.HTTPOptions.Timeout
	}

	httpResponse, err := httpClient.Do(httpRequest)
	if err != nil {
		return nil, fmt.Errorf("post /api/chat: %w", err)
	}

	if httpResponse.StatusCode < http.StatusOK || httpResponse.StatusCode >= http.StatusMultipleChoices {
		defer httpResponse.Body.Close()
		rawBody, err := io.ReadAll(httpResponse.Body)
		if err != nil {
			return nil, fmt.Errorf("read chat error response: %w", err)
		}
		return nil, decodeErrorResponse(httpResponse.StatusCode, rawBody)
	}

	return httpResponse, nil
}

func marshalChatRequestBody(requestBody *ollamaapi.ChatRequest, config *genai.GenerateContentConfig) ([]byte, error) {
	if config == nil || config.HTTPOptions == nil || (len(config.HTTPOptions.ExtraBody) == 0 && config.HTTPOptions.ExtrasRequestProvider == nil) {
		return json.Marshal(requestBody)
	}

	data, err := json.Marshal(requestBody)
	if err != nil {
		return nil, err
	}

	var body map[string]any
	if err := json.Unmarshal(data, &body); err != nil {
		return nil, err
	}

	for key, value := range config.HTTPOptions.ExtraBody {
		body[key] = value
	}
	if config.HTTPOptions.ExtrasRequestProvider != nil {
		body = config.HTTPOptions.ExtrasRequestProvider(body)
	}
	body["stream"] = false

	return json.Marshal(body)
}

func applyHeaders(headers http.Header, config *genai.GenerateContentConfig) {
	if config == nil || config.HTTPOptions == nil {
		return
	}
	for key, values := range config.HTTPOptions.Headers {
		for _, value := range values {
			headers.Add(key, value)
		}
	}
}

func (c *Client) resolveBaseURL(config *genai.GenerateContentConfig) string {
	if config != nil && config.HTTPOptions != nil && config.HTTPOptions.BaseURL != "" {
		return config.HTTPOptions.BaseURL
	}
	if c.baseURL != "" {
		return c.baseURL
	}
	return "http://localhost:11434"
}

func decodeErrorResponse(statusCode int, rawBody []byte) error {
	var errorResponse ollamaapi.ErrorResponse
	if err := json.Unmarshal(rawBody, &errorResponse); err == nil && errorResponse.Error != "" {
		return fmt.Errorf("ollama /api/chat returned %d: %s", statusCode, errorResponse.Error)
	}
	trimmed := strings.TrimSpace(string(rawBody))
	if trimmed == "" {
		trimmed = http.StatusText(statusCode)
	}
	return fmt.Errorf("ollama /api/chat returned %d: %s", statusCode, trimmed)
}

func mapChatResponse(chatResponse *ollamaapi.ChatResponse, headers http.Header, rawBody []byte) *model.LLMResponse {
	response := buildLLMResponse(chatResponse)
	response.CustomMetadata = mergeCustomMetadata(response.CustomMetadata, map[string]any{
		"sdk_http_response": &genai.HTTPResponse{
			Headers: headers,
			Body:    string(rawBody),
		},
	})
	return response
}

func mapChatStreamResponse(chatResponse *ollamaapi.ChatResponse, headers http.Header) *model.LLMResponse {
	response := buildLLMResponse(chatResponse)
	response.Partial = !chatResponse.Done
	response.TurnComplete = chatResponse.Done
	response.CustomMetadata = mergeCustomMetadata(response.CustomMetadata, map[string]any{
		"stream_headers": headers,
	})
	return response
}

func buildLLMResponse(chatResponse *ollamaapi.ChatResponse) *model.LLMResponse {
	parts := make([]*genai.Part, 0, 1+len(chatResponse.Message.ToolCalls)+len(chatResponse.Message.Images))
	if chatResponse.Message.Thinking != "" {
		parts = append(parts, &genai.Part{Text: chatResponse.Message.Thinking, Thought: true})
	}
	if chatResponse.Message.Content != "" {
		parts = append(parts, genai.NewPartFromText(chatResponse.Message.Content))
	}
	for _, toolCall := range chatResponse.Message.ToolCalls {
		parts = append(parts, genai.NewPartFromFunctionCall(toolCall.Function.Name, toolCall.Function.Arguments))
	}
	for _, image := range chatResponse.Message.Images {
		part := imagePartFromBase64(image)
		if part != nil {
			parts = append(parts, part)
		}
	}

	content := &genai.Content{
		Role:  string(genai.RoleModel),
		Parts: parts,
	}

	response := &model.LLMResponse{
		Content: content,
		UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     chatResponse.PromptEvalCount,
			CandidatesTokenCount: chatResponse.EvalCount,
			TotalTokenCount:      chatResponse.PromptEvalCount + chatResponse.EvalCount,
		},
		ModelVersion: chatResponse.Model,
		FinishReason: mapFinishReason(chatResponse.DoneReason),
		CustomMetadata: map[string]any{
			"ollama_done":                 chatResponse.Done,
			"ollama_done_reason":          chatResponse.DoneReason,
			"ollama_created_at":           chatResponse.CreatedAt,
			"ollama_total_duration":       chatResponse.TotalDuration,
			"ollama_load_duration":        chatResponse.LoadDuration,
			"ollama_prompt_eval_duration": chatResponse.PromptEvalDuration,
			"ollama_eval_duration":        chatResponse.EvalDuration,
		},
	}

	if len(chatResponse.Logprobs) > 0 {
		response.LogprobsResult = mapLogprobs(chatResponse.Logprobs)
	}

	return response
}

func mergeCustomMetadata(base map[string]any, extra map[string]any) map[string]any {
	if len(extra) == 0 {
		return base
	}
	if base == nil {
		base = map[string]any{}
	}
	for key, value := range extra {
		base[key] = value
	}
	return base
}

func imagePartFromBase64(encoded string) *genai.Part {
	data, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		return nil
	}
	mimeType := http.DetectContentType(data)
	return &genai.Part{
		InlineData: &genai.Blob{
			Data:     data,
			MIMEType: mimeType,
		},
	}
}

func mapFinishReason(doneReason string) genai.FinishReason {
	switch strings.ToLower(doneReason) {
	case "", "unknown":
		return genai.FinishReasonUnspecified
	case "stop":
		return genai.FinishReasonStop
	case "length", "max_tokens":
		return genai.FinishReasonMaxTokens
	case "safety", "content_filter":
		return genai.FinishReasonSafety
	case "tool_call":
		return genai.FinishReasonUnexpectedToolCall
	default:
		return genai.FinishReasonOther
	}
}

func mapLogprobs(logprobs []ollamaapi.Logprob) *genai.LogprobsResult {
	result := &genai.LogprobsResult{
		ChosenCandidates: make([]*genai.LogprobsResultCandidate, 0, len(logprobs)),
		TopCandidates:    make([]*genai.LogprobsResultTopCandidates, 0, len(logprobs)),
	}

	var sum float32
	for _, token := range logprobs {
		logProbability := float32(token.Logprob)
		sum += logProbability
		result.ChosenCandidates = append(result.ChosenCandidates, &genai.LogprobsResultCandidate{
			Token:          token.Token,
			LogProbability: logProbability,
		})

		topCandidates := &genai.LogprobsResultTopCandidates{}
		for _, topToken := range token.TopLogprobs {
			topCandidates.Candidates = append(topCandidates.Candidates, &genai.LogprobsResultCandidate{
				Token:          topToken.Token,
				LogProbability: float32(topToken.Logprob),
			})
		}
		result.TopCandidates = append(result.TopCandidates, topCandidates)
	}
	result.LogProbabilitySum = &sum

	return result
}
