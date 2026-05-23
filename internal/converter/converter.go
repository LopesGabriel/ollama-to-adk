package converter

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/lopesgabriel/ollama-to-adk/internal/ollamaapi"
	"google.golang.org/genai"
)

func BuildChatRequest(model string, contents []*genai.Content, config *genai.GenerateContentConfig) (*ollamaapi.ChatRequest, error) {
	request := &ollamaapi.ChatRequest{
		Model: model,
	}

	if err := applyConfig(request, config); err != nil {
		return nil, err
	}

	messages, err := buildMessages(contents, config)
	if err != nil {
		return nil, err
	}

	request.Messages = messages
	return request, nil
}

func buildMessages(contents []*genai.Content, config *genai.GenerateContentConfig) ([]ollamaapi.ChatMessage, error) {
	var messages []ollamaapi.ChatMessage

	if config != nil && config.SystemInstruction != nil {
		systemMessages, err := contentToChatMessages(config.SystemInstruction, ollamaapi.ChatRoleSystem)
		if err != nil {
			return nil, err
		}
		messages = append(messages, systemMessages...)
	}

	for _, content := range contents {
		converted, err := contentToChatMessages(content, mapContentRole(content))
		if err != nil {
			return nil, err
		}
		messages = append(messages, converted...)
	}

	if len(messages) == 0 {
		return nil, fmt.Errorf("chat request requires at least one message")
	}

	return messages, nil
}

func applyConfig(request *ollamaapi.ChatRequest, config *genai.GenerateContentConfig) error {
	if config == nil {
		return nil
	}

	options := &ollamaapi.ModelOptions{}
	hasOptions := false

	if config.Seed != nil {
		seed := int64(*config.Seed)
		options.Seed = &seed
		hasOptions = true
	}

	if config.Temperature != nil {
		temperature := float64(*config.Temperature)
		options.Temperature = &temperature
		hasOptions = true
	}

	if config.TopP != nil {
		topP := float64(*config.TopP)
		options.TopP = &topP
		hasOptions = true
	}

	if config.TopK != nil {
		topK := int64(*config.TopK)
		options.TopK = &topK
		hasOptions = true
	}

	if config.MaxOutputTokens > 0 {
		numPredict := int64(config.MaxOutputTokens)
		options.NumPredict = &numPredict
		hasOptions = true
	}

	if len(config.StopSequences) == 1 {
		options.Stop = ollamaapi.StopSequence(config.StopSequences[0])
		hasOptions = true
	} else if len(config.StopSequences) > 1 {
		options.Stop = ollamaapi.StopSequencesList(config.StopSequences...)
		hasOptions = true
	}

	extra := map[string]any{}
	if config.PresencePenalty != nil {
		extra["presence_penalty"] = float64(*config.PresencePenalty)
	}
	if config.FrequencyPenalty != nil {
		extra["frequency_penalty"] = float64(*config.FrequencyPenalty)
	}
	if len(extra) > 0 {
		options.Extra = extra
		hasOptions = true
	}

	if hasOptions {
		request.Options = options
	}

	format, err := convertFormat(config)
	if err != nil {
		return err
	}
	request.Format = format

	tools, err := convertTools(config.Tools, config.ToolConfig)
	if err != nil {
		return err
	}
	request.Tools = tools

	think := convertThinking(config.ThinkingConfig)
	request.Think = think

	if config.ResponseLogprobs || config.Logprobs != nil {
		enabled := true
		request.Logprobs = &enabled
	}

	if config.Logprobs != nil {
		topLogprobs := int(*config.Logprobs)
		request.TopLogprobs = &topLogprobs
	}

	return nil
}

func convertFormat(config *genai.GenerateContentConfig) (*ollamaapi.ChatFormat, error) {
	if config == nil {
		return nil, nil
	}

	if config.ResponseJsonSchema != nil {
		schema, err := genericSchemaToJSONSchema(config.ResponseJsonSchema)
		if err != nil {
			return nil, fmt.Errorf("convert response json schema: %w", err)
		}
		return ollamaapi.SchemaResponseFormat(schema), nil
	}

	if config.ResponseSchema != nil {
		return ollamaapi.SchemaResponseFormat(schemaToJSONSchema(config.ResponseSchema)), nil
	}

	if strings.EqualFold(config.ResponseMIMEType, "application/json") {
		return ollamaapi.JSONResponseFormat(), nil
	}

	return nil, nil
}

func convertTools(tools []*genai.Tool, toolConfig *genai.ToolConfig) ([]ollamaapi.ToolDefinition, error) {
	allowedNames := map[string]struct{}{}
	modeNone := false
	if toolConfig != nil && toolConfig.FunctionCallingConfig != nil {
		for _, name := range toolConfig.FunctionCallingConfig.AllowedFunctionNames {
			allowedNames[name] = struct{}{}
		}
		modeNone = toolConfig.FunctionCallingConfig.Mode == genai.FunctionCallingConfigModeNone
	}

	var definitions []ollamaapi.ToolDefinition
	for _, tool := range tools {
		if tool == nil {
			continue
		}

		if tool.Retrieval != nil || tool.ComputerUse != nil || tool.FileSearch != nil || tool.GoogleSearch != nil || tool.GoogleMaps != nil || tool.CodeExecution != nil || tool.EnterpriseWebSearch != nil || tool.GoogleSearchRetrieval != nil || tool.ParallelAISearch != nil || tool.URLContext != nil || len(tool.MCPServers) > 0 {
			return nil, fmt.Errorf("unsupported genai tool for ollama chat request")
		}

		for _, declaration := range tool.FunctionDeclarations {
			if declaration == nil {
				continue
			}
			if modeNone {
				continue
			}
			if len(allowedNames) > 0 {
				if _, ok := allowedNames[declaration.Name]; !ok {
					continue
				}
			}

			parameters, err := convertToolParameters(declaration)
			if err != nil {
				return nil, fmt.Errorf("convert tool %q: %w", declaration.Name, err)
			}

			definitions = append(definitions, ollamaapi.ToolDefinition{
				Type: ollamaapi.ToolTypeFunction,
				Function: ollamaapi.ToolFunction{
					Name:        declaration.Name,
					Description: declaration.Description,
					Parameters:  parameters,
				},
			})
		}
	}

	return definitions, nil
}

func convertToolParameters(declaration *genai.FunctionDeclaration) (ollamaapi.JSONSchema, error) {
	if declaration.ParametersJsonSchema != nil {
		return genericSchemaToJSONSchema(declaration.ParametersJsonSchema)
	}

	if declaration.Parameters != nil {
		return schemaToJSONSchema(declaration.Parameters), nil
	}

	return ollamaapi.JSONSchema{
		"type":       "object",
		"properties": map[string]any{},
	}, nil
}

func convertThinking(config *genai.ThinkingConfig) *ollamaapi.ThinkOption {
	if config == nil {
		return nil
	}

	switch config.ThinkingLevel {
	case genai.ThinkingLevelHigh:
		return ollamaapi.ThinkAtLevel(ollamaapi.ThinkLevelHigh)
	case genai.ThinkingLevelMedium:
		return ollamaapi.ThinkAtLevel(ollamaapi.ThinkLevelMedium)
	case genai.ThinkingLevelLow, genai.ThinkingLevelMinimal:
		return ollamaapi.ThinkAtLevel(ollamaapi.ThinkLevelLow)
	}

	if config.IncludeThoughts {
		return ollamaapi.ThinkEnabled(true)
	}

	return ollamaapi.ThinkEnabled(false)
}

func contentToChatMessages(content *genai.Content, defaultRole ollamaapi.ChatRole) ([]ollamaapi.ChatMessage, error) {
	if content == nil {
		return nil, nil
	}

	current := ollamaapi.ChatMessage{Role: defaultRole}
	var messages []ollamaapi.ChatMessage

	flushCurrent := func() {
		if current.Content == "" && len(current.Images) == 0 && len(current.ToolCalls) == 0 {
			return
		}
		messages = append(messages, current)
		current = ollamaapi.ChatMessage{Role: defaultRole}
	}

	for _, part := range content.Parts {
		if part == nil {
			continue
		}

		if part.FunctionResponse != nil {
			flushCurrent()
			toolMessage, err := functionResponseToToolMessage(part.FunctionResponse)
			if err != nil {
				return nil, err
			}
			messages = append(messages, toolMessage)
			continue
		}

		if part.FunctionCall != nil {
			current.Role = ollamaapi.ChatRoleAssistant
			current.ToolCalls = append(current.ToolCalls, ollamaapi.ToolCall{
				Function: ollamaapi.ToolCallFunction{
					Name:      part.FunctionCall.Name,
					Arguments: part.FunctionCall.Args,
				},
			})
		}

		text, err := partToText(part)
		if err != nil {
			return nil, err
		}
		if text != "" {
			current.Content = joinContent(current.Content, text)
		}

		images, err := partToImages(part)
		if err != nil {
			return nil, err
		}
		current.Images = append(current.Images, images...)
	}

	flushCurrent()
	return messages, nil
}

func functionResponseToToolMessage(response *genai.FunctionResponse) (ollamaapi.ChatMessage, error) {
	payload := map[string]any{}
	if response.Name != "" {
		payload["name"] = response.Name
	}
	if response.Response != nil {
		payload["response"] = response.Response
	}
	if len(response.Parts) > 0 {
		parts, err := functionResponsePartsToAny(response.Parts)
		if err != nil {
			return ollamaapi.ChatMessage{}, err
		}
		payload["parts"] = parts
	}
	if response.ID != "" {
		payload["id"] = response.ID
	}
	if response.WillContinue != nil {
		payload["will_continue"] = *response.WillContinue
	}

	content := "{}"
	if len(payload) > 0 {
		data, err := json.Marshal(payload)
		if err != nil {
			return ollamaapi.ChatMessage{}, fmt.Errorf("marshal function response: %w", err)
		}
		content = string(data)
	}

	return ollamaapi.ChatMessage{
		Role:    ollamaapi.ChatRoleTool,
		Content: content,
	}, nil
}

func functionResponsePartsToAny(parts []*genai.FunctionResponsePart) ([]map[string]any, error) {
	converted := make([]map[string]any, 0, len(parts))
	for _, part := range parts {
		if part == nil {
			continue
		}
		entry := map[string]any{}
		if part.InlineData != nil {
			entry["inline_data"] = map[string]any{
				"mime_type": part.InlineData.MIMEType,
				"data":      base64.StdEncoding.EncodeToString(part.InlineData.Data),
			}
		}
		if part.FileData != nil {
			entry["file_data"] = map[string]any{
				"file_uri":  part.FileData.FileURI,
				"mime_type": part.FileData.MIMEType,
			}
		}
		converted = append(converted, entry)
	}
	return converted, nil
}

func partToText(part *genai.Part) (string, error) {
	segments := []string{}

	if part.Text != "" {
		segments = append(segments, part.Text)
	}

	if part.ExecutableCode != nil && part.ExecutableCode.Code != "" {
		segments = append(segments, part.ExecutableCode.Code)
	}

	if part.CodeExecutionResult != nil && part.CodeExecutionResult.Output != "" {
		segments = append(segments, part.CodeExecutionResult.Output)
	}

	if part.ToolCall != nil {
		data, err := json.Marshal(part.ToolCall)
		if err != nil {
			return "", fmt.Errorf("marshal tool call: %w", err)
		}
		segments = append(segments, string(data))
	}

	if part.ToolResponse != nil {
		data, err := json.Marshal(part.ToolResponse)
		if err != nil {
			return "", fmt.Errorf("marshal tool response: %w", err)
		}
		segments = append(segments, string(data))
	}

	if part.FileData != nil && !isImageMIMEType(part.FileData.MIMEType) {
		segments = append(segments, part.FileData.FileURI)
	}

	if part.InlineData != nil && !isImageMIMEType(part.InlineData.MIMEType) {
		segments = append(segments, base64.StdEncoding.EncodeToString(part.InlineData.Data))
	}

	return strings.Join(segments, "\n"), nil
}

func partToImages(part *genai.Part) ([]string, error) {
	if part == nil {
		return nil, nil
	}

	var images []string
	if part.InlineData != nil && isImageMIMEType(part.InlineData.MIMEType) {
		images = append(images, base64.StdEncoding.EncodeToString(part.InlineData.Data))
	}

	if part.FileData != nil && isImageMIMEType(part.FileData.MIMEType) {
		image, err := imageFromFileData(part.FileData)
		if err != nil {
			return nil, err
		}
		if image != "" {
			images = append(images, image)
		}
	}

	return images, nil
}

func imageFromFileData(fileData *genai.FileData) (string, error) {
	if fileData == nil || fileData.FileURI == "" {
		return "", nil
	}

	path := strings.TrimPrefix(fileData.FileURI, "file://")
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read image file %q: %w", fileData.FileURI, err)
	}

	return base64.StdEncoding.EncodeToString(data), nil
}

func mapContentRole(content *genai.Content) ollamaapi.ChatRole {
	if content == nil {
		return ollamaapi.ChatRoleUser
	}

	switch strings.ToLower(content.Role) {
	case "system":
		return ollamaapi.ChatRoleSystem
	case "assistant", "model":
		return ollamaapi.ChatRoleAssistant
	case "tool":
		return ollamaapi.ChatRoleTool
	default:
		return ollamaapi.ChatRoleUser
	}
}

func genericSchemaToJSONSchema(value any) (ollamaapi.JSONSchema, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}

	var schema ollamaapi.JSONSchema
	if err := json.Unmarshal(data, &schema); err != nil {
		return nil, err
	}

	return schema, nil
}

func schemaToJSONSchema(schema *genai.Schema) ollamaapi.JSONSchema {
	if schema == nil {
		return nil
	}

	converted := ollamaapi.JSONSchema{}

	if len(schema.AnyOf) > 0 {
		anyOf := make([]any, 0, len(schema.AnyOf))
		for _, item := range schema.AnyOf {
			anyOf = append(anyOf, schemaToJSONSchema(item))
		}
		converted["anyOf"] = anyOf
	}
	if schema.Default != nil {
		converted["default"] = schema.Default
	}
	if schema.Description != "" {
		converted["description"] = schema.Description
	}
	if len(schema.Enum) > 0 {
		converted["enum"] = schema.Enum
	}
	if schema.Example != nil {
		converted["example"] = schema.Example
	}
	if schema.Format != "" {
		converted["format"] = schema.Format
	}
	if schema.Items != nil {
		converted["items"] = schemaToJSONSchema(schema.Items)
	}
	if schema.MaxItems != nil {
		converted["maxItems"] = *schema.MaxItems
	}
	if schema.MaxLength != nil {
		converted["maxLength"] = *schema.MaxLength
	}
	if schema.MaxProperties != nil {
		converted["maxProperties"] = *schema.MaxProperties
	}
	if schema.Maximum != nil {
		converted["maximum"] = *schema.Maximum
	}
	if schema.MinItems != nil {
		converted["minItems"] = *schema.MinItems
	}
	if schema.MinLength != nil {
		converted["minLength"] = *schema.MinLength
	}
	if schema.MinProperties != nil {
		converted["minProperties"] = *schema.MinProperties
	}
	if schema.Minimum != nil {
		converted["minimum"] = *schema.Minimum
	}
	if schema.Nullable != nil {
		converted["nullable"] = *schema.Nullable
	}
	if schema.Pattern != "" {
		converted["pattern"] = schema.Pattern
	}
	if len(schema.Properties) > 0 {
		properties := map[string]any{}
		for name, property := range schema.Properties {
			properties[name] = schemaToJSONSchema(property)
		}
		converted["properties"] = properties
	}
	if len(schema.PropertyOrdering) > 0 {
		converted["propertyOrdering"] = schema.PropertyOrdering
	}
	if len(schema.Required) > 0 {
		converted["required"] = schema.Required
	}
	if schema.Title != "" {
		converted["title"] = schema.Title
	}
	if schema.Type != "" {
		converted["type"] = schemaTypeToString(schema.Type)
	}

	return converted
}

func schemaTypeToString(schemaType genai.Type) string {
	switch schemaType {
	case genai.TypeString:
		return "string"
	case genai.TypeNumber:
		return "number"
	case genai.TypeInteger:
		return "integer"
	case genai.TypeBoolean:
		return "boolean"
	case genai.TypeArray:
		return "array"
	case genai.TypeObject:
		return "object"
	case genai.TypeNULL:
		return "null"
	default:
		return strings.ToLower(string(schemaType))
	}
}

func joinContent(current string, next string) string {
	if current == "" {
		return next
	}
	if next == "" {
		return current
	}
	return current + "\n" + next
}

func isImageMIMEType(mimeType string) bool {
	return strings.HasPrefix(strings.ToLower(mimeType), "image/")
}
