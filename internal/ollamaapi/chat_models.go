package ollamaapi

import (
	"encoding/json"
	"time"
)

type ChatRequest struct {
	Model       string           `json:"model"`
	Messages    []ChatMessage    `json:"messages"`
	Tools       []ToolDefinition `json:"tools,omitempty"`
	Format      *ChatFormat      `json:"format,omitempty"`
	Options     *ModelOptions    `json:"options,omitempty"`
	Stream      *bool            `json:"stream,omitempty"`
	Think       *ThinkOption     `json:"think,omitempty"`
	KeepAlive   *KeepAlive       `json:"keep_alive,omitempty"`
	Logprobs    *bool            `json:"logprobs,omitempty"`
	TopLogprobs *int             `json:"top_logprobs,omitempty"`
}

type ChatRole string

const (
	ChatRoleSystem    ChatRole = "system"
	ChatRoleUser      ChatRole = "user"
	ChatRoleAssistant ChatRole = "assistant"
	ChatRoleTool      ChatRole = "tool"
)

type ChatMessage struct {
	Role      ChatRole   `json:"role"`
	Content   string     `json:"content"`
	Images    []string   `json:"images,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type ToolType string

const (
	ToolTypeFunction ToolType = "function"
)

type ToolDefinition struct {
	Type     ToolType     `json:"type"`
	Function ToolFunction `json:"function"`
}

type ToolFunction struct {
	Name        string     `json:"name"`
	Description string     `json:"description,omitempty"`
	Parameters  JSONSchema `json:"parameters"`
}

type ToolCall struct {
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Arguments   map[string]any `json:"arguments,omitempty"`
}

type JSONSchema map[string]any

type ChatFormat struct {
	JSON   bool
	Schema JSONSchema
}

func JSONResponseFormat() *ChatFormat {
	return &ChatFormat{JSON: true}
}

func SchemaResponseFormat(schema JSONSchema) *ChatFormat {
	return &ChatFormat{Schema: schema}
}

func (f ChatFormat) MarshalJSON() ([]byte, error) {
	if f.JSON {
		return json.Marshal("json")
	}
	if f.Schema == nil {
		return []byte("null"), nil
	}
	return json.Marshal(f.Schema)
}

type ThinkLevel string

const (
	ThinkLevelHigh   ThinkLevel = "high"
	ThinkLevelMedium ThinkLevel = "medium"
	ThinkLevelLow    ThinkLevel = "low"
)

type ThinkOption struct {
	Enabled *bool
	Level   ThinkLevel
}

func ThinkEnabled(enabled bool) *ThinkOption {
	return &ThinkOption{Enabled: &enabled}
}

func ThinkAtLevel(level ThinkLevel) *ThinkOption {
	return &ThinkOption{Level: level}
}

func (t ThinkOption) MarshalJSON() ([]byte, error) {
	if t.Level != "" {
		return json.Marshal(t.Level)
	}
	if t.Enabled != nil {
		return json.Marshal(*t.Enabled)
	}
	return []byte("null"), nil
}

type KeepAlive struct {
	Duration string
	Seconds  *float64
}

func KeepAliveDuration(duration string) *KeepAlive {
	return &KeepAlive{Duration: duration}
}

func KeepAliveNumber(value float64) *KeepAlive {
	return &KeepAlive{Seconds: &value}
}

func (k KeepAlive) MarshalJSON() ([]byte, error) {
	if k.Duration != "" {
		return json.Marshal(k.Duration)
	}
	if k.Seconds != nil {
		return json.Marshal(*k.Seconds)
	}
	return []byte("null"), nil
}

type StopSequences struct {
	Single string
	Many   []string
}

func StopSequence(sequence string) *StopSequences {
	return &StopSequences{Single: sequence}
}

func StopSequencesList(sequences ...string) *StopSequences {
	return &StopSequences{Many: sequences}
}

func (s StopSequences) MarshalJSON() ([]byte, error) {
	if s.Single != "" {
		return json.Marshal(s.Single)
	}
	if len(s.Many) > 0 {
		return json.Marshal(s.Many)
	}
	return []byte("null"), nil
}

type ModelOptions struct {
	Seed        *int64         `json:"seed,omitempty"`
	Temperature *float64       `json:"temperature,omitempty"`
	TopK        *int64         `json:"top_k,omitempty"`
	TopP        *float64       `json:"top_p,omitempty"`
	MinP        *float64       `json:"min_p,omitempty"`
	Stop        *StopSequences `json:"stop,omitempty"`
	NumCtx      *int64         `json:"num_ctx,omitempty"`
	NumPredict  *int64         `json:"num_predict,omitempty"`
	Extra       map[string]any `json:"-"`
}

func (o ModelOptions) MarshalJSON() ([]byte, error) {
	base := map[string]any{}

	if o.Seed != nil {
		base["seed"] = *o.Seed
	}
	if o.Temperature != nil {
		base["temperature"] = *o.Temperature
	}
	if o.TopK != nil {
		base["top_k"] = *o.TopK
	}
	if o.TopP != nil {
		base["top_p"] = *o.TopP
	}
	if o.MinP != nil {
		base["min_p"] = *o.MinP
	}
	if o.Stop != nil {
		base["stop"] = o.Stop
	}
	if o.NumCtx != nil {
		base["num_ctx"] = *o.NumCtx
	}
	if o.NumPredict != nil {
		base["num_predict"] = *o.NumPredict
	}
	for key, value := range o.Extra {
		base[key] = value
	}

	return json.Marshal(base)
}

type ChatResponse struct {
	Model              string               `json:"model,omitempty"`
	CreatedAt          time.Time            `json:"created_at,omitempty"`
	Message            ChatResponseMessage  `json:"message,omitempty"`
	Done               bool                 `json:"done,omitempty"`
	DoneReason         string               `json:"done_reason,omitempty"`
	TotalDuration      int64                `json:"total_duration,omitempty"`
	LoadDuration       int64                `json:"load_duration,omitempty"`
	PromptEvalCount    int32                `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration int64                `json:"prompt_eval_duration,omitempty"`
	EvalCount          int32                `json:"eval_count,omitempty"`
	EvalDuration       int64                `json:"eval_duration,omitempty"`
	Logprobs           []Logprob            `json:"logprobs,omitempty"`
}

type ChatResponseMessage struct {
	Role      ChatRole   `json:"role,omitempty"`
	Content   string     `json:"content,omitempty"`
	Thinking  string     `json:"thinking,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	Images    []string   `json:"images,omitempty"`
}

type Logprob struct {
	Token       string         `json:"token,omitempty"`
	Logprob     float64        `json:"logprob,omitempty"`
	Bytes       []int          `json:"bytes,omitempty"`
	TopLogprobs []TokenLogprob `json:"top_logprobs,omitempty"`
}

type TokenLogprob struct {
	Token   string  `json:"token,omitempty"`
	Logprob float64 `json:"logprob,omitempty"`
	Bytes   []int   `json:"bytes,omitempty"`
}

type ErrorResponse struct {
	Error string `json:"error,omitempty"`
}