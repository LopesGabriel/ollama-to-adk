package ollamatoadk

import (
	"context"
	"fmt"
	"iter"
	"net/http"
	"runtime"
	"strings"

	"github.com/lopesgabriel/ollama-to-adk/internal/client"
	"github.com/lopesgabriel/ollama-to-adk/internal/version"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

type ollamaModel struct {
	model              string
	versionHeaderValue string
	client             *client.Client
}

func NewOllamaModel(model, baseURL string) (model.LLM, error) {
	if baseURL == "" {
		baseURL = "http://localhost:11434"
	}

	headerValue := fmt.Sprintf("ollama-to-adk/%s gl-go/%s", version.Version,
		strings.TrimPrefix(runtime.Version(), "go"))

	client := client.NewClient(baseURL)
	return &ollamaModel{model: model, versionHeaderValue: headerValue, client: client}, nil
}

func (o *ollamaModel) Name() string {
	return fmt.Sprintf("ollama-%s", o.model)
}

func (o *ollamaModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	o.maybeAppendUserContent(req)
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if req.Config.HTTPOptions == nil {
		req.Config.HTTPOptions = &genai.HTTPOptions{}
	}
	if req.Config.HTTPOptions.Headers == nil {
		req.Config.HTTPOptions.Headers = make(http.Header)
	}
	o.addHeaders(req.Config.HTTPOptions.Headers)

	if stream {
		return o.generateStream(ctx, req)
	}

	return func(yield func(*model.LLMResponse, error) bool) {
		resp, err := o.generate(ctx, req)
		yield(resp, err)
	}
}

func (o *ollamaModel) addHeaders(headers http.Header) {
	headers.Set("user-agent", o.versionHeaderValue)
}

func (o *ollamaModel) maybeAppendUserContent(req *model.LLMRequest) {
	if len(req.Contents) == 0 {
		req.Contents = append(req.Contents, genai.NewContentFromText("Handle the requests as specified in the System Instruction.", "user"))
	}

	if last := req.Contents[len(req.Contents)-1]; last != nil && last.Role != "user" {
		req.Contents = append(req.Contents, genai.NewContentFromText("Continue processing previous requests as instructed. Exit or provide a summary if no more outputs are needed.", "user"))
	}
}

func (o *ollamaModel) generate(ctx context.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
	resp, err := o.client.Generate(ctx, o.model, req.Contents, req.Config)
	if err != nil {
		return nil, fmt.Errorf("failed to call model: %w", err)
	}

	if resp == nil || resp.Content == nil {
		return nil, fmt.Errorf("empty response")
	}

	return resp, nil
}

func (o *ollamaModel) generateStream(ctx context.Context, req *model.LLMRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		for resp, err := range o.client.GenerateStream(ctx, o.model, req.Contents, req.Config) {
			if err != nil {
				if !yield(nil, fmt.Errorf("failed to call model: %w", err)) {
					return
				}
				continue
			}

			if !yield(resp, nil) {
				return
			}
		}
	}
}
