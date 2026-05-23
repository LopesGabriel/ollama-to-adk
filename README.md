# ollama-to-adk

`ollama-to-adk` is a small Go adapter that makes an [Ollama](https://ollama.com/) chat model look like an ADK `model.LLM`.

If you are building with Google's ADK but want inference to run through a local or self-hosted Ollama server, this package bridges the two APIs and translates requests and responses for you.

## What it does

- Exposes a simple `NewOllamaModel` constructor that returns an ADK-compatible `model.LLM`
- Sends ADK `GenerateContent` requests to Ollama's `/api/chat` endpoint
- Supports both non-streaming and streaming generation
- Maps system instructions, multi-turn chat history, JSON output formats, function tools, tool responses, thinking config, images, and logprobs
- Preserves useful HTTP response metadata in the returned ADK response

## Requirements

- Go 1.26+
- An Ollama server reachable by this process
- A model already pulled into Ollama, such as `llama3.2`

By default the adapter talks to `http://localhost:11434`. You can override that by passing a different base URL to `NewOllamaModel` or by setting `req.Config.HTTPOptions.BaseURL` on individual requests.

## Installation

```bash
go get github.com/lopesgabriel/ollama-to-adk@latest
```

## Quick start

First, make sure Ollama is running and the model from the example is available:

```bash
ollama pull llama3.2
```

```go
package main

import (
	"context"
	"fmt"
	"log"

	ollamatoadk "github.com/lopesgabriel/ollama-to-adk"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

func main() {
	model, err := ollamatoadk.NewOllamaModel("llama3.2", "")
	if err != nil {
		log.Fatal(err)
	}

	rootAgent, err := llmagent.New(llmagent.Config{
		Name:        "root_agent",
		Model:       model,
		Description: "Hello World agent",
		Instruction: `
		You are a greeter agent, you will greet new contacts.
		`
	})
	if err != nil {
		log.FatalF("Failed to create agent: %w", err)
	}

	config := &launcher.Config{
		AgentLoader: agent.NewSingleLoader(rootAgent),
	}

	l := full.NewLauncher()
	if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
		log.FatalF("Run failed: %w (syntax: %s)", err, l.CommandLineSyntax())
	}
}
```

## How it fits together

The package is intentionally small:

- `llm_implementation.go` is the public entry point and ADK-facing orchestration layer
- `internal/converter` maps ADK / GenAI inputs into Ollama chat requests
- `internal/client` performs the HTTP call to Ollama and maps responses back into ADK `LLMResponse` values
- `internal/ollamaapi` defines the local wire types for Ollama request and response payloads

That separation makes it easier to change transport details or request mapping without growing the public API.

## Supported request/response mapping

The adapter currently covers the pieces that matter most when using Ollama through ADK:

- Standard chat turns with role conversion
- System instructions
- Streaming and non-streaming responses
- JSON response mode and schema-based structured output
- Function tool declarations and tool-call responses
- Thinking configuration
- Image parts
- Logprobs and finish reason mapping
- Request-scoped HTTP options such as headers, timeout, extra body fields, and base URL override

Unsupported non-function GenAI tool types fail fast instead of being silently ignored.

## Development

From the repository root:

```bash
go build ./...
go test ./...
```

To run a single test when tests are added:

```bash
go test ./path/to/package -run '^TestName$'
```

At the moment there are no checked-in `_test.go` files, so `go test ./...` is mainly a compile-and-package verification pass.
