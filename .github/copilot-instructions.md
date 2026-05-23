# Copilot instructions for `ollama-to-adk`

## Build and test commands

Use the standard Go module commands from the repository root:

- Build all packages: `go build ./...`
- Run the full test suite: `go test ./...`
- Run a single test: `go test ./path/to/package -run '^TestName$'`

At the moment there are no checked-in `_test.go` files, so `go test ./...` is mainly a compile-and-package verification pass.

## High-level architecture

This repository is a small adapter layer that makes an Ollama chat model look like an ADK `model.LLM`.

- `llm_implementation.go` is the public entry point. `NewOllamaModel` constructs the adapter, sets the default Ollama base URL to `http://localhost:11434`, and injects the library version into the outgoing `user-agent` header.
- `ollamaModel.GenerateContent` is the orchestration point. It normalizes the incoming ADK request, ensures there is always a trailing user turn before calling Ollama, copies HTTP headers into `req.Config.HTTPOptions`, and dispatches to either the streaming or non-streaming path.
- `internal/converter` translates ADK / GenAI request structures into Ollama `/api/chat` payloads. This includes role mapping, system instruction handling, function-call/tool conversion, JSON schema conversion, thinking config mapping, image handling, and serialization of tool responses into Ollama tool-role messages.
- `internal/client` owns the HTTP call to `/api/chat` and converts Ollama responses back into ADK `model.LLMResponse` values. It handles both one-shot JSON responses and streaming JSON events, maps finish reasons and logprobs, and stores HTTP metadata in `CustomMetadata`.
- `internal/ollamaapi` defines the local wire-model types for Ollama requests and responses, including custom `MarshalJSON` behavior for fields whose wire shape is not a plain struct (`format`, `think`, `keep_alive`, `stop`, and `options`).

When changing behavior, keep the end-to-end translation path in sync: ADK request in `llm_implementation.go` -> Ollama request mapping in `internal/converter` -> HTTP transport and response mapping in `internal/client` -> Ollama wire types in `internal/ollamaapi`.

## Key repository-specific conventions

- The root package is intentionally thin. Most changes belong in `internal/converter` or `internal/client`; `llm_implementation.go` should mainly stay focused on ADK-facing orchestration.
- Unsupported GenAI tool types fail fast in `convertTools` instead of being silently dropped. Only function declarations are translated into Ollama tool definitions.
- Image parts are sent to Ollama as base64 strings and are reconstructed from base64 on the way back. Non-image file data is treated as text/URI content instead of image content.
- Tool responses are converted into Ollama `tool` messages by JSON-encoding the response payload. If you change tool-call handling, update both the request-side conversion in `internal/converter` and the response-side mapping in `internal/client`.
- Several Ollama request fields rely on custom JSON marshaling in `internal/ollamaapi`. If you add or rename request options, check whether the wire format is a scalar-or-object union before using plain struct tags.
- Base URL resolution is layered: `config.HTTPOptions.BaseURL` overrides the client base URL, and the client falls back to `http://localhost:11434` when neither is set.
