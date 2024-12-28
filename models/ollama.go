package models

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
)

const (
	OllamaRoleSystem    string = "system"
	OllamaRoleUser             = "user"
	OllamaRoleAssistant        = "assistant"
	OllamaRoleToolbox          = "tool"
)

type OllamaModel struct {
	model         string
	baseUrl       string
	isChatSession bool
}

type OllamaGenerateTextRequest struct {
	Model     string         `json:"model"`
	Prompt    string         `json:"prompt"`
	Suffix    string         `json:"suffix,omitempty"`
	Format    string         `json:"format,omitempty"`
	Stream    bool           `json:"stream"`
	Raw       bool           `json:"raw"`
	KeepAlive string         `json:"keep_alive,omitempty"`
	Template  string         `json:"template,omitempty"`
	Options   map[string]any `json:"options,omitempty"`
}

func NewOllamaGenerateTextRequest(model string) *OllamaGenerateTextRequest {
	request := &OllamaGenerateTextRequest{
		Model:     model,
		Prompt:    "",
		Suffix:    "",
		Format:    "",
		Stream:    false,
		Raw:       false,
		KeepAlive: "",
		Template:  "",
		Options:   make(map[string]any),
	}

	request.Options["temperature"] = 0.0
	return request
}

type OllamaGenerateTextResponse struct {
	Model              string `json:"model,omitempty"`
	Response           string `json:"response,omitempty"`
	CreatedAt          string `json:"created_at,omitempty"`
	Done               bool   `json:"done,omitempty"`
	PromptEvalCount    uint   `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration uint   `json:"prompt_eval_duration,omitempty"`
	EvalCount          uint   `json:"eval_count,omitempty"`
	EvalDuration       uint   `json:"eval_duration,omitempty"`
}

func NewOllamaGenerateTextResponse() *OllamaGenerateTextResponse {
	response := &OllamaGenerateTextResponse{
		Model:              "",
		Response:           "",
		CreatedAt:          "",
		Done:               false,
		PromptEvalCount:    0,
		PromptEvalDuration: 0,
		EvalCount:          0,
		EvalDuration:       0,
	}

	return response
}

type OllamaChatMessage struct {
	Role      string   `json:"role"`
	Content   string   `json:"content"`
	Images    []string `json:"images,omitempty"`
	ToolCalls []string `json:"tool_calls,omitempty"`
}

func NewOllamaChatMessage(role string, content string) *OllamaChatMessage {
	return &OllamaChatMessage{
		Role:    role,
		Content: content,
	}
}

type OllamaChatRequest struct {
	Model     string              `json:"model"`
	Messages  []OllamaChatMessage `json:"messages,omitempty"`
	Tools     []string            `json:"tools,omitempty"`
	Format    string              `json:"format,omitempty"`
	Options   map[string]any      `json:"options,omitempty"`
	Stream    bool                `json:"stream"`
	KeepAlive string              `json:"keep_alive,omitempty"`
}

func NewOllamaChatRequest(model string, messages []OllamaChatMessage) *OllamaChatRequest {
	return &OllamaChatRequest{
		Model:     model,
		Messages:  messages,
		Tools:     []string{},
		Format:    "",
		Options:   make(map[string]any),
		Stream:    false,
		KeepAlive: "5m",
	}
}

type OllamaChatResponse struct {
	Model              string            `json:"model,omitempty"`
	Message            OllamaChatMessage `json:"message,omitempty"`
	CreatedAt          string            `json:"created_at,omitempty"`
	Done               bool              `json:"done,omitempty"`
	PromptEval         string            `json:"prompt_eval,omitempty"`
	PromptEvalDuration uint              `json:"prompt_eval_duration,omitempty"`
	EvalCount          uint              `json:"eval_count,omitempty"`
	EvalDuration       uint              `json:"eval_duration,omitempty"`
	TotalDuration      uint              `json:"total_duration,omitempty"`
	LoadDuration       uint              `json:"load_duration,omitempty"`
	DoneReason         string            `json:"done_reason,omitempty"`
}

func NewOllamaChatResponse(model string) *OllamaChatResponse {
	return &OllamaChatResponse{
		Model: model,
	}
}

type OllamaEmbeddingRequest struct {
	Model     string         `json:"model"`
	Input     []string       `json:"input"`
	Truncate  bool           `json:"truncate,omitempty"`
	Options   map[string]any `json:"options,omitempty"`
	KeepAlive string         `json:"keep_alive,omitempty"`
}

func NewOllamaEmbeddingRequest(model string, input []string) *OllamaEmbeddingRequest {
	return &OllamaEmbeddingRequest{
		Model: model,
		Input: input,
	}
}

type OllamaEmbeddingResponse struct {
	Model      string      `json:"model,omitempty"`
	Embeddings [][]float64 `json:"embeddings,omitempty"`
}

func NewOllamaEmbeddingResponse() *OllamaEmbeddingResponse {
	response := &OllamaEmbeddingResponse{}
	return response
}

func NewOllamaModel(baseUrl string, model string) *OllamaModel {
	return &OllamaModel{
		model:   model,
		baseUrl: baseUrl,
	}
}

func (model *OllamaModel) Query(request ModelRequest) (ModelResponse, error) {
	switch request.(type) {
	case *OllamaGenerateTextRequest:
		response, err := model.Generate(request.(*OllamaGenerateTextRequest))

		if err != nil {
			return nil, err
		}

		return response, nil

	case *OllamaChatRequest:
		response, err := model.Chat(request.(*OllamaChatRequest))

		if err != nil {
			return nil, err
		}

		return response, nil

	default:
		return nil, errors.New("invalid request type")
	}
}

func (model *OllamaModel) Generate(request *OllamaGenerateTextRequest) (*OllamaGenerateTextResponse, error) {
	url := fmt.Sprintf("%s/api/generate", model.baseUrl)

	bodyContent, err := json.Marshal(request)

	if err != nil {
		return nil, err
	}

	webRequest, err := http.NewRequest("POST", url, bytes.NewBuffer(bodyContent))

	if err != nil {
		return nil, err
	}

	webRequest.Header.Set("Content-Type", "application/json")
	webRequest.Header.Set("Accept", "application/json")
	webRequest.Header.Set("Accept-Charset", "utf-8")

	client := &http.Client{}

	response, err := client.Do(webRequest)

	if err != nil {
		return nil, err
	}

	if response.StatusCode != 200 {
		return nil, errors.New("HTTP error " + response.Status + "received")
	}

	responseBody, err := io.ReadAll(response.Body)

	err = response.Body.Close()
	if err != nil {
		return nil, err
	}

	chatResponse := NewOllamaGenerateTextResponse()

	err = json.Unmarshal(responseBody, chatResponse)

	if err != nil {
		return nil, err
	}

	return chatResponse, nil
}

func (model *OllamaModel) Chat(request *OllamaChatRequest) (*OllamaChatResponse, error) {
	url := fmt.Sprintf("%s/api/chat", model.baseUrl)

	bodyContent, err := json.Marshal(request)

	if err != nil {
		return nil, err
	}

	webRequest, err := http.NewRequest("POST", url, bytes.NewBuffer(bodyContent))

	if err != nil {
		return nil, err
	}

	webRequest.Header.Set("Content-Type", "application/json")
	webRequest.Header.Set("Accept", "application/json")
	webRequest.Header.Set("Accept-Charset", "utf-8")

	client := &http.Client{}

	response, err := client.Do(webRequest)

	if err != nil {
		return nil, err
	}

	if response.StatusCode != 200 {
		return nil, errors.New("HTTP error " + response.Status + "received")
	}

	responseBody, err := io.ReadAll(response.Body)

	err = response.Body.Close()
	if err != nil {
		return nil, err
	}

	chatResponse := NewOllamaChatResponse(model.model)

	err = json.Unmarshal(responseBody, chatResponse)

	if err != nil {
		return nil, err
	}

	return chatResponse, nil
}

func (model *OllamaModel) GenerateEmbeddings(request *OllamaEmbeddingRequest) (*OllamaEmbeddingResponse, error) {
	url := fmt.Sprintf("%s/api/embed", model.baseUrl)

	bodyContent, err := json.Marshal(request)

	if err != nil {
		return nil, err
	}

	webRequest, err := http.NewRequest("POST", url, bytes.NewBuffer(bodyContent))

	if err != nil {
		return nil, err
	}

	webRequest.Header.Set("Content-Type", "application/json")
	webRequest.Header.Set("Accept", "application/json")
	webRequest.Header.Set("Accept-Charset", "utf-8")

	client := &http.Client{}

	response, err := client.Do(webRequest)

	if err != nil {
		return nil, err
	}

	if response.StatusCode != 200 {
		return nil, errors.New("HTTP error " + response.Status + " received")
	}

	responseBody, err := io.ReadAll(response.Body)

	err = response.Body.Close()
	if err != nil {
		return nil, err
	}

	embedResponse := NewOllamaEmbeddingResponse()

	err = json.Unmarshal(responseBody, embedResponse)

	if err != nil {
		return nil, err
	}

	return embedResponse, nil
}
