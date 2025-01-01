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

const (
	ModelTypeOllama string = "ollama"
)

type OllamaModel struct {
	model         string
	baseUrl       string
	isChatSession bool
}

type OllamaTextRequest struct {
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

func (model *OllamaModel) NewTextRequest() *OllamaTextRequest {
	request := &OllamaTextRequest{
		Model:     model.model,
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

func (request *OllamaTextRequest) GetModel() string {
	return request.Model
}

func (request *OllamaTextRequest) SetModel(model string) {
	request.Model = model
}

func (request *OllamaTextRequest) GetMessages() []string {
	return []string{request.Prompt}
}

func (request *OllamaTextRequest) SetPrompt(prompt string) {
	request.Prompt = prompt
}

func (request *OllamaTextRequest) GetSuffix() string {
	return request.Suffix
}

func (request *OllamaTextRequest) SetSuffix(suffix string) {
	request.Suffix = suffix
}

func (request *OllamaTextRequest) GetFormat() string {
	return request.Format
}

func (request *OllamaTextRequest) SetFormat(format string) {
	request.Format = format
}

func (request *OllamaTextRequest) GetStream() bool {
	return request.Stream
}

func (request *OllamaTextRequest) SetStream(stream bool) {
	request.Stream = stream
}

func (request *OllamaTextRequest) GetRaw() bool {
	return request.Raw
}

func (request *OllamaTextRequest) SetRaw(raw bool) {
	request.Raw = raw
}

func (request *OllamaTextRequest) GetKeepAlive() string {
	return request.KeepAlive
}

func (request *OllamaTextRequest) SetKeepAlive(keepAlive string) {
	request.KeepAlive = keepAlive
}

func (request *OllamaTextRequest) GetTemplate() string {
	return request.Template
}

func (request *OllamaTextRequest) SetTemplate(template string) {
	request.Template = template
}

func (request *OllamaTextRequest) GetOptions() map[string]any {
	return request.Options
}

func (request *OllamaTextRequest) SetOptions(options map[string]any) {
	request.Options = options
}

type OllamaTextResponse struct {
	Model              string `json:"model,omitempty"`
	Response           string `json:"response,omitempty"`
	CreatedAt          string `json:"created_at,omitempty"`
	Done               bool   `json:"done,omitempty"`
	PromptEvalCount    uint   `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration uint   `json:"prompt_eval_duration,omitempty"`
	EvalCount          uint   `json:"eval_count,omitempty"`
	EvalDuration       uint   `json:"eval_duration,omitempty"`
}

func (model *OllamaModel) NewTextResponse() *OllamaTextResponse {
	response := &OllamaTextResponse{
		Model:              model.model,
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

func (response *OllamaTextResponse) GetModel() string {
	return response.Model
}

func (response *OllamaTextResponse) SetModel(model string) {
	response.Model = model
}

func (response *OllamaTextResponse) GetMessage() string {
	return response.Response
}

func (response *OllamaTextResponse) GetCreatedAt() string {
	return response.CreatedAt
}

func (response *OllamaTextResponse) SetCreatedAt(createdAt string) {
	response.CreatedAt = createdAt
}

func (response *OllamaTextResponse) GetDone() bool {
	return response.Done
}

func (response *OllamaTextResponse) SetDone(done bool) {
	response.Done = done
}

func (response *OllamaTextResponse) GetPromptEvalCount() uint {
	return response.PromptEvalCount
}

func (response *OllamaTextResponse) SetPromptEvalCount(promptEvalCount uint) {
	response.PromptEvalCount = promptEvalCount
}

func (response *OllamaTextResponse) GetEvalCount() uint {
	return response.EvalCount
}

func (response *OllamaTextResponse) SetEvalCount(evalCount uint) {
	response.EvalCount = evalCount
}
func (response *OllamaTextResponse) GetEvalDuration() uint {
	return response.EvalDuration
}

func (response *OllamaTextResponse) SetEvalDuration(evalDuration uint) {
	response.EvalDuration = evalDuration
}

type OllamaChatMessage struct {
	Role      string   `json:"role"`
	Content   string   `json:"content"`
	Images    []string `json:"images,omitempty"`
	ToolCalls []string `json:"tool_calls,omitempty"`
}

func (model *OllamaModel) NewChatMessage(role string, content string) *OllamaChatMessage {
	return &OllamaChatMessage{
		Role:    role,
		Content: content,
	}
}

func (message *OllamaChatMessage) GetRole() string {
	return message.Role
}

func (message *OllamaChatMessage) SetRole(role string) {
	message.Role = role
}

func (message *OllamaChatMessage) GetContent() string {
	return message.Content
}

func (message *OllamaChatMessage) SetContent(content string) {
	message.Content = content
}

func (message *OllamaChatMessage) GetImages() []string {
	return message.Images
}

func (message *OllamaChatMessage) SetImages(images []string) {
	message.Images = images
}

func (message *OllamaChatMessage) GetToolCalls() []string {
	return message.ToolCalls
}

func (message *OllamaChatMessage) SetToolCalls(toolCalls []string) {
	message.ToolCalls = toolCalls
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

func (model *OllamaModel) NewChatRequest(messages []OllamaChatMessage) *OllamaChatRequest {
	return &OllamaChatRequest{
		Model:     model.model,
		Messages:  messages,
		Tools:     []string{},
		Format:    "",
		Options:   make(map[string]any),
		Stream:    false,
		KeepAlive: "5m",
	}
}

func (request *OllamaChatRequest) GetModel() string {
	return request.Model
}

func (request *OllamaChatRequest) SetModel(model string) {
	request.Model = model
}

func (request *OllamaChatRequest) GetMessages() []string {
	var output []string

	for _, message := range request.Messages {
		output = append(output, message.Content)
	}

	return output
}

func (request *OllamaChatRequest) SetMessages(messages []OllamaChatMessage) {
	request.Messages = messages
}

func (request *OllamaChatRequest) GetOptions() map[string]any {
	return request.Options
}

func (request *OllamaChatRequest) SetOptions(options map[string]any) {
	request.Options = options
}

func (request *OllamaChatRequest) GetStream() bool {
	return request.Stream
}

func (request *OllamaChatRequest) SetStream(stream bool) {
	request.Stream = stream
}

func (request *OllamaChatRequest) GetKeepAlive() string {
	return request.KeepAlive
}

func (request *OllamaChatRequest) SetKeepAlive(keepAlive string) {
	request.KeepAlive = keepAlive
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

func (model *OllamaModel) NewChatResponse() *OllamaChatResponse {
	return &OllamaChatResponse{
		Model: model.model,
	}
}

func (response *OllamaChatResponse) GetModel() string {
	return response.Model
}

func (response *OllamaChatResponse) SetModel(model string) {
	response.Model = model
}

func (response *OllamaChatResponse) GetMessage() string {
	return response.Message.Content
}

func (response *OllamaChatResponse) SetMessage(message OllamaChatMessage) {
	response.Message = message
}

func (response *OllamaChatResponse) GetCreatedAt() string {
	return response.CreatedAt
}

func (response *OllamaChatResponse) SetCreatedAt(createdAt string) {
	response.CreatedAt = createdAt
}

func (response *OllamaChatResponse) GetDone() bool {
	return response.Done
}

func (response *OllamaChatResponse) SetDone(done bool) {
	response.Done = done
}

func (response *OllamaChatResponse) GetPromptEval() string {
	return response.PromptEval
}

func (response *OllamaChatResponse) SetPromptEval(promptEval string) {
	response.PromptEval = promptEval
}

func (response *OllamaChatResponse) GetEvalCount() uint {
	return response.EvalCount
}

func (response *OllamaChatResponse) SetEvalCount(evalCount uint) {
	response.EvalCount = evalCount
}

func (response *OllamaChatResponse) GetEvalDuration() uint {
	return response.EvalDuration
}

func (response *OllamaChatResponse) SetEvalDuration(evalDuration uint) {
	response.EvalDuration = evalDuration
}

func (response *OllamaChatResponse) GetTotalDuration() uint {
	return response.TotalDuration
}

func (response *OllamaChatResponse) SetTotalDuration(totalDuration uint) {
	response.TotalDuration = totalDuration
}

func (response *OllamaChatResponse) GetLoadDuration() uint {
	return response.LoadDuration
}

func (response *OllamaChatResponse) SetLoadDuration(loadDuration uint) {
	response.LoadDuration = loadDuration
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

func (request *OllamaEmbeddingRequest) GetModel() string {
	return request.Model
}

func (request *OllamaEmbeddingRequest) SetModel(model string) {
	request.Model = model
}

func (request *OllamaEmbeddingRequest) GetInput() []string {
	return request.Input
}

func (request *OllamaEmbeddingRequest) SetInput(input []string) {
	request.Input = input
}

func (request *OllamaEmbeddingRequest) GetKeepAlive() string {
	return request.KeepAlive
}

func (request *OllamaEmbeddingRequest) SetKeepAlive(keepAlive string) {
	request.KeepAlive = keepAlive
}

func (request *OllamaEmbeddingRequest) SetOptions(options map[string]any) {
	request.Options = options
}

func (request *OllamaEmbeddingRequest) GetOptions() map[string]any {
	return request.Options
}

type OllamaEmbeddingResponse struct {
	Model      string      `json:"model,omitempty"`
	Embeddings [][]float64 `json:"embeddings,omitempty"`
}

func NewOllamaEmbeddingResponse() *OllamaEmbeddingResponse {
	response := &OllamaEmbeddingResponse{}
	return response
}

func (response *OllamaEmbeddingResponse) GetEmbeddings() [][]float64 {
	return response.Embeddings
}

func (response *OllamaEmbeddingResponse) SetEmbeddings(embeddings [][]float64) {
	response.Embeddings = embeddings
}

func (response *OllamaEmbeddingResponse) GetModel() string {
	return response.Model
}

func (response *OllamaEmbeddingResponse) SetModel(model string) {
	response.Model = model
}

func NewOllamaModel(baseUrl string, model string) *OllamaModel {
	return &OllamaModel{
		model:   model,
		baseUrl: baseUrl,
	}
}

func (model *OllamaModel) Query(request ModelRequest) (ModelResponse, error) {
	switch request.(type) {
	case *OllamaTextRequest:
		response, err := model.Generate(request.(*OllamaTextRequest))

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

func (model *OllamaModel) Generate(request *OllamaTextRequest) (*OllamaTextResponse, error) {
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

	chatResponse := model.NewTextResponse()

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

	chatResponse := model.NewChatResponse()

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
