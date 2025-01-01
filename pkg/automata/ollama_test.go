package automata

import "testing"

func TestOllamaModel_Generate(t *testing.T) {
	model := NewOllamaModel("http://127.0.0.1:11434", "llama3:latest")

	request := model.NewTextRequest()

	request.Prompt = "please do your best impersonation of a dog"

	response, err := model.Query(request)

	if err != nil {
		t.Error(err)
	}

	if len(response.GetModel()) < 1 {
		t.Error("response model is empty")
	}

	if len(response.GetMessage()) < 1 {
		t.Error("response is empty")
	}
}

func TestOllamaModel_Chat(t *testing.T) {
	model := NewOllamaModel("http://127.0.0.1:11434", "llama3:latest")
	chatMessage := model.NewChatMessage(OllamaRoleUser, "hi there how are you?")
	chatRequest := model.NewChatRequest([]OllamaChatMessage{*chatMessage})

	response, err := model.Chat(chatRequest)

	if err != nil {
		t.Error(err)
	}

	if len(response.Model) < 1 {
		t.Error("response model is empty")
	}

	if len(response.Message.Content) < 1 {
		t.Error("response is empty")
	}
}

func TestOllamaModel_GenerateEmbeddings(t *testing.T) {
	model := NewOllamaModel("http://127.0.0.1:11434", "llama3:latest")
	embedRequest := NewOllamaEmbeddingRequest(model.model, []string{"test"})

	response, err := model.GenerateEmbeddings(embedRequest)

	if err != nil {
		t.Error(err)
		return
	}

	if len(response.Model) == 0 {
		t.Error("No model found")
	}

	if len(response.Embeddings) < 1 {
		t.Error("No embeddings found")
	}
}
