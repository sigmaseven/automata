package automata

type Model interface {
	Query(request ModelRequest) (ModelResponse, error)
	GenerateEmbeddings(request *ModelRequest) (*ModelResponse, error)
}

type ModelRequest interface {
	GetModel() string
	GetMessages() []string
}

type ModelResponse interface {
	GetModel() string
	GetMessage() string
}
