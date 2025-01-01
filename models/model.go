package models

type Model interface {
	Query(request ModelRequest) (ModelResponse, error)
	GenerateEmbeddings(request *ModelRequest) (*ModelResponse, error)
}

type ModelRequest interface {
	GetModel() string
	GetRequest() []string
}

type ModelResponse interface {
	GetModel() string
	GetResponse() string
}
