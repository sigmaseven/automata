package models

type Model interface {
	Query(request ModelRequest) (ModelResponse, error)
	GenerateEmbeddings(request *ModelRequest) (*ModelResponse, error)
}

type ModelRequest interface {
}

type ModelResponse interface {
}
