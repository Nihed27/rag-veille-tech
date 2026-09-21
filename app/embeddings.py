from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    """Adapte le modèle d'embedding local de ChromaDB pour LangChain."""

    def __init__(self):
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


if __name__ == "__main__":
    embeddings = LocalEmbeddings()
    vector = embeddings.embed_query("Qu'est-ce qu'une ACP ?")
    print(len(vector), vector[:5])