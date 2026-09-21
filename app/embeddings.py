import os

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()


class LocalEmbeddings(Embeddings):
    """Adapte le modèle d'embedding local de ChromaDB pour LangChain."""

    def __init__(self):
        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vec] for vec in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenRouterEmbeddings(Embeddings):
    """Embeddings multilingues (e5) via l'API OpenRouter."""

    def __init__(self, model: str = "intfloat/multilingual-e5-large"):
        self._client = OpenAIEmbeddings(
            model=model,
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            check_embedding_ctx_length=False,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(f"query: {text}")


if __name__ == "__main__":
    embeddings = OpenRouterEmbeddings()
    vector = embeddings.embed_query("Qu'est-ce qu'une ACP ?")
    print(len(vector), vector[:5])