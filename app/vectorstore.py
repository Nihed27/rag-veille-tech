from langchain_chroma import Chroma

from app.embeddings import OpenRouterEmbeddings
from app.ingest import load_documents, split_documents
from app.config import CHROMA_DIR

COLLECTION_NAME = "cours_analyse_donnees"


def get_vectorstore() -> Chroma:
    """Ouvre la collection ChromaDB (la crée si elle n'existe pas)."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenRouterEmbeddings(),
        persist_directory=CHROMA_DIR,
    )


def build_vectorstore() -> Chroma:
    """Lit les PDF, découpe en chunks, puis (ré)indexe tout dans ChromaDB."""
    chunks = split_documents(load_documents())
    store = get_vectorstore()
    store.reset_collection()
    store.add_documents(chunks)
    return store


if __name__ == "__main__":
    store = build_vectorstore()
    results = store.similarity_search("Qu'est-ce que l'ACP ?", k=2)
    for doc in results:
        print(doc.metadata)
        print(doc.page_content[:200])
        print("---")