from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.llm import get_llm
from app.rag import PROMPT
from app.vectorstore import get_vectorstore


class State(TypedDict, total=False):
    question: str
    docs: list[Document]
    answer: str


def retriever(state: State) -> dict:
    """Agent 1 : cherche les chunks les plus proches de la question."""
    docs = get_vectorstore().similarity_search(state["question"], k=4)
    return {"docs": docs}


def redacteur(state: State) -> dict:
    """Agent 2 : rédige la réponse à partir des chunks trouvés."""
    contexte = "\n\n".join(
        f"[{d.metadata['source']}, page {d.metadata['page']}]\n{d.page_content}"
        for d in state["docs"]
    )
    prompt = PROMPT.format(contexte=contexte, question=state["question"])
    return {"answer": get_llm().invoke(prompt).content}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("retriever", retriever)
    graph.add_node("redacteur", redacteur)
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "redacteur")
    graph.add_edge("redacteur", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({"question": "Qu'est-ce que l'ACP ?"})
    print(result["answer"])