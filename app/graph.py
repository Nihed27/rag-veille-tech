from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.llm import get_llm
from app.rag import PROMPT
from app.vectorstore import get_vectorstore

MAX_ATTEMPTS = 2
REFUS = "Je ne trouve pas la réponse dans les documents."
CRITIQUE_PROMPT = """Tu es un relecteur strict.
Extraits :
{contexte}

Réponse à vérifier :
{answer}

Vérifie que chaque affirmation de la réponse est appuyée par les extraits, et que les sources citées (fichier, page) correspondent bien aux extraits.
Réponds par le seul mot OK si tout est correct. Sinon réponds par NON suivi d'une courte explication du problème."""


class State(TypedDict, total=False):
    question: str
    docs: list[Document]
    answer: str
    verdict: str
    feedback: str
    attempts: int


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata['source']}, page {d.metadata['page']}]\n{d.page_content}"
        for d in docs
    )


def retriever(state: State) -> dict:
    """Agent 1 : cherche les chunks les plus proches de la question."""
    docs = get_vectorstore().similarity_search(state["question"], k=4)
    return {"docs": docs}


def redacteur(state: State) -> dict:
    """Agent 2 : rédige la réponse (et la corrige si la critique a signalé un problème)."""
    prompt = PROMPT.format(
        contexte=format_context(state["docs"]), question=state["question"]
    )
    if state.get("feedback"):
        prompt += (
            f"\n\nUne relecture a signalé ce problème dans ta réponse précédente : "
            f"{state['feedback']}\nCorrige-le en t'appuyant uniquement sur les extraits."
        )
    return {
        "answer": get_llm().invoke(prompt).content,
        "attempts": state.get("attempts", 0) + 1,
    }


def critique(state: State) -> dict:
    """Agent 3 : vérifie que la réponse est bien appuyée par les extraits."""
    if state["answer"].strip().startswith(REFUS):
        return {"verdict": "ok", "feedback": ""}
    prompt = CRITIQUE_PROMPT.format(
        contexte=format_context(state["docs"]), answer=state["answer"]
    )
    verdict = get_llm().invoke(prompt).content.strip()
    ok = verdict.upper().startswith("OK")
    return {"verdict": "ok" if ok else "ko", "feedback": "" if ok else verdict}

def route_after_critique(state: State) -> str:
    if state["verdict"] == "ok" or state["attempts"] >= MAX_ATTEMPTS:
        return "fin"
    return "redacteur"


def build_graph():
    graph = StateGraph(State)
    graph.add_node("retriever", retriever)
    graph.add_node("redacteur", redacteur)
    graph.add_node("critique", critique)
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "redacteur")
    graph.add_edge("redacteur", "critique")
    graph.add_conditional_edges(
        "critique", route_after_critique, {"redacteur": "redacteur", "fin": END}
    )
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    for question in ["Qu'est-ce que l'ACP ?", "Quelle est la capitale de la Tunisie ?"]:
        result = app.invoke({"question": question})
        print("===", question)
        print(result["answer"])
        print("verdict:", result["verdict"], "| tentatives:", result["attempts"])
        print("feedback:", result.get("feedback"))