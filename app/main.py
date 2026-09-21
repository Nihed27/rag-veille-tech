from fastapi import FastAPI
from pydantic import BaseModel

from app.graph import REFUS, build_graph

app = FastAPI(title="RAG veille tech", version="0.1.0")
graph = build_graph()


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    source: str
    page: int


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    verdict: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = graph.invoke({"question": request.question})
    answer = result["answer"]
    if answer.strip().startswith(REFUS):
        sources = []
    else:
        seen = set()
        sources = []
        for doc in result["docs"]:
            key = (doc.metadata["source"], doc.metadata["page"])
            if key not in seen:
                seen.add(key)
                sources.append(Source(source=key[0], page=key[1]))
    return AskResponse(answer=answer, sources=sources, verdict=result["verdict"])