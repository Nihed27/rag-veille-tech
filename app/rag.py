from app.llm import get_llm
from app.vectorstore import get_vectorstore

PROMPT = """Tu es un assistant qui répond à des questions à partir d'extraits de cours.
Réponds uniquement à partir des extraits ci-dessous. Si la réponse n'y figure pas, dis-le clairement.
Cite tes sources à la fin de chaque affirmation, en recopiant exactement le nom du fichier et la page indiqués entre crochets avant l'extrait, par exemple (Chapitre5.pdf, page 3).

Extraits :
{contexte}

Question : {question}
Réponse :"""


def answer(question: str, k: int = 4) -> str:
    """Cherche les k chunks les plus proches puis fait rédiger la réponse par le LLM."""
    docs = get_vectorstore().similarity_search(question, k=k)
    contexte = "\n\n".join(
        f"[{d.metadata['source']}, page {d.metadata['page']}]\n{d.page_content}"
        for d in docs
    )
    prompt = PROMPT.format(contexte=contexte, question=question)
    return get_llm().invoke(prompt).content


if __name__ == "__main__":
    print(answer("Qu'est-ce que l'ACP ?"))