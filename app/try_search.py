from app.vectorstore import get_vectorstore

QUESTIONS = [
    "Qu'est-ce que l'ACP ?",
    "À quoi sert l'analyse factorielle des correspondances ?",
    "Comment choisir le nombre de classes en classification ?",
    "Qu'est-ce que l'inertie ?",
]

if __name__ == "__main__":
    store = get_vectorstore()
    for question in QUESTIONS:
        print("===", question)
        for doc in store.similarity_search(question, k=3):
            print(doc.metadata["source"], "- page", doc.metadata["page"])
            print("   ", doc.page_content[:120].replace("\n", " "))