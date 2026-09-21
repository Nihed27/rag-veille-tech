from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

DOCS_DIR = Path("data/docs")


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """Lit tous les PDF du dossier et renvoie un Document par page."""
    documents = []
    for pdf_path in sorted(docs_dir.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_path.name, "page": page_number},
                    )
                )
    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"{len(docs)} pages chargées")
    print(docs[0].metadata)
    print(docs[0].page_content[:200])