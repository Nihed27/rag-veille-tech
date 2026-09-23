from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader
from app.config import DOCS_DIR



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
def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    min_length: int = 100,
) -> list[Document]:
    """Découpe les pages en chunks et écarte les chunks trop courts."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    return [c for c in chunks if len(c.page_content.strip()) >= min_length]


if __name__ == "__main__":
    pages = load_documents()
    chunks = split_documents(pages)
    print(f"{len(pages)} pages -> {len(chunks)} chunks")
    print(chunks[5].metadata)
    print(chunks[5].page_content)