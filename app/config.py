import os
from pathlib import Path

DOCS_DIR = Path(os.getenv("DOCS_DIR", "data/docs"))
CHROMA_DIR = os.getenv("CHROMA_DIR", "data/chroma")