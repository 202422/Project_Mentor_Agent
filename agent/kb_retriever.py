"""
KBRetriever: Loads project description and problem-solving book from the data folder,
splits with RecursiveCharacterTextSplitter, embeds with OpenAIEmbeddings, and stores in FAISS.
Allows semantic search via retrieve().
"""
import os
from typing import List, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from utils.settings import settings


# Architecture:
#   src/
#   ├── agent/
#   │   └── kb_retriever.py   ← __file__
#   ├── data/                 ← KB documents live here
#   └── embeddings/           ← FAISS index persisted here
#
# So SRC_DIR = one level up from agent/
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class KBRetriever:
    def __init__(self, data_dir: Optional[str] = None, persist_dir: Optional[str] = None):
        """
        data_dir:    folder with KB documents (PDF/MD/TXT).
                     Defaults to <src>/<settings.DATA_DIR>   → src/data/
        persist_dir: where to store/load FAISS vector store.
                     Defaults to <src>/<settings.PERSIST_DIR> → src/embeddings/
        """
        # Strip any leading ./ or .\ so os.path.join works cleanly
        data_rel    = settings.DATA_DIR.lstrip("./\\")
        persist_rel = settings.PERSIST_DIR.lstrip("./\\")

        self.data_dir    = data_dir    or os.path.join(SRC_DIR, data_rel)
        self.persist_dir = persist_dir or os.path.join(SRC_DIR, persist_rel)

        self.chunk_size    = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

        self.embedding_model = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
        self.vectorstore: Optional[FAISS] = None

        print(f"[KBRetriever] src_dir      : {SRC_DIR}")
        print(f"[KBRetriever] data_dir     : {self.data_dir}")
        print(f"[KBRetriever] persist_dir  : {self.persist_dir}")

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(
                f"[KBRetriever] data_dir not found: {self.data_dir}\n"
                f"  Expected location: src/data/\n"
                f"  Add your PDF/TXT/MD files there and re-run."
            )

        # Load existing vectorstore if available, otherwise build from scratch
        if os.path.isdir(self.persist_dir) and os.listdir(self.persist_dir):
            try:
                self.vectorstore = FAISS.load_local(
                    self.persist_dir,
                    self.embedding_model,
                    allow_dangerous_deserialization=True,
                )
                print(f"[KBRetriever] loaded existing vectorstore from {self.persist_dir}")
            except Exception as e:
                print(f"[KBRetriever] could not load vectorstore ({e}), rebuilding...")
                self.vectorstore = None

        if self.vectorstore is None:
            self._build_vectorstore()

    def _load_documents(self) -> List[Document]:
        """Load all PDFs and text/markdown files from data_dir."""
        docs = []
        files = os.listdir(self.data_dir)
        print(f"[KBRetriever] files found in data_dir: {files}")

        for fname in files:
            fpath = os.path.join(self.data_dir, fname)

            if fname.lower().endswith(".pdf"):
                try:
                    loader = PyPDFLoader(fpath)
                    loaded = loader.load()
                    print(f"[KBRetriever] loaded PDF  : {fname} ({len(loaded)} pages)")
                    docs.extend(loaded)
                except Exception as e:
                    print(f"[KBRetriever] skipping PDF {fname}: {e}")

            elif fname.lower().endswith((".md", ".txt")):
                try:
                    loader = TextLoader(fpath, autodetect_encoding=True)
                    loaded = loader.load()
                    print(f"[KBRetriever] loaded text : {fname} ({len(loaded)} doc(s))")
                    docs.extend(loaded)
                except Exception as e:
                    print(f"[KBRetriever] skipping text {fname}: {e}")

        return docs

    def _build_vectorstore(self):
        """Load, split, embed, and persist the KB."""
        docs = self._load_documents()
        if not docs:
            print("[KBRetriever] no documents found — vectorstore not built.")
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        split_docs = splitter.split_documents(docs)
        print(f"[KBRetriever] split into {len(split_docs)} chunks")

        if not split_docs:
            return

        self.vectorstore = FAISS.from_documents(split_docs, self.embedding_model)

        os.makedirs(self.persist_dir, exist_ok=True)
        self.vectorstore.save_local(self.persist_dir)
        print(f"[KBRetriever] vectorstore saved to {self.persist_dir}")

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[dict]:
        """Return top-k relevant chunks for the query."""
        if self.vectorstore is None:
            raise RuntimeError(
                "[KBRetriever] Vector store not initialized. "
                "Add documents to src/data/ and re-run."
            )

        k = top_k or settings.RETRIEVAL_TOP_K
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        return [
            {"text": doc.page_content, "score": float(score), "metadata": doc.metadata}
            for doc, score in results
        ]