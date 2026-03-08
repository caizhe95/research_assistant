from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

class KnowledgeBase:
    def __init__(self, knowledge_dir: Path, embeddings):
        self.knowledge_dir = Path(knowledge_dir)
        self.embeddings = embeddings
        self.documents: list[Document] = []
        self.vector_store: InMemoryVectorStore | None = None

    def build(self) -> None:
        raw_documents = self._load_local_documents()

        if not raw_documents:
            self.documents = []
            self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=120,
        )
        chunks = splitter.split_documents(raw_documents)

        self.documents = chunks
        self.vector_store = InMemoryVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,
        )

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if self.vector_store is None:
            self.build()

        if not self.documents:
            return []

        retriever = self.vector_store.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.invoke(query)

        results = []
        for doc in docs:
            results.append(
                {
                    "title": doc.metadata.get("title", "未命名文档"),
                    "source": doc.metadata.get("source", "local_knowledge"),
                    "url": None,
                    "snippet": doc.page_content[:240],
                    "content": doc.page_content,
                    "source_type": "local_knowledge",
                }
            )

        return results

    def _load_local_documents(self) -> list[Document]:
        documents = []

        for path in sorted(self.knowledge_dir.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".md", ".txt"}:
                continue

            content = path.read_text(encoding="utf-8").strip()
            if not content:
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "title": path.stem.replace("_", " "),
                        "source": path.name,
                    },
                )
            )

        return documents
