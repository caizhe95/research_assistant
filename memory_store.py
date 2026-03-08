import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from utils import now_str

class ResearchMemoryStore:
    def __init__(self, memory_dir: Path, embeddings):
        self.memory_dir = Path(memory_dir)
        self.embeddings = embeddings
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def add_memory(
        self,
        user_id: str,
        content: str,
        tags: list[str] | None = None,
        memory_type: str = "fact",
    ) -> dict:
        if tags is None:
            tags = []

        memories = self.load_memories(user_id)
        normalized_content = content.strip()

        for item in memories:
            if item.get("content", "").strip() == normalized_content:
                return item

        item = {
            "id": f"mem_{len(memories) + 1}",
            "content": normalized_content,
            "tags": tags,
            "memory_type": memory_type,
            "created_at": now_str(),
        }
        memories.append(item)
        self._save_memories(user_id, memories)
        return item

    def search(self, user_id: str, query: str, limit: int = 3) -> list[dict]:
        memories = self.load_memories(user_id)

        if not memories:
            return []

        docs = [
            Document(page_content=item["content"], metadata=item)
            for item in memories
            if item.get("content")
        ]

        if not docs:
            return []

        store = InMemoryVectorStore.from_documents(
            documents=docs,
            embedding=self.embeddings,
        )
        retriever = store.as_retriever(search_kwargs={"k": limit})
        hits = retriever.invoke(query)

        results = []
        for doc in hits:
            results.append(
                {
                    **doc.metadata,
                    "content": doc.page_content,
                }
            )

        return results

    def load_memories(self, user_id: str) -> list[dict]:
        path = self._get_user_memory_path(user_id)
        if not path.exists():
            return []

        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []

    def _save_memories(self, user_id: str, memories: list[dict]) -> None:
        path = self._get_user_memory_path(user_id)
        path.write_text(
            json.dumps(memories, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _get_user_memory_path(self, user_id: str) -> Path:
        return self.memory_dir / f"{user_id}.json"
