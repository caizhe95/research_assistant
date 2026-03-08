import json

import arxiv
from langchain.tools import tool

from config import ARXIV_TOP_K, LOCAL_RETRIEVAL_TOP_K, MEMORY_TOP_K

def create_tools(knowledge_base, memory_store, user_id: str) -> list:
    @tool
    def search_local_knowledge(query: str, top_k: int = LOCAL_RETRIEVAL_TOP_K) -> str:
        items = knowledge_base.search(query=query, top_k=top_k)
        return json.dumps(
            {
                "tool": "search_local_knowledge",
                "query": query,
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        )

    @tool
    def search_arxiv_papers(query: str, max_results: int = ARXIV_TOP_K) -> str:
        items = []

        try:
            client = arxiv.Client(page_size=max_results, delay_seconds=1, num_retries=2)
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            for paper in client.results(search):
                items.append(
                    {
                        "title": paper.title.replace("\n", " ").strip(),
                        "source": "arXiv",
                        "url": paper.entry_id,
                        "snippet": paper.summary.replace("\n", " ").strip()[:300],
                        "content": paper.summary.replace("\n", " ").strip(),
                        "source_type": "arxiv",
                    }
                )
        except Exception as error:
            items.append(
                {
                    "title": "arXiv 检索失败",
                    "source": "arXiv",
                    "url": None,
                    "snippet": f"错误信息：{error}",
                    "content": f"错误信息：{error}",
                    "source_type": "arxiv_error",
                }
            )

        return json.dumps(
            {
                "tool": "search_arxiv_papers",
                "query": query,
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        )

    @tool
    def recall_user_memory(query: str, top_k: int = MEMORY_TOP_K) -> str:
        items = []
        for item in memory_store.search(user_id=user_id, query=query, limit=top_k):
            items.append(
                {
                    "title": f"用户长期记忆-{item.get('memory_type', 'fact')}",
                    "source": "long_term_memory",
                    "url": None,
                    "snippet": item.get("content", ""),
                    "content": item.get("content", ""),
                    "source_type": "memory",
                }
            )

        return json.dumps(
            {
                "tool": "recall_user_memory",
                "query": query,
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        )

    return [search_local_knowledge, search_arxiv_papers, recall_user_memory]
