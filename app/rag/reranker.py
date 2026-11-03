
from openai import OpenAI
from dotenv import dotenv_values

from app.rag.config import ENV_FILE, get_rag_settings


class Reranker:

    def __init__(self):

        settings = get_rag_settings()
        config = dotenv_values(ENV_FILE)

        self.model = config.get(
            "RAG_RERANK_MODEL", "qwen3-rerank"
        )

        base_url = config.get("RAG_RERANK_BASE_URL")

        if not base_url:
            raise ValueError("请配置 RAG_RERANK_BASE_URL")

        self.client = OpenAI(
            api_key=settings.api_key,
            base_url=base_url,
            timeout=30.0,
            max_retries=1,
        )

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_n: int = 3,
    ) -> list[dict]:

        if not documents:
            return []

        if not query.strip() or top_n <= 0:
            raise ValueError("query 不能为空，top_n 必须大于 0")

        # 提取初步检索的 Chunk 文本
        texts = [doc["text"] for doc in documents]

        # 调用 Rerank API
        response = self.client.post(
            "/reranks",
            body={
                "model": self.model,
                "query": query,
                "documents": texts,
                "top_n": min(top_n, len(texts)),
            },
            cast_to=object,
        )

        # 根据模型返回的索引重新排序
        results = []

        for item in response["results"]:
            doc = documents[item["index"]].copy()

            doc["rerank_score"] = item["relevance_score"]

            results.append(doc)

        return results


if __name__ == "__main__":

    query = "Checkpointer 如何实现跨进程恢复？"

    documents = [
        {"text": "Checkpointer 用来保存 Agent 状态。"},
        {"text": "数据库 Checkpointer 可以在进程重启后恢复状态。"},
        {"text": "ContextVar 用于异步上下文变量隔离。"},
    ]

    reranker = Reranker()

    results = reranker.rerank(query, documents)

    for i, doc in enumerate(results, 1):
        print(
            f"{i}. 分数={doc['rerank_score']:.4f}"
            f" | {doc['text']}"
        )
