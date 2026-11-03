
import argparse
from pathlib import Path

from app.rag.config import get_rag_settings
from app.rag.document_loader import load_document
from app.rag.chunker import chunk_documents
from app.rag.embedding import get_embeddings
from app.rag.vector_store import VectorStore
from app.rag.reranker import Reranker


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 本地知识库保存位置
DEFAULT_DB_PATH = (
    PROJECT_ROOT / "data" / "knowledge_base" / "vectors.db"
)


class RAGPipeline:
    """
    轻量级 RAG 检索流水线。

    功能：
    1. 文档加载
    2. 文本分块
    3. Embedding 向量化
    4. SQLite 向量存储
    5. 余弦相似度检索
    """

    def __init__(self, db_path=DEFAULT_DB_PATH):

        # 获取统一配置
        self.settings = get_rag_settings()

        # 初始化向量数据库
        self.vector_store = VectorStore(
            db_path=db_path,
            dimension=self.settings.dimensions,
            model=self.settings.model,
        )

    def index_document(
        self,
        file_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> int:
        """
        将文档添加到知识库。

        读取 -> 分块 -> Embedding -> 保存
        """

        print(f"\n开始处理文档: {file_path}")

        # 第一步：加载原始文档
        documents = load_document(file_path)

        print(f"原始文本单元: {len(documents)}")

        # 第二步：文本切分
        chunks = chunk_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not chunks:
            raise ValueError("没有生成有效的 Chunk")

        print(f"Chunk 数量: {len(chunks)}")

        # 第三步：提取每个 Chunk 的文本
        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        # 第四步：生成 Embedding
        print("正在调用 Embedding API...")

        embeddings = get_embeddings(texts)

        # 第五步：保存到 SQLite
        saved_count = self.vector_store.add(
            chunks=chunks,
            embeddings=embeddings,
        )

        print(f"成功写入 {saved_count} 个 Chunk")

        return saved_count

   
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> list[dict]:
        """
        检索与 Query 最相关的 Chunk。

        参数：
            query: 用户问题
            top_k: 初步检索数量
            score_threshold: 最低余弦相似度阈值

        返回：
            通过阈值过滤后的检索结果
        """

        if not query.strip():
            raise ValueError("Query 不能为空")

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        if score_threshold is not None:
            if not -1.0 <= score_threshold <= 1.0:
                raise ValueError(
                    "score_threshold 必须在 -1 到 1 之间"
                )

        # 1. 将用户问题转换为向量
        query_embedding = get_embeddings([query])[0]

        # 2. 从 Vector Store 中检索 Top-K
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        # 3. 根据相似度阈值过滤
        if score_threshold is not None:
            results = [
                item
                for item in results
                if item["score"] >= score_threshold
            ]

        return results


    def retrieve_reranked(
        self,
        query: str,
        candidate_k: int = 10,
        top_n: int = 3,
        score_threshold: float = 0.4,
    ) -> list[dict]:
        """
        两阶段检索：
        1. 向量检索候选 Chunk
        2. Rerank 精细排序
        """

        # 第一阶段：向量检索 + 阈值过滤
        candidates = self.retrieve(
            query=query,
            top_k=candidate_k,
            score_threshold=score_threshold,
        )

        if not candidates:
            return []

        # 第二阶段：Rerank
        reranker = Reranker()

        results = reranker.rerank(
            query=query,
            documents=candidates,
            top_n=top_n,
        )

        return results

# ======================================
# 命令行入口
# ======================================

def main():

    parser = argparse.ArgumentParser(
        description="自研 RAG 知识库"
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # index 命令：导入文档
    index_parser = subparsers.add_parser("index")

    index_parser.add_argument("file_path")

    index_parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
    )

    index_parser.add_argument(
        "--overlap",
        type=int,
        default=100,
    )

    # search 命令：检索知识库
    search_parser = subparsers.add_parser("search")

    search_parser.add_argument("query")

    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    pipeline = RAGPipeline()

    if args.command == "index":

        pipeline.index_document(
            file_path=args.file_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.overlap,
        )

    elif args.command == "search":

        results = pipeline.retrieve(
            query=args.query,
            top_k=args.top_k,
        )

        print(f"\n检索到 {len(results)} 个结果")

        for rank, item in enumerate(results, start=1):

            print("\n" + "=" * 50)
            print(f"排名: {rank}")
            print(f"相似度: {item['score']:.4f}")
            print(f"来源: {item['source']}")
            print(f"页码: {item['page']}")
            print(f"内容:\n{item['text'][:500]}")


if __name__ == "__main__":
    main()
