
import hashlib
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np


class VectorStore:
    """
    轻量级本地向量数据库。

    SQLite：保存文本、元数据和向量。
    NumPy：计算余弦相似度。
    """

    def __init__(
        self,
        db_path: str | Path,
        dimension: int,
        model: str,
    ):
        if dimension <= 0:
            raise ValueError("dimension 必须大于 0")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.dimension = dimension
        self.model = model

        # 初始化 SQLite 数据库
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS store_config (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER,
                        start_char INTEGER,
                        end_char INTEGER,
                        embedding BLOB NOT NULL
                    )
                """)

                # 防止不同 Embedding 模型或维度混用
                expected = {
                    "embedding_model": model,
                    "dimension": str(dimension),
                }

                for key, value in expected.items():
                    row = conn.execute(
                        "SELECT value FROM store_config WHERE key = ?",
                        (key,),
                    ).fetchone()

                    if row is not None and row[0] != value:
                        raise ValueError(
                            f"向量库配置不匹配: {key} "
                            f"已保存 {row[0]!r}，当前 {value!r}"
                        )

                    conn.execute(
                        """
                        INSERT OR IGNORE INTO store_config
                        (key, value) VALUES (?, ?)
                        """,
                        (key, value),
                    )

    def _vector(self, values: list[float]) -> np.ndarray:
        """将向量转换为 float32，并验证有效性。"""

        vector = np.asarray(values, dtype=np.float32)

        if vector.ndim != 1 or vector.size != self.dimension:
            raise ValueError(
                f"向量必须是一维、长度为 {self.dimension}"
            )

        if not np.all(np.isfinite(vector)):
            raise ValueError("向量不能包含 NaN 或 Infinity")

        if np.linalg.norm(vector) == 0:
            raise ValueError("向量不能全为 0")

        return vector

    def add(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> int:
        """批量保存 Chunk 和对应的 Embedding。"""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunk 数量和向量数量必须一致"
            )

        rows = []

        for chunk, values in zip(chunks, embeddings):
            vector = self._vector(values)

            text = chunk["text"]
            source = chunk["source"]
            page = chunk.get("page")
            start = chunk.get("start_char")
            end = chunk.get("end_char")

            # 生成稳定的内容标识，避免重复插入相同 Chunk
            identity = f"{source}|{page}|{start}|{end}|{text}"

            chunk_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()

            rows.append((
                chunk_id,
                text,
                source,
                page,
                start,
                end,
                vector.tobytes(),
            ))

        # 一批 Chunk 使用同一个事务写入
        with closing(sqlite3.connect(self.db_path)) as conn:
            with conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO chunks
                    (id, text, source, page,
                     start_char, end_char, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )

        return len(rows)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[dict]:
        """计算 Query 与知识库向量的余弦相似度。"""

        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")

        query = self._vector(query_embedding)

        # 1. 从 SQLite 读取向量和文本
        with closing(sqlite3.connect(self.db_path)) as conn:
            rows = conn.execute("""
                SELECT id, text, source, page,
                       start_char, end_char, embedding
                FROM chunks
            """).fetchall()

        if not rows:
            return []

        # 2. 将所有向量组成 NumPy 矩阵
        matrix = np.stack([
            np.frombuffer(row[6], dtype=np.float32)
            for row in rows
        ])

        if matrix.shape[1] != self.dimension:
            raise RuntimeError("数据库中的向量维度异常")

        # 3. 计算余弦相似度
        scores = (matrix @ query) / (
            np.linalg.norm(matrix, axis=1)
            * np.linalg.norm(query)
        )

        # 4. 从高到低排序，取前 top_k 个
        order = np.argsort(-scores)[:top_k]

        # 5. 返回文本、来源和相似度
        return [
            {
                "chunk_id": rows[i][0],
                "text": rows[i][1],
                "source": rows[i][2],
                "page": rows[i][3],
                "start_char": rows[i][4],
                "end_char": rows[i][5],
                "score": float(scores[i]),
            }
            for i in order
        ]


# ==============================
# 本地测试：不调用任何 API
# ==============================

if __name__ == "__main__":

    # 临时数据库，测试结束后自动删除
    with tempfile.TemporaryDirectory() as temp:
        db_path = Path(temp) / "test_vectors.db"

        store = VectorStore(
            db_path=db_path,
            dimension=3,
            model="demo-model",
        )

        chunks = [
            {
                "text": "Agent 可以调用工具",
                "source": "agent.md",
                "page": None,
            },
            {
                "text": "智能体能调用外部 API",
                "source": "agent.md",
                "page": None,
            },
            {
                "text": "今天武汉天气很好",
                "source": "weather.md",
                "page": None,
            },
        ]

        # 手工构造三维向量用于验证计算
        fake_vectors = [
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 0.0, 1.0],
        ]

        count = store.add(chunks, fake_vectors)
        print("写入数量:", count)

        # 重新打开数据库，验证数据确实已保存
        reopened_store = VectorStore(
            db_path=db_path,
            dimension=3,
            model="demo-model",
        )

        query_vector = [1.0, 0.0, 0.0]

        results = reopened_store.search(
            query_embedding=query_vector,
            top_k=3,
        )

        for rank, item in enumerate(results, start=1):
            print(
                f"{rank}. 相似度={item['score']:.4f}"
                f" | {item['text']}"
            )
