
from openai import OpenAI

from app.rag.config import get_rag_settings


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """
    调用云端 Embedding API，将文本转换为向量。

    参数：
        texts: 文本列表，例如多个 Chunk。

    返回：
        list[list[float]]
        每段文本对应一个 Embedding 向量。
    """

    # 1. 检查输入
    if not texts:
        raise ValueError("文本列表不能为空")

    if any(
        not isinstance(t, str) or not t.strip()
        for t in texts
    ):
        raise ValueError("texts 必须包含非空字符串")

    # 2. 读取 RAG 专用配置
    settings = get_rag_settings()

    # 3. 创建 OpenAI 兼容 API 客户端
    client = OpenAI(
        api_key=settings.api_key,
        base_url=settings.base_url,
        timeout=30.0,
        max_retries=2,
    )

    # 4. 保存生成的所有向量
    vectors = []

    # 5. 分批调用 Embedding API
    batch_size = 10

    for start in range(0, len(texts), batch_size):

        batch = texts[start:start + batch_size]

        response = client.embeddings.create(
            model=settings.model,
            input=batch,
            dimensions=settings.dimensions,
            encoding_format="float",
        )

        # 6. 按原始输入顺序排列结果
        items = sorted(
            response.data,
            key=lambda item: item.index,
        )

        # 7. 检查返回的向量数量
        if len(items) != len(batch):
            raise RuntimeError(
                "Embedding 返回数量与输入文本数量不一致"
            )

        if [item.index for item in items] != list(
            range(len(batch))
        ):
            raise RuntimeError(
                "Embedding 返回的索引异常"
            )

        # 8. 检查每个向量的维度
        for item in items:

            if len(item.embedding) != settings.dimensions:
                raise RuntimeError(
                    "Embedding 返回的向量维度异常"
                )

            vectors.append(item.embedding)

    return vectors


# =====================================
# 本地测试
# =====================================

if __name__ == "__main__":

    sentences = [
        "Agent 可以调用外部工具完成任务",
        "智能体能够使用工具执行操作",
        "今天武汉的天气很好",
    ]

    # 调用 Embedding API
    vectors = get_embeddings(sentences)

    print("文本数量:", len(sentences))
    print("向量数量:", len(vectors))
    print("向量维度:", len(vectors[0]))

    # 输出每句话对应的向量信息
    for i, vector in enumerate(vectors):

        print(f"\n文本 {i + 1}: {sentences[i]}")
        print("向量前5个数值:", vector[:5])
