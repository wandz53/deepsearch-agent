
from app.rag.document_loader import load_document


def chunk_documents(
    documents: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[dict]:
    """
    将 Document Loader 的输出切分为多个 Chunk。

    每个 Chunk 保留：
    - text: 分块后的文本
    - source: 原始文件名
    - page: 原始页码
    - chunk_id: 分块编号
    - start_char / end_char: 在原始文本中的位置
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")

    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            "chunk_overlap 必须 >= 0 且 < chunk_size"
        )

    chunks = []
    step = chunk_size - chunk_overlap

    for doc_index, document in enumerate(documents):

        text = document["text"]
        source = document["source"]
        page = document["page"]

        start = 0
        chunk_index = 0

        while start < len(text):

            end = min(start + chunk_size, len(text))

            chunk_text = text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "chunk_id": f"{doc_index}_{chunk_index}",
                    "text": chunk_text,
                    "source": source,
                    "page": page,
                    "start_char": start,
                    "end_char": end,
                })

                chunk_index += 1

            # 已经到达文本末尾，停止切分
            if end == len(text):
                break

            # 向后移动，同时保留重叠部分
            start += step

    return chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("用法: python -m app.rag.chunker 文件路径")
        sys.exit(1)

    documents = load_document(sys.argv[1])

    chunks = chunk_documents(
        documents,
        chunk_size=500,
        chunk_overlap=100,
    )

    print(f"原始文本单元数量: {len(documents)}")
    print(f"切分后 Chunk 数量: {len(chunks)}")

    for chunk in chunks[:3]:
        print("-" * 50)
        print("Chunk ID:", chunk["chunk_id"])
        print("来源:", chunk["source"])
        print("页码:", chunk["page"])
        print(
            "字符区间:",
            chunk["start_char"],
            chunk["end_char"],
        )
        print("文本长度:", len(chunk["text"]))
        print("内容预览:", repr(chunk["text"][:120]))

    # 使用同一份文本中的相邻 Chunk 验证重叠
    if len(chunks) >= 2:
        first = chunks[0]
        second = chunks[1]

        if (
            first["source"] == second["source"]
            and first["page"] == second["page"]
            and second["start_char"] == 400
        ):
            print(
                "相邻 Chunk 的重叠部分一致:",
                first["text"][-100:] == second["text"][:100]
            )
