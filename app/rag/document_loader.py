
from pathlib import Path

from pypdf import PdfReader
from docx import Document


def load_document(file_path: str) -> list[dict]:
    """
    读取文档，返回文本及来源信息。
    支持 PDF、DOCX、TXT、Markdown。
    """

    path = Path(file_path)

    # 1. 检查文件是否存在
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 2. 获取文件类型
    suffix = path.suffix.lower() 

    results = []

    # 3. 读取 PDF：按页提取文本
    if suffix == ".pdf":
        reader = PdfReader(str(path))

        for page_num, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if text:
                results.append({
                    "text": text,
                    "source": path.name,
                    "page": page_num
                })

    # 4. 读取 Word
    elif suffix == ".docx":
        doc = Document(str(path))

        text = "\n".join(
            p.text for p in doc.paragraphs if p.text.strip()
        )

        if text.strip():
            results.append({
                "text": text,
                "source": path.name,
                "page": None
            })

    # 5. 读取 TXT、Markdown
    elif suffix in [".txt", ".md"]:
        text = path.read_text(encoding="utf-8-sig").strip()

        if text:
            results.append({
                "text": text,
                "source": path.name,
                "page": None
            })

    else:
        raise ValueError(f"不支持的文件格式: {suffix}")

    if not results:
        raise ValueError("文档中没有提取到可用文本")

    return results


# 本地测试入口
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("用法: python -m app.rag.document_loader 文件路径")
        sys.exit(1)

    documents = load_document(sys.argv[1])

    print(f"成功读取 {len(documents)} 个文本单元")

    for item in documents[:3]:
        print("文件:", item["source"])
        print("页码:", item["page"])
        print("文本长度:", len(item["text"]))
        print("内容预览:", item["text"][:150])
        print("-" * 40)
