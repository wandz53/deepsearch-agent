
import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import dotenv_values


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class RAGSettings:
    api_key: str
    base_url: str
    model: str
    dimensions: int


def get_rag_settings() -> RAGSettings:
    """
    优先读取项目 .env，其次读取环境变量。
    只读取 RAG 专用配置，不修改 Pi 或主 Agent 的环境变量。
    """

    local_config = dotenv_values(ENV_FILE)

    def get_value(name: str, default=None):
        return (
            local_config.get(name)
            or os.getenv(name)
            or default
        )

    api_key = get_value("RAG_EMBEDDING_API_KEY")
    base_url = get_value("RAG_EMBEDDING_BASE_URL")

    if not api_key or not base_url:
        raise RuntimeError(
            "缺少 RAG_EMBEDDING_API_KEY "
            "或 RAG_EMBEDDING_BASE_URL"
        )

    return RAGSettings(
        api_key=api_key,
        base_url=base_url,
        model=get_value(
            "RAG_EMBEDDING_MODEL",
            "text-embedding-v4"
        ),
        dimensions=int(
            get_value("RAG_EMBEDDING_DIM", "512")
        ),
    )
