"""
任务产物检查 Tool

提供给 Reviewer Agent 使用。
Reviewer 不需要知道真实 session_dir，
由 Tool 内部自动获取当前任务工作目录。
"""

from langchain_core.tools import tool

from app.api.context import get_session_context
from app.utils.artifact_checker import inspect_artifacts


@tool
def check_artifacts() -> dict:
    """
    检查当前任务真实工作目录中的文件。

    用于确认 Markdown、PDF 等任务交付物是否真实存在、
    是否为空以及实际保存路径。

    检查交付物时应优先使用本工具，
    不需要自行通过 ls、glob、grep 等方式寻找文件。
    """

    # 当前 main_agent 已经通过 set_session_context()
    # 保存了真实任务工作目录，这里直接读取
    session_dir = get_session_context()

    if not session_dir:
        return {
            "session_dir_exists": False,
            "error": "当前任务没有可用的 session_dir",
            "generated_files": [],
            "all_files": [],
        }

    return inspect_artifacts(
        session_dir=session_dir,
        input_files=None,
    )
