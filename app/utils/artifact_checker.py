"""
任务产物检查模块

直接检查当前 session_dir 对应的真实文件系统，
用于确认 Agent 实际生成了哪些文件。

注意：
该模块不是 LLM Tool，不使用 @tool。
"""

from pathlib import Path
from typing import Iterable


def inspect_artifacts(
    session_dir: str | Path,
    input_files: Iterable[str] | None = None,
) -> dict:
    """
    检查当前会话目录中的真实文件。

    :param session_dir:
        当前任务真实工作目录，例如：
        app/output/session_eval_T02_xxx

    :param input_files:
        本次用户上传的原始文件名，例如：
        ["T02_facts.md"]

        这些文件虽然也被复制进 session_dir，
        但不能算作 Agent 生成的交付物。

    :return:
        当前会话目录和生成文件的结构化检查结果
    """

    session_path = Path(session_dir).resolve()

    # 转为 set，方便后面快速判断某个文件是不是输入文件
    input_file_names = set(input_files or [])

    if not session_path.exists():
        return {
            "session_dir_exists": False,
            "generated_files": [],
            "all_files": [],
        }

    if not session_path.is_dir():
        return {
            "session_dir_exists": False,
            "generated_files": [],
            "all_files": [],
        }

    all_files = []

    # 使用 rglob 而不是 iterdir：
    # 除了根目录，也能够检查 Agent 生成到子目录中的文件
    for file_path in sorted(session_path.rglob("*")):

        if not file_path.is_file():
            continue

        relative_path = file_path.relative_to(session_path)
        
   # file_size = file_path.stat().st_size
        
        file_info = {
            "name": file_path.name,
            "relative_path": str(relative_path).replace("\\", "/"),
            "suffix": file_path.suffix.lower(),
            "size_bytes": file_path.stat().st_size,
            "exists": True,
            "non_empty": file_path.stat().st_size > 0,
            "is_input": file_path.name in input_file_names,
        }

        all_files.append(file_info)

    # 去掉用户原始上传文件，只保留 Agent 后续生成的文件
    generated_files = [
        file_info
        for file_info in all_files
        if not file_info["is_input"]
    ]

    return {
        "session_dir_exists": True,
        "generated_file_count": len(generated_files),
        "generated_files": generated_files,
        "all_files": all_files,
    }
