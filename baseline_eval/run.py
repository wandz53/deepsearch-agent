"""独立评测外壳：不改变原 Agent 的提示词、工具和控制流程。Python 3.12。"""
import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def plain(value):
    if hasattr(value, "model_dump"):
        return plain(value.model_dump())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_lines(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # 强制超时可能留下最后半行；此前的记录仍然有效。
    return rows


class Recorder:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.lock = threading.RLock()

    def emit(self, filename, kind, **data):
        with self.lock:
            with (self.folder / filename).open("a", encoding="utf-8") as f:
                f.write(json.dumps(plain({"timestamp": now(), "kind": kind, **data}), ensure_ascii=False) + "\n")


def make_callback(recorder):
    from langchain_core.callbacks import BaseCallbackHandler

    class Callback(BaseCallbackHandler):
        # 记录先落盘再继续；不同工具线程通过锁串行写文件。
        run_inline = True
        raise_error = True

        def log(self, kind, run_id, parent_run_id=None, **data):
            recorder.emit("trace.jsonl", kind, run_id=str(run_id),
                          parent_run_id=str(parent_run_id) if parent_run_id else None, **data)

        def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
            self.log("tool_start", run_id, parent_run_id,
                     name=(serialized or {}).get("name", kwargs.get("name", "unknown")), input=input_str)

        def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
            self.log("tool_end", run_id, parent_run_id, output=output)

        def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
            self.log("tool_error", run_id, parent_run_id, error=str(error))

        def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kwargs):
            # 不转储模型初始化参数、环境变量或完整配置中的密钥。
            self.log("llm_start", run_id, parent_run_id)

        def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
            self.log("llm_start", run_id, parent_run_id)

        def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
            usage = None
            for group in response.generations:
                for generation in group:
                    message = getattr(generation, "message", None)
                    candidate = getattr(message, "usage_metadata", None)
                    if candidate and candidate.get("total_tokens") is not None:
                        usage = candidate
                        break
                if usage is not None:
                    break
            if usage is None:
                raw = (response.llm_output or {}).get("token_usage", {})
                if raw.get("total_tokens") is not None:
                    usage = {"input_tokens": raw.get("prompt_tokens"),
                             "output_tokens": raw.get("completion_tokens"),
                             "total_tokens": raw["total_tokens"]}
            self.log("llm_end", run_id, parent_run_id, usage=usage)

        def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
            self.log("llm_error", run_id, parent_run_id, error=str(error))

    return Callback()


def worker(root, folder):
    sys.path.insert(0, str(root))
    record = Recorder(folder)
    case_input = read_json(folder / "input.json")
    start = None
    state = "initialization_error"
    detail = None
    try:
        from app.api.monitor import monitor
        original_emit = monitor._emit

        def wrapped_emit(event_type, message, data=None):
            record.emit("monitor_events.jsonl", event_type, message=message, data=data or {})
            return original_emit(event_type, message, data)

        monitor._emit = wrapped_emit
        import app.agent.main_agent as module
        callback = make_callback(record)
        module.main_agent = module.main_agent.with_config(callbacks=[callback])
        model = getattr(module, "model", None)
        write_json(folder / "model.json", {
            "model_name": getattr(model, "model_name", None),
            "temperature": getattr(model, "temperature", None),
            "max_tokens": getattr(model, "max_tokens", None),
            "note": "null表示配置未显式暴露；不记录API密钥。工具观察次数不等于语义成功次数。",
        })
        start = time.perf_counter()
        write_json(folder / "worker_status.json", {"state": "running", "started_at": now()})
        asyncio.run(module.run_deep_agent(case_input["question"], case_input["thread_id"]))
        state = "returned"
    except BaseException as exc:
        detail = f"{type(exc).__name__}: {exc}"
        if start is not None:
            state = "exception"
        traceback.print_exc()
    finally:
        write_json(folder / "worker_status.json", {
            "state": state, "detail": detail, "finished_at": now(),
            "agent_seconds": round(time.perf_counter() - start, 3) if start is not None else None,
        })
    return 0 if state == "returned" else 1


def metrics(folder):
    trace = read_lines(folder / "trace.jsonl")
    events = read_lines(folder / "monitor_events.jsonl")
    starts = {e["run_id"]: e for e in trace if e["kind"] == "tool_start"}
    llm_ids = {e["run_id"] for e in trace if e["kind"].startswith("llm_")}
    usages = {e["run_id"]: e["usage"] for e in trace
              if e["kind"] == "llm_end" and e.get("usage") is not None}
    observed = sum(u["total_tokens"] for u in usages.values())
    searches = []
    for e in trace:
        if e["kind"] == "tool_end" and starts.get(e["run_id"], {}).get("name") == "internet_search":
            searches.append({"run_id": e["run_id"], "input": starts[e["run_id"]]["input"], "output": e["output"]})
    write_json(folder / "search_results.json", searches)
    answers = [e.get("data", {}).get("result") for e in events if e["kind"] == "task_result"]
    answers = [a for a in answers if a]
    if answers:
        answer = answers[-1]
        (folder / "answer.md").write_text(answer if isinstance(answer, str) else
                                         json.dumps(answer, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "answer_received": bool(answers),
        "tool_call_count_observed": len(starts),
        "search_call_count_observed": sum(e.get("name") == "internet_search" for e in starts.values()),
        "subagent_call_count_observed": sum(e.get("name") == "task" for e in starts.values()),
        "tool_error_count_observed": len({e["run_id"] for e in trace if e["kind"] == "tool_error"}),
        "monitor_error_count": sum(e["kind"] == "error" for e in events),
        "llm_call_count_observed": len(llm_ids),
        "llm_calls_with_usage": len(usages),
        "observed_total_tokens": observed if usages else None,
        "total_tokens": observed if llm_ids and len(usages) == len(llm_ids) else None,
        "trace_observed": bool(trace),
    }


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root, *args):
    try:
        p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=15)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def manifest(root, cases, args):
    files = [root / "pyproject.toml", root / "uv.lock"]
    for p in (root / "app").rglob("*"):
        if p.is_file() and p.suffix in (".py", ".yml", ".yaml") and not {"output", "updated", "__pycache__"}.intersection(p.relative_to(root / "app").parts):
            files.append(p)
    versions = {}
    for name in ("deepagents", "langchain", "langchain-core", "langgraph", "langchain-openai", "tavily-python"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    fixtures = {f: digest(HERE / f) for c in cases for f in c["fixtures"]}
    dataset = hashlib.sha256(json.dumps({"cases": cases, "fixtures": fixtures}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return {"created_at": now(), "label": args.label, "git_commit": git(root, "rev-parse", "HEAD"),
            "git_status": git(root, "status", "--short"), "python": sys.version,
            "versions": versions, "source_sha256": {str(p.relative_to(root)): digest(p) for p in files if p.exists()},
            "dataset_sha256": dataset, "fixture_sha256": fixtures, "cases": cases,
            "repeat": args.repeat, "timeout_seconds": args.timeout,
            "runner_sha256": digest(Path(__file__)), "protocol": "original-run_deep_agent-with-observation-v1"}


def summarize(batch):
    rows = []
    for folder in sorted(batch.glob("T*_r*")):
        if not (folder / "result.json").exists():
            continue
        row = read_json(folder / "result.json")
        ratings = read_json(folder / "scores.json")["ratings"]
        if any(r.get("passed") is not None and type(r["passed"]) is not bool for r in ratings):
            raise ValueError(f"{folder}/scores.json: passed只能是true、false或null")
        complete = bool(ratings) and all(type(r.get("passed")) is bool for r in ratings)
        row["rubric_score"] = round(sum(r["passed"] for r in ratings) / len(ratings), 4) if complete else None
        if row["execution_state"] != "returned_with_answer" or not all(row["expected_files_exist"].values()):
            row["quality_pass"] = False
        else:
            row["quality_pass"] = all(r["passed"] for r in ratings) if complete else None
        rows.append(row)
    known = [r for r in rows if r["quality_pass"] is not None]
    meta = read_json(batch / "manifest.json")
    planned = len(meta["cases"]) * meta["repeat"]
    summary = {"batch": batch.name, "planned_runs": planned, "runs": len(rows), "quality_scored_runs": len(known),
               "quality_pass_rate": sum(r["quality_pass"] for r in known) / len(rows)
               if rows and len(known) == len(rows) == planned else None, "results": rows}
    write_json(batch / "summary.json", summary)
    lines = [f"# {batch.name} 评测汇总", "", f"计划{planned}次，已有记录{len(rows)}次。",
             "", "未评分用 — 表示。运行返回和文件存在均不能单独证明内容正确。", "",
             "| 题目/轮次 | 运行状态 | Agent秒数 | 搜索调用（观察值） | Token（观察完整时） | 人工分 | 全部通过 |",
             "|---|---|---:|---:|---:|---:|---|"]
    for r in rows:
        values = [r["case_id"] + f'/r{r["repeat"]}', r["execution_state"], r.get("agent_seconds"),
                  r["search_call_count_observed"], r["total_tokens"], r["rubric_score"], r["quality_pass"]]
        lines.append("| " + " | ".join("—" if v is None else str(v) for v in values) + " |")
    lines += ["", f"全部题目内容通过率：{summary['quality_pass_rate'] if summary['quality_pass_rate'] is not None else '待完成评分'}",
              "", "成本未按金额估算；Token只覆盖收到的模型回调，不包含不可见服务端重试或检索费用。",
              "T01使用实时网络，跨批次结果还会受到检索变化影响。T02/T03采用固定附件。"]
    (batch / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def compare(first, second):
    a, b = read_json(first / "manifest.json"), read_json(second / "manifest.json")
    if a["dataset_sha256"] != b["dataset_sha256"]:
        raise ValueError("两批题目或附件不同，不能直接对比。请用相同cases.json和fixtures重跑。")
    summaries = [summarize(first), summarize(second)]
    lines = ["# 版本比较", "", "描述性对比，不代表统计显著性。检查模型与环境配置是否一致；T01含实时网络波动。", "",
             "| 批次 | 次数 | 内容通过率 | 平均Agent耗时/秒 | 平均搜索观察次数 |", "|---|---:|---:|---:|---:|"]
    for s in summaries:
        times = [r["agent_seconds"] for r in s["results"] if r.get("agent_seconds") is not None]
        searches = [r["search_call_count_observed"] for r in s["results"]]
        vals = [s["batch"], s["runs"], s["quality_pass_rate"],
                round(statistics.mean(times), 2) if times else None,
                round(statistics.mean(searches), 2) if searches else None]
        lines.append("| " + " | ".join("—" if v is None else str(v) for v in vals) + " |")
    lines += ["", "超时的Agent耗时没有完成值，未纳入平均；同时查看各批次运行状态，不能据此宣称速度提升。",
              "对比依据：两批manifest.json、model.json、summary.json，以及每题原始证据和人工评分。"]
    output = second / f"comparison_with_{first.name}.md"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"比较已保存：{output}")


def run_batch(args, cases):
    root = Path(args.project_root).resolve() if args.project_root else HERE.parent
    if not (root / "app/agent/main_agent.py").is_file():
        raise ValueError("请把baseline_eval文件夹放到项目根目录，与app、pyproject.toml同级。")
    batch = HERE / "results" / f"{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    batch.mkdir(parents=True)
    write_json(batch / "manifest.json", manifest(root, cases, args))
    (batch / "RUN_PATH.txt").write_text(str(batch), encoding="utf-8")
    print(f"结果目录：{batch}", flush=True)
    stop = False
    try:
        for c in cases:
            for repeat in range(1, args.repeat + 1):
                folder = batch / f'{c["id"]}_r{repeat}'
                folder.mkdir()
                thread = f'eval_{c["id"]}_{uuid.uuid4().hex}'
                write_json(folder / "input.json", {"case_id": c["id"], "question": c["question"],
                                                  "thread_id": thread, "repeat": repeat, "fixtures": c["fixtures"]})
                write_json(folder / "scores.json", {"reviewer": "", "ratings": [
                    {**r, "passed": None, "evidence_or_reason": ""} for r in c["rubric"]]})
                inputs = folder / "inputs"
                inputs.mkdir()
                upload = root / "app/updated" / f"session_{thread}"
                for fixture in c["fixtures"]:
                    upload.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(HERE / fixture, upload / Path(fixture).name)
                    shutil.copy2(HERE / fixture, inputs / Path(fixture).name)
                t = time.perf_counter()
                forced_state = None
                print(f'运行 {c["id"]} 第{repeat}次：{c["name"]}', flush=True)
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONUNBUFFERED"] = "1"
                with (folder / "console.log").open("w", encoding="utf-8") as log:
                    try:
                        subprocess.run([sys.executable, str(Path(__file__)), "--_worker", str(folder),
                                        "--project-root", str(root)], cwd=root, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                    except subprocess.TimeoutExpired:
                        forced_state = "timeout"
                    except KeyboardInterrupt:
                        forced_state = "interrupted"
                        stop = True
                elapsed = round(time.perf_counter() - t, 3)
                statusfile = folder / "worker_status.json"
                status = read_json(statusfile) if statusfile.exists() else {"state": "worker_failed"}
                m = metrics(folder)
                state = forced_state or status["state"]
                if state == "returned":
                    state = "error_event" if m["monitor_error_count"] else (
                        "returned_with_answer" if m["answer_received"] else "returned_without_answer")
                source = root / "app/output" / f"session_{thread}"
                artifacts = folder / "artifacts"
                if source.exists():
                    shutil.copytree(source, artifacts)
                else:
                    artifacts.mkdir()
                result = {"case_id": c["id"], "repeat": repeat, "thread_id": thread,
                          "execution_state": state, "parent_wall_seconds": elapsed,
                          "agent_seconds": status.get("agent_seconds"), "detail": status.get("detail"),
                          "expected_files_exist": {f: (artifacts / f).is_file() for f in c["expected_files"]}, **m}
                write_json(folder / "result.json", result)
                summarize(batch)
                print(f'  {state}；总耗时{elapsed}秒；保存到{folder.name}', flush=True)
                if state in ("initialization_error", "worker_failed"):
                    print("初始化失败，停止后续题目。请先查看本题console.log修复环境。", flush=True)
                    stop = True
                if stop:
                    break
            if stop:
                break
    finally:
        completed = {(read_json(p)["case_id"], read_json(p)["repeat"]) for p in batch.glob("T*_r*/result.json")}
        write_json(batch / "not_run.json", [{"case_id": c["id"], "repeat": r} for c in cases
                                           for r in range(1, args.repeat + 1) if (c["id"], r) not in completed])
    print(f"查看汇总：{batch / 'summary.md'}\n人工评分后重新执行 --summarize。", flush=True)
    return 1 if stop else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="v0")
    parser.add_argument("--case", choices=["T01", "T02", "T03"])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--project-root")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--_worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args._worker:
        return worker(Path(args.project_root).resolve(), args._worker.resolve())
    if args.summarize:
        summarize(args.summarize.resolve())
        print(f"已更新：{args.summarize / 'summary.md'}")
        return 0
    if args.compare:
        compare(*(p.resolve() for p in args.compare))
        return 0
    if args.repeat < 1 or args.timeout <= 0 or not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("repeat和timeout须为正数；label只能包含英文字母、数字、下划线或短横线。")
    cases = read_json(HERE / "cases.json")
    cases = [c for c in cases if not args.case or c["id"] == args.case]
    for c in cases:
        for fixture in c["fixtures"]:
            if not (HERE / fixture).is_file():
                raise FileNotFoundError(fixture)
    if args.dry_run:
        for c in cases:
            print(f'\n{c["id"]} {c["name"]}\n{c["question"]}\n附件：{c["fixtures"]}')
        print("\n预览完成，未调用模型。评分标准见cases.json，不会传给Agent。")
        return 0
    return run_batch(args, cases)


if __name__ == "__main__":
    raise SystemExit(main())
