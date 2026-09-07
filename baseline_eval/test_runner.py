"""离线测试记录器，不测试真实Agent能力，不需要API。"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("run.py")
spec = importlib.util.spec_from_file_location("eval_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

FAKE_MAIN = '''
from pathlib import Path
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from app.api.monitor import monitor

@tool
def internet_search(query: str):
    """Offline test tool."""
    return {"results": [{"url": "https://example.invalid", "content": "OFFLINE TEST"}]}

model = FakeMessagesListChatModel(responses=[AIMessage(content="OFFLINE TEST",
    usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})])

async def execute(question):
    await internet_search.ainvoke({"query": question})
    return await model.ainvoke(question)

main_agent = RunnableLambda(execute)

async def run_deep_agent(question, session_id):
    answer = await main_agent.ainvoke(question, config={"configurable":{"thread_id": session_id}})
    folder = Path(__file__).parents[1] / "output" / f"session_{session_id}"
    folder.mkdir(parents=True)
    (folder / "T02_analysis.md").write_text("OFFLINE TEST", encoding="utf-8")
    monitor._emit("task_result", "done", {"result": answer.content})
'''


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = self.root / "project"
        for name in ("app", "app/api", "app/agent"):
            p = self.project / name
            p.mkdir(parents=True, exist_ok=True)
            (p / "__init__.py").write_text("", encoding="utf-8")
        (self.project / "app/api/monitor.py").write_text(
            'class Monitor:\n    def _emit(self, event_type, message, data=None):\n        pass\nmonitor=Monitor()\n', encoding="utf-8")
        self.folder = self.root / "run"
        self.folder.mkdir()
        runner.write_json(self.folder / "input.json", {"question": "offline question", "thread_id": "offline_test"})

    def tearDown(self):
        self.temp.cleanup()

    def launch(self, source):
        (self.project / "app/agent/main_agent.py").write_text(source, encoding="utf-8")
        return subprocess.run([sys.executable, str(SCRIPT), "--_worker", str(self.folder),
                               "--project-root", str(self.project)], capture_output=True, text=True, timeout=20)

    def test_nested_callbacks_and_full_tool_output_are_saved(self):
        p = self.launch(FAKE_MAIN)
        self.assertEqual(p.returncode, 0, p.stderr)
        m = runner.metrics(self.folder)
        self.assertEqual(m["search_call_count_observed"], 1)
        self.assertEqual(m["llm_call_count_observed"], 1)
        self.assertEqual(m["total_tokens"], 10)
        self.assertTrue(m["answer_received"])
        self.assertEqual(runner.read_json(self.folder / "search_results.json")[0]["output"]["results"][0]["content"], "OFFLINE TEST")
        self.assertTrue((self.project / "app/output/session_offline_test/T02_analysis.md").exists())

    def test_missing_usage_is_unknown_instead_of_zero(self):
        recorder = runner.Recorder(self.folder)
        recorder.emit("trace.jsonl", "llm_start", run_id="one")
        recorder.emit("trace.jsonl", "llm_end", run_id="one", usage={"total_tokens": 10})
        recorder.emit("trace.jsonl", "llm_start", run_id="two")
        recorder.emit("trace.jsonl", "llm_end", run_id="two", usage=None)
        m = runner.metrics(self.folder)
        self.assertIsNone(m["total_tokens"])
        self.assertEqual(m["observed_total_tokens"], 10)

    def test_swallowed_error_remains_visible(self):
        p = self.launch(FAKE_MAIN + '\nasync def run_deep_agent(question, session_id):\n    monitor._emit("error", "simulated swallowed error")\n')
        self.assertEqual(p.returncode, 0)
        m = runner.metrics(self.folder)
        self.assertEqual(m["monitor_error_count"], 1)
        self.assertFalse(m["answer_received"])

    def test_initialization_failure_has_explicit_record(self):
        p = self.launch('raise RuntimeError("offline initialization failure")\n')
        self.assertNotEqual(p.returncode, 0)
        self.assertEqual(runner.read_json(self.folder / "worker_status.json")["state"], "initialization_error")

    def test_pending_ratings_and_incomplete_batch_do_not_claim_pass_rate(self):
        batch = self.root / "batch"
        folder = batch / "T02_r1"
        folder.mkdir(parents=True)
        runner.write_json(batch / "manifest.json", {"cases": ["T02", "T03"], "repeat": 1})
        runner.write_json(folder / "result.json", {"case_id": "T02", "repeat": 1,
            "execution_state": "returned_with_answer", "expected_files_exist": {"report": True},
            "search_call_count_observed": 0, "total_tokens": None})
        runner.write_json(folder / "scores.json", {"ratings": [{"passed": None}]})
        self.assertIsNone(runner.summarize(batch)["results"][0]["quality_pass"])
        runner.write_json(folder / "scores.json", {"ratings": [{"passed": True}]})
        summary = runner.summarize(batch)
        self.assertTrue(summary["results"][0]["quality_pass"])
        self.assertIsNone(summary["quality_pass_rate"])

    def test_parent_timeout_saves_result_and_summary(self):
        # 给完整批次外壳使用临时副本，测试产物不混入交付包。
        import shutil
        harness = self.root / "baseline_eval"
        shutil.copytree(SCRIPT.parent, harness, ignore=shutil.ignore_patterns("results", "__pycache__"))
        (self.project / "app/agent/main_agent.py").write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
        p = subprocess.run([sys.executable, str(harness / "run.py"), "--project-root", str(self.project),
                            "--case", "T02", "--timeout", "0.2"], capture_output=True, text=True, timeout=20)
        self.assertEqual(p.returncode, 0, p.stderr)
        result_path = next((harness / "results").glob("*/T02_r1/result.json"))
        self.assertEqual(runner.read_json(result_path)["execution_state"], "timeout")
        self.assertFalse(runner.read_json(result_path.parent.parent / "summary.json")["results"][0]["quality_pass"])


if __name__ == "__main__":
    unittest.main()
