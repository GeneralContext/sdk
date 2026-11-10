"""The `generalcontext` command and prompt improvement, offline: a fake collector and scripted models.

    python -m unittest test_cli
"""

import contextlib
import copy
import io
import json
import os
import tempfile
import threading
import unittest
import unittest.mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from generalcontext import cli, evals, improve
from test_evals import TASK, scripted

EVALS_KEY, INGEST_KEY = "gck_evals", "gck_ingest"
BENCHMARK = {"format": evals.BENCHMARK_FORMAT, "tasks": [
    dict(TASK, id="t1", split="train"), dict(TASK, id="t2", split="train"), dict(TASK, id="t3", split="held_out")]}
FORBIDDEN = {"error": "This key cannot download the benchmark. On API keys, create a key that can run the benchmark."}


class Collector(BaseHTTPRequestHandler):
    posted = []

    def key(self):
        return (self.headers.get("Authorization") or "").replace("Bearer ", "")

    def do_GET(self):
        key = self.key()
        if key not in (EVALS_KEY, INGEST_KEY):
            return self.reply(401, {"error": "Unknown key."})
        if self.path.startswith("/api/platform-collect/status"):
            session = None
            if "session=" in self.path:
                found = "session=chat%3A1" in self.path
                session = {"id": "chat:1", "found": found, "calls": 3 if found else 0,
                           "models": ["gpt-test"] if found else [], "received_at": None}
            return self.reply(200, {"workspace": "elise", "key": {"label": "ci", "scopes": ["evals"] if key == EVALS_KEY else ["ingest"]},
                                    "last_call_at": "2026-11-09T00:00:00Z", "last_source": "sdk", "session": session})
        if self.path == "/api/platform-collect/benchmark":
            return self.reply(200, BENCHMARK) if key == EVALS_KEY else self.reply(403, FORBIDDEN)
        self.reply(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.key() != EVALS_KEY:
            return self.reply(403, FORBIDDEN)
        Collector.posted.append((self.path, body))
        self.reply(200, {"recorded": len(body["runs"]), "unknown": []})

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class CliTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Collector)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def run_cli(self, *argv, key=EVALS_KEY, url=None):
        out, err = io.StringIO(), io.StringIO()
        env = {"GENERALCONTEXT_API_KEY": key or "", "GENERALCONTEXT_BASE_URL": url or self.url}
        with unittest.mock.patch.dict(os.environ, env), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_verify_prints_workspace_scopes_and_last_call(self):
        code, out, _ = self.run_cli("verify", key=INGEST_KEY)
        self.assertEqual(code, 0)
        self.assertIn("Workspace elise: key ci (sends traffic).", out)
        self.assertIn("from sdk", out)

    def test_verify_session_found_and_not_found(self):
        code, out, _ = self.run_cli("verify", "--session", "chat:1")
        self.assertEqual(code, 0)
        self.assertIn("Conversation chat:1: arrived, 3 calls (gpt-test).", out)
        code, out, _ = self.run_cli("verify", "--session", "chat:2")
        self.assertEqual(code, 1)
        self.assertIn("Conversation chat:2: not found yet.", out)

    def test_verify_json(self):
        code, out, _ = self.run_cli("verify", "--json", "--session", "chat:2")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["session"]["found"], False)
        code, out, _ = self.run_cli("verify", "--json")
        self.assertEqual((code, json.loads(out)["workspace"]), (0, "elise"))

    def test_config_errors_exit_2_and_a_bad_key_exits_1(self):
        self.assertEqual(self.run_cli("verify", key="")[0], 2)
        code, _, err = self.run_cli("verify", url="http://example.com")
        self.assertEqual(code, 2)
        self.assertIn("not https", err)
        code, _, err = self.run_cli("verify", key="gck_wrong")
        self.assertEqual(code, 1)
        self.assertIn("Unknown key.", err)
        self.assertEqual(self.run_cli("nonsense")[0], 2)
        self.assertEqual(self.run_cli()[0], 2)

    def test_benchmark_pull_writes_the_file_and_counts_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "b.json")
            code, out, _ = self.run_cli("benchmark", "pull", "--out", path)
            with open(path) as handle:
                self.assertEqual(json.load(handle), BENCHMARK)
        self.assertEqual(code, 0)
        self.assertIn("3 tasks (1 held_out, 2 train).", out)

    def test_benchmark_pull_with_an_ingest_key_explains_the_403(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "b.json")
            code, _, err = self.run_cli("benchmark", "pull", "--out", path, key=INGEST_KEY)
            self.assertFalse(os.path.exists(path))
        self.assertEqual(code, 1)
        self.assertIn("create a key that can run the benchmark", err)

    def test_evals_run_delegates_and_uploads(self):
        Collector.posted.clear()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "b.json")
            with open(path, "w") as handle:
                json.dump(BENCHMARK, handle)
            with unittest.mock.patch.object(evals, "model", lambda spec: scripted({"text": "{}", "tool_calls": []})):
                code, out, _ = self.run_cli("evals", "run", path, "--agent", "openai:a", "--task", "t1", "--out", directory, "--upload")
        self.assertEqual(code, 0)
        self.assertIn("Uploaded 1 runs", out)
        self.assertEqual(Collector.posted[0][0], "/api/platform-collect/runs")
        self.assertEqual(self.run_cli("evals", "run")[0], 2)

    def test_upload_with_an_ingest_key_fails_with_exit_1(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "b.json")
            with open(path, "w") as handle:
                json.dump(BENCHMARK, handle)
            with unittest.mock.patch.object(evals, "model", lambda spec: scripted({"text": "{}", "tool_calls": []})):
                code, _, err = self.run_cli("evals", "run", path, "--agent", "openai:a", "--task", "none", "--out", directory,
                                            "--upload", key=INGEST_KEY)
        self.assertEqual(code, 1)
        self.assertIn("upload failed", err)


# ---- improve ------------------------------------------------------------------------------

GOOD = "Always ask when they paid before routing."
SECRET = "HELDOUT-SECRET-7731"


def benchmark():
    tasks = []
    for number, split in ((1, "train"), (2, "train"), (3, "held_out")):
        task = copy.deepcopy(TASK)
        task.update(id=f"t{number}", split=split, title=f"Task {number}")
        if split == "held_out":
            task.update(title=SECRET, expected=SECRET + " expected")
            task["scenario"]["opening"] = SECRET + " opening"
            task["scenario"]["context"] = SECRET + " context"
        tasks.append(task)
    return tasks


def agent_model(system, messages, tools):
    """Says 'good' when its instructions contain GOOD, 'okay' for a half rule, else 'bad'."""
    word = "good" if GOOD in system else "okay" if "half" in system else "bad"
    return {"text": word, "tool_calls": []}


def judge_model(system, messages, tools):
    content = messages[0]["content"]
    first = "agent: good" in content or "agent: okay" in content
    second = "agent: good" in content
    return {"text": json.dumps({"criteria": [{"id": "asks-date", "passed": first, "reason": "r1"},
                                             {"id": "routes", "passed": second, "reason": "r2"}]}), "tool_calls": []}


def person_model(system, messages, tools):
    return {"text": "", "tool_calls": [{"id": "1", "name": "leave", "arguments": {"message": "bye"}}]}


def proposer(*instructions):
    return scripted(*[{"text": json.dumps({"instructions": text, "rationale": f"try {i}"}), "tool_calls": []}
                      for i, text in enumerate(instructions)])


class ImproveTest(unittest.TestCase):
    def run_improve(self, proposer_model, rounds=2, optimize_on="train", tasks=None):
        logs = []
        report, best, results = improve.improve(tasks or benchmark(), "openai:a", agent_model, proposer_model, person_model,
                                                judge_model, "You are the front desk.", rounds=rounds, trials=2,
                                                optimize_on=optimize_on, log=logs.append)
        return report, best, results, logs

    def test_reference_agent_takes_an_instructions_override(self):
        model = scripted({"text": "hi", "tool_calls": []})
        evals.reference_agent(model, TASK, "Override.")([{"role": "user", "content": "x"}], evals.Toolbox(TASK))
        evals.reference_agent(model, TASK)([{"role": "user", "content": "x"}], evals.Toolbox(TASK))
        self.assertEqual([seen["system"] for seen in model.seen], ["Override.", "You are the front desk."])

    def test_a_better_candidate_is_kept_and_held_out_is_never_shown_to_the_proposer(self):
        model = proposer("You are the front desk. " + GOOD)
        report, best, results, logs = self.run_improve(model)
        self.assertEqual(best, "You are the front desk. " + GOOD)
        self.assertEqual(report["optimize"]["before"]["pass_rate"], 0.0)
        self.assertEqual(report["optimize"]["after"]["pass_rate"], 1.0)
        self.assertEqual(report["held_out"], {"before": {"runs": 2, "pass_rate": 0.0, "reward": 0.0},
                                              "after": {"runs": 2, "pass_rate": 1.0, "reward": 1.0}})
        self.assertEqual(len(model.seen), 1)  # A perfect score stops the rounds.
        shown = json.dumps(model.seen)
        self.assertNotIn(SECRET, shown)
        self.assertNotIn("t3", shown)
        self.assertIn("Task 1", shown)
        self.assertIn("r2", shown)  # The judge's reasons reach the proposer.
        self.assertIn("never copy", model.seen[0]["system"].lower())
        labels = {r["label"] for r in results}
        self.assertEqual(labels, {"openai:a (current instructions)", "openai:a (proposed instructions)"})

    def test_a_worse_or_equal_candidate_is_rejected(self):
        report, best, _, _ = self.run_improve(proposer("Something else entirely."))
        self.assertEqual(best, "You are the front desk.")
        self.assertFalse(report["changed"])
        self.assertEqual([r["kept"] for r in report["rounds"]], [False])
        self.assertEqual(report["held_out"]["before"], report["held_out"]["after"])

    def test_ties_on_pass_rate_are_broken_by_reward_and_rounds_continue(self):
        model = proposer("You are the front desk. half", "You are the front desk. " + GOOD)
        report, best, _, _ = self.run_improve(model)
        self.assertEqual([(r["pass_rate"], r["reward"], r["kept"]) for r in report["rounds"]],
                         [(0.0, 0.5, True), (1.0, 1.0, True)])
        self.assertEqual(best, "You are the front desk. " + GOOD)
        self.assertIn("half", model.seen[1]["messages"][0]["content"])

    def test_optimize_on_all_warns_and_has_no_held_out(self):
        report, _, _, logs = self.run_improve(proposer("x"), optimize_on="all")
        self.assertIn("no longer an honest check", logs[0])
        self.assertEqual((report["optimize_tasks"], report["held_out"]), (3, None))

    def test_base_instructions_come_from_the_file_or_the_most_common(self):
        tasks = benchmark()
        tasks[0]["agent"]["instructions"] = "Rare."
        self.assertEqual(improve.base_instructions(tasks), "You are the front desk.")
        for task in tasks:
            task["agent"]["instructions"] = ""
        with self.assertRaises(ValueError):
            improve.base_instructions(tasks)

    def test_the_command_writes_the_proposal_and_uploads(self):
        Collector.posted.clear()
        server = ThreadingHTTPServer(("127.0.0.1", 0), Collector)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        models = {"openai:a": agent_model, "openai:proposer": proposer("You are the front desk.\n" + GOOD),
                  "openai:person": person_model, "openai:judge": judge_model}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "b.json")
            with open(path, "w") as handle:
                json.dump({"format": evals.BENCHMARK_FORMAT, "tasks": benchmark()}, handle)
            out = os.path.join(directory, "improve")
            env = {"GENERALCONTEXT_API_KEY": EVALS_KEY, "GENERALCONTEXT_BASE_URL": f"http://127.0.0.1:{server.server_port}"}
            stdout = io.StringIO()
            with unittest.mock.patch.dict(os.environ, env), unittest.mock.patch.object(evals, "model", models.__getitem__), \
                    contextlib.redirect_stdout(stdout):
                code = cli.main(["improve", path, "--agent", "openai:a", "--proposer", "openai:proposer", "--person", "openai:person",
                                 "--judge", "openai:judge", "--out", out, "--upload"])
            files = {}
            for name in sorted(os.listdir(out)):
                with open(os.path.join(out, name)) as handle:
                    files[name] = handle.read()
        self.assertEqual(code, 0)
        self.assertEqual(sorted(files), ["instructions.diff", "instructions.md", "report.json", "runs.json"])
        self.assertEqual(files["instructions.md"], "You are the front desk.\n" + GOOD)
        self.assertIn("+" + GOOD, files["instructions.diff"])
        self.assertIn("-You are the front desk.", files["instructions.diff"])
        self.assertEqual(json.loads(files["report.json"])["held_out"]["after"]["pass_rate"], 1.0)
        self.assertEqual(json.loads(files["runs.json"])["format"], evals.RUNS_FORMAT)
        self.assertIn("Optimize set: 0% -> 100% pass", stdout.getvalue())
        self.assertIn("Held out: 0% -> 100% pass", stdout.getvalue())
        self.assertIn("instructions.diff", stdout.getvalue())
        self.assertEqual(Collector.posted[0][0], "/api/platform-collect/runs")
        self.assertEqual(len(Collector.posted[0][1]["runs"]), 4 + 4 + 2 + 2)

    def test_the_command_without_instructions_exits_2(self):
        tasks = benchmark()
        for task in tasks:
            task["agent"]["instructions"] = ""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "b.json")
            with open(path, "w") as handle:
                json.dump({"format": evals.BENCHMARK_FORMAT, "tasks": tasks}, handle)
            with contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(cli.main(["improve", path, "--agent", "openai:a"]), 2)
        self.assertIn("--instructions", err.getvalue())


if __name__ == "__main__":
    unittest.main()
