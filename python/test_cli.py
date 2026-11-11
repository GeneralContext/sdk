"""The `generalcontext` command, offline, against a fake collector.

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

from generalcontext import cli, evals
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
                found = "session=chat%3A1" in self.path or "session=chat%3A3" in self.path
                session = {"id": "chat:1", "found": found, "calls": (1 if "chat%3A3" in self.path else 3) if found else 0,
                           "models": ["gpt-test"] if found else [], "received_at": None}
            return self.reply(200, {"workspace": "acme", "key": {"label": "ci", "scopes": ["evals"] if key == EVALS_KEY else ["ingest"]},
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
        self.assertIn('Workspace acme. Key "ci" sends traffic.', out)
        self.assertIn("from sdk", out)

    def test_verify_session_found_and_not_found(self):
        code, out, _ = self.run_cli("verify", "--session", "chat:1")
        self.assertEqual(code, 0)
        self.assertIn("Conversation chat:1: arrived, 3 calls (gpt-test).", out)
        code, out, _ = self.run_cli("verify", "--session", "chat:3")
        self.assertIn("Conversation chat:3: arrived, 1 call (gpt-test).", out)
        code, out, _ = self.run_cli("verify", "--session", "chat:2")
        self.assertEqual(code, 1)
        self.assertIn("Conversation chat:2: not found yet.", out)

    def test_verify_json(self):
        code, out, _ = self.run_cli("verify", "--json", "--session", "chat:2")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out)["session"]["found"], False)
        code, out, _ = self.run_cli("verify", "--json")
        self.assertEqual((code, json.loads(out)["workspace"]), (0, "acme"))

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


if __name__ == "__main__":
    unittest.main()
