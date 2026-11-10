"""The replay eval harness, offline: fake models stand in for the agent, the person and the judge.

    python -m unittest test_evals -v
"""

import json
import os
import tempfile
import unittest
import unittest.mock

from generalcontext import evals

TASK = {
    "format": "generalcontext.task/v1", "id": "tst_1", "slug": "late-fee", "split": "train",
    "scenario": {
        "channel": "sms", "context": "SMS from +1 612 555 0177: My late fee is wrong.", "opening": "My late fee is wrong.",
        "requester": {"brief": "You were charged a late fee during the portal outage.", "turns": ["It was the 5th."],
                      "facts": [{"id": "date", "text": "You tried to pay on the 5th.", "disclose": "when_asked", "terms": ["when", "date"]},
                                {"id": "unit", "text": "You live in unit 303.", "disclose": "volunteer", "terms": []}]},
    },
    "world": {
        "tools": [{"name": "ledger__get", "description": "Read a ledger", "parameters": {"type": "object", "properties": {"unit": {"type": "string"}}}, "source": "captured"}],
        "observations": [
            {"tool": "ledger__get", "arguments": {"unit": "303"}, "result": {"fees": [{"date": "11-06", "amount": 75}]}, "error": False},
            {"tool": "residents__search", "arguments": {"phone": "+16125550177", "property": "ly02"}, "result": {"resident": "t0303"}, "error": False},
        ],
    },
    "agent": {"instructions": "You are the front desk.", "tools": ["ledger__get"]},
    "policies": [{"id": "p1", "title": "Late fees", "text": "Only outage fees may be waived."}],
    "expected": "Ask when they paid, then route a waiver request.",
    "rubric": {"facts": ["Fee on 11-06 during outage."], "criteria": [{"id": "asks-date", "claim": "Asks when they paid."}, {"id": "routes", "claim": "Routes a waiver."}]},
}


def scripted(*replies):
    """A fake model that returns the given replies in order and records what it was shown."""
    seen = []

    def call(system, messages, tools):
        seen.append({"system": system, "messages": [dict(m) for m in messages], "tools": tools})
        reply = replies[min(len(seen) - 1, len(replies) - 1)]
        return reply(system, messages) if callable(reply) else reply
    call.seen = seen
    return call


class ReplayWorld(unittest.TestCase):
    def test_exact_close_and_unrecorded_calls(self):
        box = evals.Toolbox(TASK)
        self.assertEqual(box.call("ledger__get", {"unit": " 303 "}), {"fees": [{"date": "11-06", "amount": 75}]})
        # Phone formatting differs and an extra argument is missing: still the recorded answer.
        self.assertEqual(box.call("residents__search", {"phone": "(612) 555-0177"}), {"resident": "t0303"})
        missing = box.call("ledger__get", {"unit": "404"})
        self.assertIn("No recorded result", missing["error"])
        self.assertEqual(missing["recorded_arguments"], [{"unit": "303"}])
        write = box.call("planner__create-task", {"title": "Waive fee"})
        self.assertTrue(write["ok"])
        self.assertEqual(box.fidelity(), {"exact": 1, "close": 1, "unrecorded": 2, "unrecorded_reads": 1})
        self.assertEqual([c["write"] for c in box.calls], [False, False, False, True])

    def test_tools_offered_include_called_tools_without_a_spec(self):
        names = [spec["name"] for spec in evals.Toolbox(TASK).specs]
        self.assertEqual(names, ["ledger__get", "residents__search"])
        self.assertEqual(evals.Toolbox(TASK).anthropic_tools()[0]["input_schema"]["properties"], {"unit": {"type": "string"}})

    def test_recorded_errors_replay_as_errors(self):
        task = {"world": {"observations": [{"tool": "x__get", "arguments": {}, "result": "Timeout", "error": True}]}}
        self.assertEqual(evals.Toolbox(task).call("x__get", {}), {"error": "Timeout"})

    def test_spoken_reads_messages_to_the_person_only(self):
        self.assertEqual(evals.spoken({"name": "sms__CreateMessage", "write": True, "arguments": {"Body": "Hi"}}), "Hi")
        self.assertEqual(evals.spoken({"name": "outlook__reply-mail", "write": True, "arguments": {"body": {"content": "Hello"}}}), "Hello")
        self.assertIsNone(evals.spoken({"name": "planner__create-task", "write": True, "arguments": {"title": "x", "description": "y"}}))


class Person(unittest.TestCase):
    def test_the_person_knows_every_fact_and_when_to_share_it(self):
        model = scripted({"text": "", "tool_calls": [{"id": "1", "name": "reply", "arguments": {"message": "ok"}}]})
        person = evals.Person(TASK, model)
        person.respond("Sorry about that, let me look.")
        notes = model.seen[-1]["system"]
        self.assertIn("You tried to pay on the 5th. Tell them this only if they ask about it.", notes)
        self.assertIn("You live in unit 303. Tell them this without waiting to be asked.", notes)
        person.respond("What date did you try to pay?")
        self.assertEqual([m["role"] for m in model.seen[-1]["messages"]], ["user", "assistant", "user"])

    def test_leave_ends_the_conversation(self):
        person = evals.Person(TASK, scripted({"text": "", "tool_calls": [{"id": "1", "name": "leave", "arguments": {"message": "Thanks, bye"}}]}))
        self.assertEqual(person.respond("Done."), ("Thanks, bye", True))

    def test_without_a_brief_the_recorded_turns_replay(self):
        task = {"scenario": {"requester": {"brief": "", "turns": ["first", "second"]}}}
        person = evals.Person(task, scripted({"text": "never used", "tool_calls": []}))
        self.assertEqual(person.respond("a"), ("first", False))
        self.assertEqual(person.respond("b"), ("second", True))


class EndToEnd(unittest.TestCase):
    def agent_model(self):
        # Turn 1: look up the ledger, then ask for the date. Turn 2: route a waiver and confirm by SMS.
        return scripted(
            {"text": "", "tool_calls": [{"id": "a", "name": "ledger__get", "arguments": {"unit": "303"}}]},
            {"text": "", "tool_calls": [{"id": "b", "name": "sms__CreateMessage", "arguments": {"Body": "When did you try to pay?"}}]},
            {"text": "Asked for the date.", "tool_calls": []},
            {"text": "", "tool_calls": [{"id": "c", "name": "planner__create-task", "arguments": {"title": "Waive 11-06 fee"}},
                                        {"id": "d", "name": "sms__CreateMessage", "arguments": {"Body": "I sent this to your manager."}}]},
            {"text": "Routed.", "tool_calls": []},
        )

    def test_a_run_grades_every_criterion_and_reports_fidelity(self):
        person = scripted(
            {"text": "", "tool_calls": [{"id": "1", "name": "reply", "arguments": {"message": "The 5th."}}]},
            {"text": "", "tool_calls": [{"id": "2", "name": "leave", "arguments": {"message": "Thanks."}}]},
        )
        judge = scripted({"text": json.dumps({"criteria": [{"id": "asks-date", "passed": True, "reason": "asked"}, {"id": "routes", "passed": True, "reason": "task"}]}), "tool_calls": []})
        agent_model = self.agent_model()
        result = evals.run_task(TASK, evals.reference_agent(agent_model, TASK), person, judge)
        self.assertTrue(result["passed"])
        self.assertEqual(result["reward"], 1.0)
        self.assertEqual([t["from"] for t in result["transcript"]], ["person", "agent", "person", "agent", "person"])
        self.assertEqual(agent_model.seen[0]["system"], "You are the front desk.")
        self.assertIn("planner__create-task", json.dumps(judge.seen[0]["messages"]))
        self.assertIn("Only outage fees may be waived.", judge.seen[0]["messages"][0]["content"])
        self.assertEqual(result["fidelity"]["unrecorded_reads"], 0)

    def test_a_criterion_the_judge_skips_fails(self):
        person = scripted({"text": "", "tool_calls": [{"id": "1", "name": "leave", "arguments": {"message": "bye"}}]})
        judge = scripted({"text": '{"criteria": [{"id": "asks-date", "passed": true, "reason": "x"}]}', "tool_calls": []})
        result = evals.run_task(TASK, evals.reference_agent(self.agent_model(), TASK), person, judge)
        self.assertFalse(result["passed"])
        self.assertEqual(result["reward"], 0.5)

    def test_an_agent_that_raises_is_a_failed_run(self):
        def broken(messages, toolbox):
            raise RuntimeError("boom")
        result = evals.run_task(TASK, broken, scripted({"text": "", "tool_calls": []}), scripted({"text": "{}", "tool_calls": []}))
        self.assertFalse(result["passed"])
        self.assertEqual(result["error"], "RuntimeError: boom")

    def test_the_cli_writes_runs_for_import(self):
        def fake_model(spec):
            if "judge" in spec:
                return scripted({"text": '{"criteria": [{"id": "asks-date", "passed": true, "reason": "x"}, {"id": "routes", "passed": false, "reason": "y"}]}', "tool_calls": []})
            if "person" in spec:
                return scripted({"text": "", "tool_calls": [{"id": "1", "name": "leave", "arguments": {"message": "bye"}}]})
            return self.agent_model()
        original = evals.model
        evals.model = fake_model
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "bench.json")
                with open(path, "w") as handle:
                    json.dump({"format": evals.BENCHMARK_FORMAT, "tasks": [TASK]}, handle)
                evals.main([path, "--agent", "openai:agent", "--person", "openai:person", "--judge", "openai:judge", "--trials", "2", "--out", directory])
                with open(os.path.join(directory, "runs.json")) as handle:
                    runs = json.load(handle)
        finally:
            evals.model = original
        self.assertEqual(runs["format"], "generalcontext.runs/v1")
        self.assertEqual([(r["task"], r["passed"], r["reward"]) for r in runs["runs"]], [("tst_1", False, 0.5), ("tst_1", False, 0.5)])
        self.assertIn("failed: routes", runs["runs"][0]["note"])


class ClaudeCode(unittest.TestCase):
    def test_runs_the_cli_without_the_api_key_and_parses_tool_calls(self):
        # A stand-in `claude` that reports whether it could see ANTHROPIC_API_KEY.
        with tempfile.TemporaryDirectory() as directory:
            fake = os.path.join(directory, "claude")
            with open(fake, "w") as handle:
                handle.write("#!/usr/bin/env python3\nimport json, os, sys\nsys.stdin.read()\n"
                             "print(json.dumps({'structured_output': {'text': 'key' if os.environ.get('ANTHROPIC_API_KEY') else 'no key',"
                             " 'tool_calls': [{'name': 'ledger__get', 'arguments': {'unit': '303'}}]}}))\n")
            os.chmod(fake, 0o700)
            with unittest.mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test", "PATH": directory + os.pathsep + os.environ["PATH"]}):
                reply = evals.model("claude-code:claude-haiku-5-5")("system", [{"role": "user", "content": "hi"}], evals.Toolbox(TASK).specs)
        self.assertEqual(reply["text"], "no key")
        self.assertEqual([(c["name"], c["arguments"]) for c in reply["tool_calls"]], [("ledger__get", {"unit": "303"})])


class Upload(unittest.TestCase):
    def test_upload_posts_runs_with_the_key_and_refuses_plain_http(self):
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                seen.append((self.path, self.headers.get("Authorization"), json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
                data = b'{"recorded": 1, "unknown": []}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        body = {"format": evals.RUNS_FORMAT, "runs": [{"task": "tst_1", "passed": True}]}
        env = {"GENERALCONTEXT_API_KEY": "gck_test", "GENERALCONTEXT_BASE_URL": f"http://127.0.0.1:{server.server_port}"}
        with unittest.mock.patch.dict(os.environ, env):
            self.assertEqual(evals.upload(body), {"recorded": 1, "unknown": []})
        self.assertEqual(seen, [("/api/platform-collect/runs", "Bearer gck_test", body)])
        with self.assertRaises(ValueError):
            evals.upload(body, api_key="k", base_url="http://example.com")

    def test_the_cli_uploads_with_upload_and_needs_the_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bench.json")
            with open(path, "w") as handle:
                json.dump({"format": evals.BENCHMARK_FORMAT, "tasks": [TASK]}, handle)
            args = [path, "--agent", "openai:agent", "--task", "none", "--out", directory, "--upload"]
            with unittest.mock.patch.dict(os.environ, {"GENERALCONTEXT_API_KEY": ""}), self.assertRaises(SystemExit):
                evals.main(args)
            with unittest.mock.patch.dict(os.environ, {"GENERALCONTEXT_API_KEY": "gck_test"}), \
                    unittest.mock.patch.object(evals, "model", lambda spec: None), \
                    unittest.mock.patch.object(evals, "upload", return_value={"recorded": 0}) as upload:
                self.assertEqual(evals.main(args), 0)
            upload.assert_called_once_with({"format": evals.RUNS_FORMAT, "runs": []})

    def test_console_script_entry_point_exists(self):
        self.assertTrue(callable(evals.cli))


if __name__ == "__main__":
    unittest.main()
