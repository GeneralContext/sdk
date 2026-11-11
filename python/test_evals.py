"""The replay eval harness, offline: fake models stand in for the agent, the person and the judge.

    python -m unittest test_evals -v
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
import unittest.mock

from generalcontext import evals

TASK = {
    "format": "generalcontext.task/v1", "id": "tst_1", "slug": "double-charge", "split": "train",
    "scenario": {
        "channel": "sms", "context": "SMS from +1 612 555 0177: I was charged twice.", "opening": "I was charged twice.",
        "requester": {"brief": "You were charged twice during the checkout incident.", "turns": ["It was order 10442."],
                      "facts": [{"id": "order", "text": "It was order 10442.", "disclose": "when_asked", "terms": ["which", "order"]},
                                {"id": "card", "text": "You paid by card.", "disclose": "volunteer", "terms": []}]},
    },
    "world": {
        "tools": [{"name": "orders__get", "description": "Read an order", "parameters": {"type": "object", "properties": {"order": {"type": "string"}}}, "source": "captured"}],
        "observations": [
            {"tool": "orders__get", "arguments": {"order": "10442"}, "result": {"payments": [{"id": "pay_9120", "amount": 129}, {"id": "pay_9121", "amount": 129}]}, "error": False},
            {"tool": "customers__search", "arguments": {"phone": "+16125550177", "store": "s02"}, "result": {"customer": "c0303"}, "error": False},
        ],
    },
    "agent": {"instructions": "You are the support desk.", "tools": ["orders__get"]},
    "policies": [{"id": "p1", "title": "Refunds", "text": "Only duplicate charges are refunded."}],  # older tasks carried these; still read, unused
    "expected": "Ask which order it was, then route a refund request.",
    "rubric": {"facts": ["Order 10442 charged twice during the incident."], "criteria": [{"id": "asks-order", "claim": "Asks which order it was."}, {"id": "routes", "claim": "Routes a refund."}]},
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
        self.assertEqual(box.call("orders__get", {"order": " 10442 "}), {"payments": [{"id": "pay_9120", "amount": 129}, {"id": "pay_9121", "amount": 129}]})
        # Phone formatting differs and an extra argument is missing: still the recorded answer.
        self.assertEqual(box.call("customers__search", {"phone": "(612) 555-0177"}), {"customer": "c0303"})
        missing = box.call("orders__get", {"order": "99999"})
        self.assertIn("No recorded result", missing["error"])
        self.assertEqual(missing["recorded_arguments"], [{"order": "10442"}])
        write = box.call("tasks__create", {"title": "Refund duplicate"})
        self.assertTrue(write["ok"])
        self.assertEqual(box.fidelity(), {"exact": 1, "close": 1, "unrecorded": 2, "unrecorded_reads": 1})
        self.assertEqual([c["write"] for c in box.calls], [False, False, False, True])

    def test_tools_offered_include_called_tools_without_a_spec(self):
        names = [spec["name"] for spec in evals.Toolbox(TASK).specs]
        self.assertEqual(names, ["orders__get", "customers__search"])
        self.assertEqual(evals.Toolbox(TASK).anthropic_tools()[0]["input_schema"]["properties"], {"order": {"type": "string"}})

    def test_recorded_errors_replay_as_errors(self):
        task = {"world": {"observations": [{"tool": "x__get", "arguments": {}, "result": "Timeout", "error": True}]}}
        self.assertEqual(evals.Toolbox(task).call("x__get", {}), {"error": "Timeout"})

    def test_spoken_reads_messages_to_the_person_only(self):
        self.assertEqual(evals.spoken({"name": "sms__CreateMessage", "write": True, "arguments": {"Body": "Hi"}}), "Hi")
        self.assertEqual(evals.spoken({"name": "outlook__reply-mail", "write": True, "arguments": {"body": {"content": "Hello"}}}), "Hello")
        self.assertIsNone(evals.spoken({"name": "tasks__create", "write": True, "arguments": {"title": "x", "description": "y"}}))


class Person(unittest.TestCase):
    def test_the_person_knows_every_fact_and_when_to_share_it(self):
        model = scripted({"text": "", "tool_calls": [{"id": "1", "name": "reply", "arguments": {"message": "ok"}}]})
        person = evals.Person(TASK, model)
        person.respond("Sorry about that, let me look.")
        notes = model.seen[-1]["system"]
        self.assertIn("It was order 10442. Tell them this only if they ask about it.", notes)
        self.assertIn("You paid by card. Tell them this without waiting to be asked.", notes)
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
    @staticmethod
    def agent_model(self=None):
        # Turn 1: look up the order, then ask which one. Turn 2: route a refund and confirm by SMS.
        return scripted(
            {"text": "", "tool_calls": [{"id": "a", "name": "orders__get", "arguments": {"order": "10442"}}]},
            {"text": "", "tool_calls": [{"id": "b", "name": "sms__CreateMessage", "arguments": {"Body": "Which order was it?"}}]},
            {"text": "Asked for the date.", "tool_calls": []},
            {"text": "", "tool_calls": [{"id": "c", "name": "tasks__create", "arguments": {"title": "Refund duplicate on 10442"}},
                                        {"id": "d", "name": "sms__CreateMessage", "arguments": {"Body": "I sent this to billing."}}]},
            {"text": "Routed.", "tool_calls": []},
        )

    def test_a_run_grades_every_criterion_and_reports_fidelity(self):
        person = scripted(
            {"text": "", "tool_calls": [{"id": "1", "name": "reply", "arguments": {"message": "Order 10442."}}]},
            {"text": "", "tool_calls": [{"id": "2", "name": "leave", "arguments": {"message": "Thanks."}}]},
        )
        judge = scripted({"text": json.dumps({"criteria": [{"id": "asks-order", "passed": True, "reason": "asked"}, {"id": "routes", "passed": True, "reason": "task"}]}), "tool_calls": []})
        agent_model = self.agent_model()
        result = evals.run_task(TASK, evals.reference_agent(agent_model, TASK), person, judge)
        self.assertTrue(result["passed"])
        self.assertEqual(result["reward"], 1.0)
        self.assertEqual([t["from"] for t in result["transcript"]], ["person", "agent", "person", "agent", "person"])
        self.assertEqual(agent_model.seen[0]["system"], "You are the support desk.")
        self.assertIn("tasks__create", json.dumps(judge.seen[0]["messages"]))
        self.assertIn("Order 10442 charged twice during the incident.", judge.seen[0]["messages"][0]["content"])
        self.assertNotIn("Only duplicate charges are refunded.", judge.seen[0]["messages"][0]["content"], "an older task's policies are not graded")
        self.assertEqual(result["fidelity"]["unrecorded_reads"], 0)

    def test_a_criterion_the_judge_skips_fails(self):
        person = scripted({"text": "", "tool_calls": [{"id": "1", "name": "leave", "arguments": {"message": "bye"}}]})
        judge = scripted({"text": '{"criteria": [{"id": "asks-order", "passed": true, "reason": "x"}]}', "tool_calls": []})
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
                return scripted({"text": '{"criteria": [{"id": "asks-order", "passed": true, "reason": "x"}, {"id": "routes", "passed": false, "reason": "y"}]}', "tool_calls": []})
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
                             " 'tool_calls': [{'name': 'orders__get', 'arguments': {'order': '10442'}}]}}))\n")
            os.chmod(fake, 0o700)
            with unittest.mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test", "PATH": directory + os.pathsep + os.environ["PATH"]}):
                reply = evals.model("claude-code:claude-haiku-5-5")("system", [{"role": "user", "content": "hi"}], evals.Toolbox(TASK).specs)
        self.assertEqual(reply["text"], "no key")
        self.assertEqual([(c["name"], c["arguments"]) for c in reply["tool_calls"]], [("orders__get", {"order": "10442"})])


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


# The same task in the revised shape. TASK above stays in the older shape to prove both are read.
NEW_TASK = {
    "format": "generalcontext.task/v1", "id": "tst_2", "slug": "double-charge-new", "split": "held_out", "origin": "failure",
    "tags": ["scope"], "prompt": TASK["scenario"]["opening"], "expected": TASK["expected"],
    "agent": TASK["agent"],
    "environment": {"kind": "replay", "channel": "sms", "context": TASK["scenario"]["context"],
                    "person": TASK["scenario"]["requester"], "tools": TASK["world"]["tools"], "observations": TASK["world"]["observations"]},
    "grader": {"kind": "rubric", **TASK["rubric"]},
}
CONTRACT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "contract")


class Counter:
    """A toy environment that is not a conversation: the agent increments a counter through a tool."""

    def __init__(self, task):
        self.count = task["environment"].get("start", 0)
        self.tools = [{"name": "increment", "description": "Add one.", "parameters": {"type": "object", "properties": {}}}]
        self.closed = False

    def call(self, name, arguments):
        if name != "increment":
            raise KeyError(name)
        self.count += 1
        return {"count": self.count}

    def start(self):
        return "Increment the counter to 3."

    def respond(self, said):
        return None, True

    def state(self):
        return {"count": self.count}

    def close(self):
        self.closed = True


def counter_check(task, episode):
    target = task["grader"]["target"]
    count = (episode.get("state") or {}).get("count")
    if count is None:  # A reference run has no live state: count the increments it made.
        count = sum(1 for call in episode["tool_calls"] if call["name"] == "increment")
    return {"reward": 1.0 if count == target else 0.0, "passed": count == target, "note": f"count {count}"}


def counting_agent(times):
    def agent(messages, toolbox):
        for _ in range(times):
            toolbox.call("increment", {})
        return "done"
    return agent


COUNTER_TASK = {"format": "generalcontext.task/v1", "id": "tst_c", "slug": "counter", "origin": "success", "split": "train",
                "prompt": "Count to 3.", "environment": {"kind": "counter", "start": 0}, "grader": {"kind": "counter-check", "target": 3}}


def all_pass_judge():
    return lambda system, messages, tools: {"text": json.dumps({"criteria": [
        {"id": item.split(":")[0].strip("- "), "passed": True, "reason": "ok"}
        for item in messages[0]["content"].split("Criteria:\n")[1].splitlines()]}), "tool_calls": []}


class TaskShape(unittest.TestCase):
    def test_old_shape_maps_to_the_new_one(self):
        view = evals._task_view(dict(TASK, set="fees", failure_modes=["scope"], status="requested"))
        self.assertEqual(view["prompt"], "I was charged twice.")
        self.assertEqual(view["environment"]["kind"], "replay")
        self.assertEqual(view["environment"]["context"], TASK["scenario"]["context"])
        self.assertEqual(view["environment"]["person"], TASK["scenario"]["requester"])
        self.assertEqual(view["environment"]["observations"], TASK["world"]["observations"])
        self.assertEqual(view["grader"], {"kind": "rubric", **TASK["rubric"]})
        self.assertEqual(view["tags"], ["fees", "scope"])
        self.assertEqual(view["status"], "building")
        self.assertNotIn("scenario", view)

    def test_new_shape_keeps_unknown_fields(self):
        view = evals._task_view(COUNTER_TASK)
        self.assertEqual(view["environment"], {"kind": "counter", "start": 0})
        self.assertEqual(view["grader"]["target"], 3)

    def test_contract_tasks_validate(self):
        for name in ("task-replay.json", "task-custom.json"):
            with open(os.path.join(CONTRACT, name)) as handle:
                task = json.load(handle)
            view = evals._task_view(task)
            self.assertEqual(view["format"], evals.TASK_FORMAT)
            self.assertIn(view["origin"], ("failure", "success"))
            self.assertTrue(view["prompt"])
            self.assertTrue(view["reference"])
            self.assertIn(view["reference"][0]["expect"], ("pass", "fail"))


class NewShape(unittest.TestCase):
    def test_new_shape_replay_task_end_to_end(self):
        person = scripted({"text": "", "tool_calls": [{"id": "1", "name": "reply", "arguments": {"message": "Order 10442."}}]},
                          {"text": "", "tool_calls": [{"id": "2", "name": "leave", "arguments": {"message": "Thanks."}}]})
        judge = all_pass_judge()
        result = evals.run_task(NEW_TASK, evals.reference_agent(EndToEnd.agent_model(None), NEW_TASK), person, judge)
        self.assertTrue(result["passed"])
        self.assertEqual(result["origin"], "failure")
        self.assertEqual(result["fidelity"]["unrecorded_reads"], 0)
        self.assertEqual(result["transcript"][0], {"from": "person", "text": "I was charged twice."})

    def test_positional_signature_is_unchanged(self):
        person = scripted({"text": "", "tool_calls": [{"id": "1", "name": "leave", "arguments": {"message": "bye"}}]})
        result = evals.run_task(TASK, lambda messages, toolbox: "Hello", person, all_pass_judge(), 2)
        self.assertTrue(result["passed"])
        self.assertIsNone(result["origin"])
        for key in ("passed", "reward", "criteria", "transcript", "tool_calls", "fidelity", "error", "origin"):
            self.assertIn(key, result)

    def test_custom_environment_and_grader(self):
        envs = []

        def make(task):
            envs.append(Counter(task))
            return envs[-1]
        good = evals.run_task(COUNTER_TASK, counting_agent(3), None, None, environment=make, grader=counter_check)
        self.assertTrue(good["passed"])
        self.assertEqual(good["reward"], 1.0)
        self.assertEqual(good["state"], {"count": 3})
        self.assertIsNone(good["fidelity"])
        self.assertEqual(good["origin"], "success")
        self.assertEqual([c["name"] for c in good["tool_calls"]], ["increment"] * 3)
        self.assertTrue(envs[-1].closed)
        self.assertIn("count 3", evals.note(good))
        bad = evals.run_task(COUNTER_TASK, counting_agent(2), None, None, environment=make, grader=counter_check)
        self.assertFalse(bad["passed"])
        self.assertEqual(bad["reward"], 0.0)

    def test_custom_env_tools_offered_to_agents(self):
        seen = {}

        def agent(messages, toolbox):
            seen["openai"] = toolbox.openai_tools()
            seen["anthropic"] = toolbox.anthropic_tools()
            seen["error"] = toolbox.call("missing", {})
            return ""
        evals.run_task(COUNTER_TASK, agent, None, None, environment=Counter, grader=counter_check)
        self.assertEqual(seen["openai"][0]["function"]["name"], "increment")
        self.assertEqual(seen["anthropic"][0]["name"], "increment")
        self.assertIn("KeyError", seen["error"]["error"])

    def test_a_kind_without_code_is_skipped(self):
        with self.assertRaises(evals.SkipTask) as raised:
            evals.run_task(COUNTER_TASK, counting_agent(3), None, None)
        self.assertIn("'counter'", str(raised.exception))
        with self.assertRaises(evals.SkipTask) as raised:
            evals.run_task(COUNTER_TASK, counting_agent(3), None, None, environment=Counter)
        self.assertIn("'counter-check'", str(raised.exception))

    def test_cli_skips_unknown_kinds_and_filters_by_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "bench.json")
            with open(path, "w") as handle:
                json.dump({"format": evals.BENCHMARK_FORMAT, "tasks": [NEW_TASK, COUNTER_TASK]}, handle)
            output = io.StringIO()
            with unittest.mock.patch.object(evals, "model", lambda spec: all_pass_judge()), \
                    contextlib.redirect_stdout(output):
                evals.main([path, "--agent", "openai:a", "--out", directory])
            with open(os.path.join(directory, "runs.json")) as handle:
                runs = json.load(handle)["runs"]
            self.assertIn("SKIP  counter  environment kind 'counter' needs --environment", output.getvalue())
            self.assertEqual([(r["task"], r["origin"]) for r in runs], [("tst_2", "failure")])
            with unittest.mock.patch.object(evals, "model", lambda spec: all_pass_judge()), \
                    contextlib.redirect_stdout(io.StringIO()):
                evals.main([path, "--agent", "openai:a", "--out", directory, "--origin", "success"])
            with open(os.path.join(directory, "runs.json")) as handle:
                self.assertEqual(json.load(handle)["runs"], [])


class Check(unittest.TestCase):
    def write(self, directory, tasks):
        path = os.path.join(directory, "bench.json")
        with open(path, "w") as handle:
            json.dump({"format": evals.BENCHMARK_FORMAT, "tasks": tasks}, handle)
        return path

    def test_check_agrees_with_a_custom_grader(self):
        task = dict(COUNTER_TASK, reference=[
            {"label": "good", "expect": "pass", "tool_calls": [{"name": "increment"}] * 3},
            {"label": "short", "expect": "fail", "tool_calls": [{"name": "increment"}]}])
        rows = evals.check([task], grader=counter_check)
        self.assertEqual([row["agrees"] for row in rows], [True, True])
        self.assertIn("SKIP", "SKIP" if evals.check([task])[0]["skipped"] else "")

    def test_check_reports_a_failure_reference_the_judge_passes(self):
        task = dict(NEW_TASK, reference=[{"label": "production", "expect": "fail", "transcript": [
            {"from": "person", "text": "I was charged twice."}, {"from": "agent", "text": "Refunded it."}], "tool_calls": []}])
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, [task])
            output = io.StringIO()
            with unittest.mock.patch.object(evals, "model", lambda spec: all_pass_judge()), contextlib.redirect_stdout(output):
                self.assertEqual(evals.main(["check", path]), 1)
            self.assertIn("DISAGREE", output.getvalue())
            self.assertIn("1 reference run checked, 1 disagrees with expect.", output.getvalue())
            strict = scripted({"text": '{"criteria": [{"id": "asks-order", "passed": false, "reason": "no"}]}', "tool_calls": []})
            with unittest.mock.patch.object(evals, "model", lambda spec: strict), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evals.main(["check", path]), 0)

    def test_cli_delegates_check(self):
        from generalcontext import cli
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, [dict(COUNTER_TASK, reference=[{"expect": "pass", "tool_calls": [{"name": "increment"}] * 3}])])
            with unittest.mock.patch.object(evals, "load_callable", lambda spec, what: counter_check), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["evals", "check", path, "--grader", "x:y"]), 0)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["evals", "check", path]), 0)  # skipped, not a disagreement


    def test_a_judge_that_cannot_start_is_a_setup_error_not_a_traceback(self):
        def missing_key(spec):
            raise RuntimeError("Missing credentials")
        task = dict(NEW_TASK, reference=[{"label": "production", "expect": "fail", "transcript": [], "tool_calls": []}])
        with tempfile.TemporaryDirectory() as directory:
            path = self.write(directory, [task])
            error = io.StringIO()
            with unittest.mock.patch.object(evals, "model", missing_key), contextlib.redirect_stderr(error):
                from generalcontext import cli
                self.assertEqual(cli.main(["evals", "check", path]), 2)
            self.assertIn("set OPENAI_API_KEY", error.getvalue())


class Contract(unittest.TestCase):
    def test_contract_benchmark_runs_both_tasks(self):
        tasks = evals.load_tasks(os.path.join(CONTRACT, "benchmark.json"))
        with open(os.path.join(CONTRACT, "benchmark.json")) as handle:
            file = json.load(handle)
        summary = file["summary"]
        # The platform's export: these keys, and nothing the SDK has to guess.
        self.assertEqual(set(file), {"format", "workspace", "name", "exported_at", "exported_by", "summary", "tasks"})
        self.assertEqual(set(summary), {"tasks", "by_origin", "by_split", "by_environment", "by_grader", "by_tag"})
        self.assertEqual(summary["tasks"], len(tasks))
        replay, custom = tasks
        person = scripted({"text": "", "tool_calls": [{"id": "1", "name": "leave", "arguments": {"message": "Thanks."}}]})
        result = evals.run_task(replay, lambda messages, toolbox: (toolbox.call("orders__get", {"order": "10442"}), "Which order was it?")[1],
                                person, all_pass_judge())
        self.assertTrue(result["passed"])
        self.assertEqual(result["origin"], "failure")
        self.assertEqual(result["fidelity"]["exact"], 1)
        result = evals.run_task(custom, counting_agent(3), None, None, environment=Counter, grader=counter_check)
        self.assertTrue(result["passed"])
        self.assertEqual(result["origin"], "success")
        self.assertEqual([row["agrees"] for row in evals.check([custom], grader=counter_check)], [True])

    def test_contract_runs_carry_origin(self):
        with open(os.path.join(CONTRACT, "runs.json")) as handle:
            runs = json.load(handle)
        self.assertEqual(runs["format"], evals.RUNS_FORMAT)
        self.assertEqual({run["origin"] for run in runs["runs"]}, {"failure", "success"})


if __name__ == "__main__":
    unittest.main()
