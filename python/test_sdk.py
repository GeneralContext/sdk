"""The SDK records calls without changing them, never raises into the agent, and redacts before sending.

    python -m unittest discover -s sdk/python -p 'test_*.py'

One local server plays both General Context's collector and a model provider. Tests that
need the real `openai`, `anthropic` or `httpx` packages skip when they are not installed.
"""

import asyncio
import importlib.util
import json
import os
import threading
import time
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import generalcontext

HAS_OPENAI = importlib.util.find_spec("openai") is not None
HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None
HAS_HTTPX = importlib.util.find_spec("httpx") is not None
try:
    HAS_GENAI = importlib.util.find_spec("google.genai") is not None
except ModuleNotFoundError:
    HAS_GENAI = False
HAS_BOTO3 = importlib.util.find_spec("boto3") is not None
KEY = "gck_" + "a" * 48

if HAS_OPENAI:
    import openai
    EARLY_CLIENT = openai.OpenAI(api_key="sk-test", base_url="http://unused")  # Created before initialize().

CHAT = {"id": "c1", "object": "chat.completion", "created": 1, "model": "gpt-test",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Unit 402 is open."}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}}
CHAT_CHUNKS = [
    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Checking "}}]},
    {"choices": [{"index": 0, "delta": {"content": "units."}}]},
    {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                                         "function": {"name": "find_units", "arguments": "{\"be"}}]}}]},
    {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "ds\": 2}"}}]}}]},
    {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
    {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}},
]
for chunk in CHAT_CHUNKS:
    chunk.update(id="c2", object="chat.completion.chunk", created=1, model="gpt-test")
MESSAGE = {"id": "m1", "type": "message", "role": "assistant", "model": "claude-test", "stop_reason": "end_turn",
           "content": [{"type": "text", "text": "Tours run daily."}], "usage": {"input_tokens": 3, "output_tokens": 4}}
MESSAGE_EVENTS = [
    {"type": "message_start", "message": {"id": "m2", "type": "message", "role": "assistant", "model": "claude-test",
                                          "content": [], "usage": {"input_tokens": 3, "output_tokens": 0}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Booking "}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "now."}},
    {"type": "content_block_stop", "index": 0},
    {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "book", "input": {}}},
    {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"day\": \"mon\"}"}},
    {"type": "content_block_stop", "index": 1},
    {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
    {"type": "message_stop"},
]
RESPONSE = {"id": "r1", "object": "response", "created_at": 1, "model": "gpt-test", "status": "completed",
            "output": [{"type": "message", "id": "msg1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "Hi there.", "annotations": []}]}],
            "parallel_tool_calls": False, "tool_choice": "auto", "tools": []}
RESPONSE_EVENTS = [
    {"type": "response.created", "sequence_number": 0, "response": dict(RESPONSE, status="in_progress", output=[])},
    {"type": "response.completed", "sequence_number": 1, "response": RESPONSE},
]
GEMINI_CHUNKS = [
    {"candidates": [{"content": {"role": "model", "parts": [{"text": "Unit 402 "}]}, "index": 0}], "modelVersion": "gemini-test"},
    {"candidates": [{"content": {"role": "model", "parts": [{"text": "is open."}]}, "index": 0}], "modelVersion": "gemini-test"},
    {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "book", "args": {"day": "mon"}}}]},
                     "finishReason": "STOP", "index": 0}],
     "usageMetadata": {"promptTokenCount": 6, "candidatesTokenCount": 5, "totalTokenCount": 11}, "modelVersion": "gemini-test"},
]
GEMINI = {"candidates": [{"content": {"role": "model", "parts": [{"text": "Tours run daily."}]}, "finishReason": "STOP", "index": 0}],
          "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 4, "totalTokenCount": 7}, "modelVersion": "gemini-test"}


class Server(BaseHTTPRequestHandler):
    received = []  # (path, authorization, body) for collector requests
    status = 200
    providers = []   # Paths of model provider requests

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        if self.path.startswith("/api/platform-collect"):
            Server.received.append((self.path, self.headers.get("Authorization"), body))
            return self.reply(Server.status, {"recorded": len(body.get("runs") or []), "unknown": []}
                              if self.path.endswith("/runs") else {})
        Server.providers.append(self.path)
        path = self.path.split("?")[0]
        if body.get("model") == "bad" or "/models/missing:" in path:
            return self.reply(400, {"error": {"message": "bad model", "type": "invalid_request_error"}})
        if path.endswith("/chat/completions"):
            return self.sse([("", c) for c in CHAT_CHUNKS], done=True) if body.get("stream") else self.reply(200, CHAT)
        if path.endswith("/messages"):
            return self.sse([(e["type"], e) for e in MESSAGE_EVENTS]) if body.get("stream") else self.reply(200, MESSAGE)
        if path.endswith("/responses"):
            return self.sse([(e["type"], e) for e in RESPONSE_EVENTS]) if body.get("stream") else self.reply(200, RESPONSE)
        if path.endswith(":streamGenerateContent"):
            return self.sse([("", c) for c in GEMINI_CHUNKS])
        if path.endswith(":generateContent"):
            return self.reply(200, GEMINI)
        self.reply(404, {})

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def sse(self, events, done=False):
        lines = "".join((f"event: {name}\n" if name else "") + f"data: {json.dumps(data)}\n\n" for name, data in events)
        data = (lines + ("data: [DONE]\n\n" if done else "")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class Chat:
    def __init__(self, reply):
        self.reply = reply

    def create(self, **kwargs):
        if kwargs.get("model") == "bad":
            raise ValueError("bad model")
        return self.reply


class Client:
    def __init__(self, reply):
        self.chat = type("Chat", (), {"completions": Chat(reply)})()


class SdkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        os.environ["GENERALCONTEXT_REDACT_SALT"] = "shared-secret"
        generalcontext.initialize(api_key=KEY, base_url=cls.url, app_name="support-bot", instrument_http=True)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        self.assertTrue(generalcontext.flush(5))
        Server.received.clear()
        Server.providers.clear()
        Server.status = 200

    def calls(self):
        self.assertTrue(generalcontext.flush(5))
        return [call for path, _, body in Server.received if path.endswith("/sdk") for call in body["calls"]]

    def openai(self, cls_name="OpenAI"):
        return getattr(openai, cls_name)(api_key="sk-test", base_url=self.url + "/v1", max_retries=0)

    # Core behavior, with a stand-in client.

    def test_records_a_call_with_session_and_task_and_returns_the_reply_unchanged(self):
        reply = {"choices": [{"message": {"role": "assistant", "content": "Unit 402 is open."}}]}
        client = generalcontext.wrap(Client(reply))
        with generalcontext.session("chat:thread-1", user_id="u1", channel="chat", product="support"):
            with generalcontext.task("qualify"):
                result = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "2 beds?"}])
        self.assertIs(result, reply)
        self.assertTrue(generalcontext.flush(5))
        path, authorization, body = Server.received[0]
        self.assertEqual((path, authorization), ("/api/platform-collect/sdk", "Bearer " + KEY))
        self.assertEqual((body["sdk"], body["app"]), ("python/" + generalcontext.__version__, "support-bot"))
        call = body["calls"][0]
        self.assertEqual(call["api"], "openai-chat")
        self.assertEqual(call["conversation"], {"id": "chat:thread-1", "user_id": "u1", "channel": "chat", "product": "support"})
        self.assertEqual(call["task"], "qualify")
        self.assertEqual(call["request"]["messages"][0]["content"], "2 beds?")
        self.assertEqual((call["response"], call["error"]), (reply, None))

    def test_conversation_is_an_alias_of_session(self):
        self.assertIs(generalcontext.conversation, generalcontext.session)
        self.assertIs(generalcontext.with_session, generalcontext.session)

    def test_an_exception_is_recorded_and_reraised_unchanged(self):
        client = generalcontext.wrap(Client({}))
        with self.assertRaisesRegex(ValueError, "bad model"):
            client.chat.completions.create(model="bad", messages=[])
        call = self.calls()[0]
        self.assertEqual((call["response"], call["error"]), (None, "ValueError: bad model"))

    def test_a_collector_outage_never_reaches_the_agent(self):
        Server.status = 503
        client = generalcontext.wrap(Client({"ok": True}))
        self.assertEqual(client.chat.completions.create(messages=[]), {"ok": True})
        self.assertTrue(generalcontext.flush(5))
        self.assertFalse(generalcontext.report_outcome("s1", "success"))
        self.assertFalse(generalcontext.log_event("s1", "tour.booked"))

    def test_wrapping_twice_records_once(self):
        client = generalcontext.wrap(generalcontext.wrap(Client({"ok": True})))
        client.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
        self.assertEqual(len(self.calls()), 1)

    def test_redaction_replaces_personal_details_with_stable_placeholders(self):
        client = generalcontext.wrap(Client({"choices": []}), redact=True)
        text = "Text me at (612) 555-0177 or jordan@example.net, card 4111 1111 1111 1111."
        client.chat.completions.create(messages=[{"role": "user", "content": text}, {"role": "user", "content": "612-555-0177"}])
        sent = self.calls()[0]["request"]["messages"]
        for value in ("555-0177", "jordan@example.net", "4111"):
            self.assertNotIn(value, json.dumps(sent))
        self.assertIn("<email:", sent[0]["content"])
        self.assertIn("<card:", sent[0]["content"])
        first = sent[0]["content"].split("<phone:")[1][:6]
        self.assertEqual(sent[1]["content"], f"<phone:{first}>")

    def test_second_initialize_is_ignored(self):
        with self.assertLogs("generalcontext", "WARNING"):
            generalcontext.initialize(api_key="other")
        self.assertEqual(generalcontext._api_key(), KEY)

    def test_report_outcome_flushes_then_posts_to_outcomes(self):
        generalcontext.wrap(Client({})).chat.completions.create(messages=[])
        self.assertTrue(generalcontext.report_outcome("sms:thread-1", "success", note="booked"))
        self.assertEqual([path for path, _, _ in Server.received][-2:],
                         ["/api/platform-collect/sdk", "/api/platform-collect/outcomes"])
        _, authorization, body = Server.received[-1]
        self.assertEqual(authorization, "Bearer " + KEY)
        self.assertEqual(body, {"session_id": "sms:thread-1", "outcome": "success", "note": "booked"})
        self.assertFalse(generalcontext.report_outcome("sms:thread-1", "maybe"))

    def test_log_event_posts_to_events(self):
        self.assertTrue(generalcontext.log_event("sms:thread-1", "tour.booked", {"unit": "402", "beds": 2},
                                                 event_id="evt-1", timestamp="2026-11-09T00:00:00+00:00"))
        path, authorization, body = Server.received[-1]
        self.assertEqual((path, authorization), ("/api/platform-collect/events", "Bearer " + KEY))
        self.assertEqual(body, {"events": [{"event_id": "evt-1", "event_type": "tour.booked",
                                            "timestamp": "2026-11-09T00:00:00+00:00", "session_id": "sms:thread-1",
                                            "properties": {"unit": "402", "beds": 2}}]})
        generalcontext.log_event("s", "lease.signed")
        event = Server.received[-1][2]["events"][0]
        self.assertTrue(event["event_id"] and event["timestamp"])

    # Hardening.

    def using(self, **config):
        """Temporarily override _config entries."""
        old = {key: generalcontext._config[key] for key in config}
        generalcontext._config.update(config)
        self.addCleanup(generalcontext._config.update, old)

    def test_refuses_plain_http_to_other_hosts(self):
        self.assertTrue(generalcontext._secure("https://generalcontextlabs.com"))
        for url in ("http://localhost:1", "http://127.0.0.1:1", "http://[::1]:1"):
            self.assertTrue(generalcontext._secure(url))
        self.assertFalse(generalcontext._secure("http://generalcontextlabs.com"))
        self.using(base_url="http://example.com")
        generalcontext._warned_at.pop("http", None)
        with self.assertLogs("generalcontext", "WARNING"):
            generalcontext.wrap(Client({})).chat.completions.create(messages=[])
        self.assertEqual(generalcontext._sender.queue.qsize(), 0)
        self.assertFalse(generalcontext.report_outcome("s", "success"))

    def test_redirects_are_not_followed(self):
        class Second(BaseHTTPRequestHandler):
            hits = []

            def do_POST(self):
                Second.hits.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args):
                pass

        class Redirect(Second):
            def do_POST(self):
                self.send_response(302)
                self.send_header("Location", second_url + "/api/platform-collect/sdk")
                self.send_header("Content-Length", "0")
                self.end_headers()

        servers = [ThreadingHTTPServer(("127.0.0.1", 0), handler) for handler in (Second, Redirect)]
        for server in servers:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.addCleanup(server.shutdown)
        second_url = f"http://127.0.0.1:{servers[0].server_port}"
        self.using(base_url=f"http://127.0.0.1:{servers[1].server_port}")
        generalcontext.wrap(Client({})).chat.completions.create(messages=[])
        self.assertTrue(generalcontext.flush(5))
        self.assertFalse(generalcontext.log_event("s", "x"))
        self.assertEqual(Second.hits, [])

    def test_a_class_argument_is_recorded_by_name_and_never_breaks_the_call(self):
        class Answer:
            @classmethod
            def model_dump(cls, **kwargs):
                raise TypeError("unbound")
        client = generalcontext.wrap(Client({"ok": True}))
        self.assertEqual(client.chat.completions.create(messages=[], response_format=Answer), {"ok": True})
        self.assertTrue(self.calls()[0]["request"]["response_format"].endswith("Answer"))
        with mock.patch.object(generalcontext, "_start", side_effect=RuntimeError("boom")):
            self.assertEqual(client.chat.completions.create(messages=[]), {"ok": True})

    def test_oversized_call_is_dropped(self):
        client = generalcontext.wrap(Client({}))
        # Many strings under the clipping threshold, so compaction cannot bring the call under the limit.
        client.chat.completions.create(messages=[{"role": "user", "content": "x" * 150_000}] * 30)
        client.chat.completions.create(messages=[{"role": "user", "content": "small"}])
        calls = self.calls()
        self.assertEqual([c["request"]["messages"][0]["content"] for c in calls], ["small"])

    def test_long_text_is_clipped_and_inline_binary_omitted_without_touching_the_request(self):
        import base64
        data = base64.b64encode(b"\x07" * 450_000).decode()
        request = {"messages": [
            {"role": "user", "content": "s" * 4_100_000},
            {"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}},
                {"type": "input_audio", "input_audio": {"format": "wav", "data": data}}]}]}
        before = json.loads(json.dumps(request))
        client = generalcontext.wrap(Client({"output": [{"type": "image_generation_call", "id": "ig", "result": data}]}))
        client.chat.completions.create(**request)
        self.assertEqual(request, before)
        [call] = self.calls()
        text, parts = call["request"]["messages"]
        self.assertEqual(len(text["content"]), 120_000 + len("\n[... clipped 3980000 characters ...]\n"))
        self.assertIn("\n[... clipped 3980000 characters ...]\n", text["content"])
        image, url, audio = parts["content"]
        self.assertEqual(image["source"]["media_type"], "image/jpeg")
        self.assertRegex(image["source"]["data"], r"^\[omitted 600000 bytes sha256:[0-9a-f]{12}\]$")
        self.assertRegex(url["image_url"]["url"], r"^data:image/png;base64,\[omitted 600000 bytes sha256:[0-9a-f]{12}\]$")
        self.assertRegex(audio["input_audio"]["data"], r"^\[omitted 600000 bytes")
        self.assertRegex(call["response"]["output"][0]["result"], r"^\[omitted 600000 bytes")
        gemini = generalcontext.compact_for_recording({"inline_data": {"mime_type": "image/png", "data": data}})
        self.assertEqual(gemini["inline_data"]["mime_type"], "image/png")
        self.assertTrue(gemini["inline_data"]["data"].startswith("[omitted"))
        self.assertEqual(generalcontext.compact_for_recording({"id": "call_1", "name": "run"}), {"id": "call_1", "name": "run"})

    @unittest.skipUnless(HAS_OPENAI, "openai is not installed")
    def test_a_growing_responses_agent_loop_records_every_call_under_the_limit(self):
        import base64
        client = self.openai()
        image = lambda n: "data:image/png;base64," + base64.b64encode(b"\x01" * n).decode()
        history = [{"role": "user", "content": [
            {"type": "input_text", "text": "src " * 375_000},
            {"type": "input_image", "image_url": image(450_000)},
            {"type": "input_image", "image_url": image(450_003)}]}]
        for turn in range(20):
            request = {"model": "gpt-test", "input": json.loads(json.dumps(history)), "stream": True, "store": False}
            before = json.loads(json.dumps(request))
            for _ in client.responses.create(**request):
                pass
            self.assertEqual(request, before)
            history += [{"type": "function_call", "call_id": f"c{turn}", "name": "run", "arguments": "{}"},
                        {"type": "function_call_output", "call_id": f"c{turn}", "output": "ok"},
                        {"role": "user", "content": [{"type": "input_image", "image_url": image(300_000 + turn)}]}]
        calls = self.calls()
        self.assertEqual(len(calls), 20)
        for turn, call in enumerate(calls):
            self.assertLess(len(json.dumps(call)), generalcontext.MAX_CALL_BYTES)
            text, a, b = call["request"]["input"][0]["content"]
            self.assertIn("\n[... clipped 1380000 characters ...]\n", text["text"])
            self.assertRegex(a["image_url"], r"^data:image/png;base64,\[omitted 600000 bytes sha256:[0-9a-f]{12}\]$")
            self.assertRegex(b["image_url"], r"^data:image/png;base64,\[omitted 600004 bytes")
            self.assertEqual(len(call["request"]["input"]), 1 + 3 * turn)
            self.assertIsNotNone(call["response"])

    @unittest.skipUnless(HAS_ANTHROPIC, "anthropic is not installed")
    def test_anthropic_base64_image_block_is_recorded_as_a_placeholder(self):
        import anthropic, base64
        data = base64.b64encode(b"\x02" * 300_000).decode()
        messages = [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}},
            {"type": "text", "text": "what?"}]}]
        anthropic.Anthropic(api_key="sk-test", base_url=self.url, max_retries=0).messages.create(
            model="claude-test", max_tokens=5, messages=messages)
        self.assertEqual(messages[0]["content"][0]["source"]["data"], data)
        [call] = self.calls()
        source = call["request"]["messages"][0]["content"][0]["source"]
        self.assertEqual(source["media_type"], "image/png")
        self.assertRegex(source["data"], r"^\[omitted 400000 bytes sha256:[0-9a-f]{12}\]$")

    def test_nan_and_infinity_in_a_request_never_poison_the_batch(self):
        # The collector rejects NaN/Infinity (not JSON), which used to drop every call in the batch.
        client = generalcontext.wrap(Client({}))
        client.chat.completions.create(messages=[], temperature=float("nan"), extra_body={"x": float("-inf")})
        client.chat.completions.create(messages=[{"role": "user", "content": "next"}])
        self.assertTrue(generalcontext.flush(5))
        raw = [body for path, _, body in Server.received if path.endswith("/sdk")]
        json.dumps(raw, allow_nan=False)  # Raises if a NaN reached the wire.
        calls = self.calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["request"]["temperature"], "nan")

    def test_shared_and_circular_references_are_recorded_quickly(self):
        # A dict that refers to itself twice used to take 2**40 steps on the caller's thread.
        loop = {"name": "loop"}
        loop["a"] = loop
        loop["b"] = loop
        client = generalcontext.wrap(Client({"ok": True}))
        start = time.monotonic()
        self.assertEqual(client.chat.completions.create(messages=[], extra_body=loop), {"ok": True})
        self.assertLess(time.monotonic() - start, 1)
        recorded = self.calls()[0]["request"]["extra_body"]
        self.assertEqual((recorded["name"], recorded["a"], recorded["b"]), ("loop", "<circular>", "<circular>"))
        shared = {"k": "v"}
        client.chat.completions.create(messages=[shared, shared])
        self.assertEqual(self.calls()[-1]["request"]["messages"], [{"k": "v"}, {"k": "v"}])

    def test_batches_split_by_bytes(self):
        client = generalcontext.wrap(Client({}))
        big = [{"role": "user", "content": "y" * 150_000}] * 10
        for _ in range(4):
            client.chat.completions.create(messages=big)
        self.assertEqual(len(self.calls()), 4)
        batches = [body for path, _, body in Server.received if path.endswith("/sdk")]
        self.assertGreaterEqual(len(batches), 2)
        self.assertTrue(all(len(b["calls"]) <= 2 for b in batches))

    def test_before_send_can_modify_or_drop(self):
        def before_send(call):
            if call["request"].get("drop"):
                return None
            if call["request"].get("raise"):
                raise RuntimeError("nope")
            call["task"] = "changed"
            return call
        self.using(before_send=before_send)
        client = generalcontext.wrap(Client({}))
        client.chat.completions.create(messages=[], drop=True)
        client.chat.completions.create(messages=[], **{"raise": True})
        client.chat.completions.create(messages=[])
        calls = self.calls()
        self.assertEqual([c["task"] for c in calls], ["changed"])

    def test_outcomes_and_events_are_redacted_like_calls(self):
        self.using(scrub=generalcontext._redactor(True))
        generalcontext.wrap(Client({}), redact=True).chat.completions.create(messages=[])
        with generalcontext.session("jo@example.com"):
            generalcontext.wrap(Client({}), redact=True).chat.completions.create(messages=[])
        session_id = self.calls()[-1]["conversation"]["id"]
        self.assertTrue(generalcontext.report_outcome("jo@example.com", "fail", note="call 612-555-0177"))
        body = Server.received[-1][2]
        self.assertEqual(body["session_id"], session_id)
        self.assertNotIn("555-0177", body["note"])
        self.assertTrue(generalcontext.log_event("jo@example.com", "x", {"who": "jo@example.com", "n": 1}))
        event = Server.received[-1][2]["events"][0]
        self.assertEqual((event["session_id"], event["properties"]["n"]), (session_id, 1))
        self.assertNotIn("@", event["properties"]["who"])

    def test_redaction_is_fast_on_adversarial_input(self):
        scrub = generalcontext._redactor(True)
        start = time.monotonic()
        scrub("a." * 500000 + "@")
        self.assertLess(time.monotonic() - start, 1)

    def test_disabled_env_skips_initialize(self):
        with mock.patch.dict(os.environ, {"GENERALCONTEXT_DISABLED": "TRUE"}), \
                mock.patch.object(generalcontext, "_patch_sdks") as patch:
            with self.assertLogs("generalcontext", "INFO") as logs:
                generalcontext.initialize(api_key="other")
        patch.assert_not_called()
        self.assertIn("DISABLED", logs.output[0])
        self.assertEqual(generalcontext._api_key(), KEY)

    def test_fork_reset_gives_a_fresh_queue_and_no_thread(self):
        sender = generalcontext._Sender()
        sender.queue.put_nowait(({"x": "y"}, None, 10))
        sender.bytes, sender.thread = 10, threading.current_thread()
        old_queue, old_lock = sender.queue, sender.lock
        sender._reset()
        self.assertIsNot(sender.queue, old_queue)
        self.assertIsNot(sender.lock, old_lock)
        self.assertEqual((sender.bytes, sender.thread, sender.queue.qsize()), (0, None, 0))

    @unittest.skipUnless(hasattr(os, "fork"), "needs os.fork")
    def test_a_forked_child_records_and_sends_on_its_own(self):
        # Like gunicorn or Celery: the parent has a live sender thread, then forks.
        client = generalcontext.wrap(Client({"choices": []}))
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "parent"}])
        self.assertTrue(generalcontext.flush(5))
        Server.received.clear()
        pid = os.fork()
        if pid == 0:
            try:
                client.chat.completions.create(model="m", messages=[{"role": "user", "content": "child"}])
                os._exit(0 if generalcontext.flush(5) else 1)
            except BaseException:
                os._exit(2)
        _, status = os.waitpid(pid, 0)
        self.assertEqual(os.waitstatus_to_exitcode(status) if hasattr(os, "waitstatus_to_exitcode") else status >> 8, 0)
        contents = [m["content"] for _, _, body in Server.received for c in body["calls"] for m in c["request"]["messages"]]
        self.assertIn("child", contents)

    def test_without_a_key_outcomes_return_false_immediately(self):
        self.using(api_key=None)
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(generalcontext.report_outcome("s", "success"))
            self.assertFalse(generalcontext.log_event("s", "x"))

    # Real SDKs, patched by initialize().

    @unittest.skipUnless(HAS_OPENAI, "openai is not installed")
    def test_openai_clients_created_before_and_after_initialize_are_recorded(self):
        EARLY_CLIENT.base_url = self.url + "/v1"
        early = EARLY_CLIENT.chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": "a"}])
        late = self.openai().chat.completions.create(model="gpt-test", messages=[{"role": "user", "content": "b"}])
        self.assertEqual(early.choices[0].message.content, "Unit 402 is open.")
        self.assertEqual(late.model, "gpt-test")
        calls = self.calls()
        self.assertEqual([c["request"]["messages"][0]["content"] for c in calls], ["a", "b"])  # Once each, not via httpx too.
        self.assertEqual(calls[1]["response"]["usage"]["total_tokens"], 9)

    @unittest.skipUnless(HAS_OPENAI, "openai is not installed")
    def test_openai_streamed_chat_is_recorded_as_one_reassembled_reply(self):
        stream = self.openai().chat.completions.create(model="gpt-test", messages=[], stream=True,
                                                       stream_options={"include_usage": True})
        with stream:
            chunks = list(stream)
        self.assertEqual(len(chunks), len(CHAT_CHUNKS))
        self.assertEqual(chunks[0].choices[0].delta.content, "Checking ")
        response = self.calls()[0]["response"]
        message = response["choices"][0]["message"]
        self.assertEqual(message["content"], "Checking units.")
        self.assertEqual(message["tool_calls"], [{"id": "call_1", "type": "function",
                                                  "function": {"name": "find_units", "arguments": "{\"beds\": 2}"}}])
        self.assertEqual((response["model"], response["usage"]["total_tokens"]), ("gpt-test", 10))

    @unittest.skipUnless(HAS_OPENAI, "openai is not installed")
    def test_async_openai_plain_and_streamed(self):
        async def run():
            client = self.openai("AsyncOpenAI")
            with generalcontext.session("async-1"):
                reply = await client.chat.completions.create(model="gpt-test", messages=[])
                stream = await client.chat.completions.create(model="gpt-test", messages=[], stream=True)
                async with stream:
                    texts = [c.choices[0].delta.content async for c in stream if c.choices]
            return reply, texts
        reply, texts = asyncio.run(run())
        self.assertEqual(reply.choices[0].message.content, "Unit 402 is open.")
        self.assertEqual(texts[:2], ["Checking ", "units."])
        calls = self.calls()
        self.assertEqual([c["conversation"]["id"] for c in calls], ["async-1", "async-1"])
        self.assertEqual(calls[1]["response"]["choices"][0]["message"]["content"], "Checking units.")

    @unittest.skipUnless(HAS_OPENAI, "openai is not installed")
    def test_openai_error_is_recorded_and_reraised(self):
        with self.assertRaises(openai.BadRequestError):
            self.openai().chat.completions.create(model="bad", messages=[])
        call = self.calls()[0]
        self.assertIsNone(call["response"])
        self.assertTrue(call["error"].startswith("BadRequestError"))

    @unittest.skipUnless(HAS_OPENAI, "openai is not installed")
    def test_with_raw_response_records_the_parsed_reply_not_the_wrapper(self):
        # Regression: the APIResponse wrapper was recorded as its repr string.
        raw = self.openai().chat.completions.with_raw_response.create(model="gpt-test", messages=[])
        self.assertEqual(raw.status_code, 200)
        self.assertIs(raw.parse(), raw.parse())
        self.assertEqual(raw.parse().choices[0].message.content, "Unit 402 is open.")

        async def run():
            client = openai.AsyncOpenAI(api_key="sk-test", base_url=self.url + "/v1", max_retries=0)
            raw = await client.chat.completions.with_raw_response.create(model="gpt-test", messages=[])
            parsed = raw.parse()  # Sync on the legacy raw response, a coroutine on some versions.
            return await parsed if asyncio.iscoroutine(parsed) else parsed
        self.assertEqual(asyncio.run(run()).usage.total_tokens, 9)
        with self.openai().chat.completions.with_streaming_response.create(model="gpt-test", messages=[]) as streaming:
            self.assertEqual(streaming.parse().choices[0].message.content, "Unit 402 is open.")
        sync, asynchronous, streaming_call = self.calls()
        for call in (sync, asynchronous):
            self.assertEqual(call["response"]["choices"][0]["message"]["content"], "Unit 402 is open.")
            self.assertEqual(call["response"]["usage"]["total_tokens"], 9)
        self.assertIsNone(streaming_call["response"])  # The body is the caller's to read.

    @unittest.skipUnless(HAS_OPENAI and importlib.util.find_spec("openai.resources.beta.chat") is not None,
                         "needs an openai release with beta.chat.completions (1.40 to 1.x)")
    def test_openai_beta_chat_completions_parse_is_recorded_once(self):
        # Regression: openai 1.40 to 1.92 offer parse only as beta.chat.completions.parse, which posts
        # directly instead of calling create, so it was not recorded at all. Raw HTTP capture (on in
        # this test class) would hide that, so it is off here, as it is by default.
        http_off = mock.patch.object(generalcontext, "_http_start", lambda *args: None)
        http_off.start()
        self.addCleanup(http_off.stop)
        client = self.openai()
        reply = client.beta.chat.completions.parse(model="gpt-test", messages=[{"role": "user", "content": "p"}],
                                                   response_format={"type": "json_object"})

        async def run():
            other = openai.AsyncOpenAI(api_key="sk-test", base_url=self.url + "/v1", max_retries=0)
            return await other.beta.chat.completions.parse(model="gpt-test", messages=[], response_format={"type": "json_object"})
        asyncio.run(run())
        with client.beta.chat.completions.stream(model="gpt-test", messages=[]) as stream:
            list(stream)
        self.assertEqual(reply.choices[0].message.content, "Unit 402 is open.")
        calls = self.calls()
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]["request"]["messages"][0]["content"], "p")
        self.assertEqual(calls[0]["response"]["choices"][0]["message"]["content"], "Unit 402 is open.")

    @unittest.skipUnless(HAS_ANTHROPIC, "anthropic is not installed")
    def test_anthropic_plain_streamed_and_stream_helper(self):
        import anthropic
        client = anthropic.Anthropic(api_key="sk-test", base_url=self.url, max_retries=0)
        reply = client.messages.create(model="claude-test", max_tokens=10, messages=[{"role": "user", "content": "hi"}])
        events = list(client.messages.create(model="claude-test", max_tokens=10, messages=[], stream=True))
        with client.messages.stream(model="claude-test", max_tokens=10, messages=[]) as stream:
            text = "".join(stream.text_stream)
        self.assertEqual(reply.content[0].text, "Tours run daily.")
        self.assertEqual(len(events), len(MESSAGE_EVENTS))
        self.assertEqual(text, "Booking now.")
        plain, streamed, helper = self.calls()
        self.assertEqual(plain["api"], "anthropic")
        self.assertEqual(plain["response"]["content"][0]["text"], "Tours run daily.")
        for call in (streamed, helper):
            self.assertEqual(call["response"]["content"][0]["text"], "Booking now.")
            self.assertEqual(call["response"]["content"][1]["input"], {"day": "mon"})
            self.assertEqual(call["response"]["usage"]["output_tokens"], 9)
            self.assertEqual(call["response"]["model"], "claude-test")

    @unittest.skipUnless(HAS_ANTHROPIC, "anthropic is not installed")
    def test_async_anthropic(self):
        import anthropic

        async def run():
            client = anthropic.AsyncAnthropic(api_key="sk-test", base_url=self.url, max_retries=0)
            stream = await client.messages.create(model="claude-test", max_tokens=10, messages=[], stream=True)
            return [event async for event in stream]
        self.assertEqual(len(asyncio.run(run())), len(MESSAGE_EVENTS))
        self.assertEqual(self.calls()[0]["response"]["content"][0]["text"], "Booking now.")

    @unittest.skipUnless(HAS_HTTPX, "httpx is not installed")
    def test_raw_httpx_chat_completions_post_is_recorded(self):
        import httpx
        with generalcontext.task("raw"):
            response = httpx.post(self.url + "/v1/chat/completions", json={"model": "gpt-test", "messages": []})
            streamed = httpx.post(self.url + "/v1/chat/completions", json={"model": "gpt-test", "stream": True})
        self.assertEqual(response.json()["id"], "c1")
        self.assertIn("data: [DONE]", streamed.text)
        plain, sse = self.calls()
        self.assertEqual((plain["api"], plain["task"], plain["response"]["id"]), ("openai-chat", "raw", "c1"))
        self.assertEqual(sse["response"]["choices"][0]["message"]["content"], "Checking units.")

    # More OpenAI and Anthropic entry points.

    @unittest.skipUnless(HAS_OPENAI, "openai is not installed")
    def test_openai_parse_and_stream_helpers_are_recorded_once_each(self):
        client = self.openai()
        parsed = client.chat.completions.parse(model="gpt-test", messages=[{"role": "user", "content": "parse"}])
        with client.chat.completions.stream(model="gpt-test", messages=[{"role": "user", "content": "stream"}]) as stream:
            final = stream.get_final_completion()
        response = client.responses.parse(model="gpt-test", input="rparse")
        with client.responses.stream(model="gpt-test", input="rstream") as rstream:
            rfinal = rstream.get_final_response()
        self.assertEqual(parsed.choices[0].message.content, "Unit 402 is open.")
        self.assertEqual(final.choices[0].message.content, "Checking units.")
        self.assertEqual((response.output_text, rfinal.output_text), ("Hi there.", "Hi there."))
        calls = self.calls()
        self.assertEqual([c["api"] for c in calls], ["openai-chat", "openai-chat", "openai-responses", "openai-responses"])
        self.assertEqual(calls[0]["request"]["messages"][0]["content"], "parse")
        self.assertEqual(calls[1]["response"]["choices"][0]["message"]["content"], "Checking units.")
        self.assertEqual([c["request"]["input"] for c in calls[2:]], ["rparse", "rstream"])
        self.assertEqual(calls[3]["response"]["id"], "r1")
        self.assertEqual(len(Server.providers), 4)

    @unittest.skipUnless(HAS_OPENAI, "openai is not installed")
    def test_async_openai_parse_is_recorded(self):
        async def run():
            client = self.openai("AsyncOpenAI")
            return await client.chat.completions.parse(model="gpt-test", messages=[])
        self.assertEqual(asyncio.run(run()).model, "gpt-test")
        self.assertEqual([c["api"] for c in self.calls()], ["openai-chat"])

    @unittest.skipUnless(HAS_OPENAI, "openai is not installed")
    def test_azure_openai_client_is_recorded(self):
        client = openai.AzureOpenAI(api_key="sk-test", api_version="2024-10-21", azure_endpoint=self.url, max_retries=0)
        reply = client.chat.completions.create(model="my-deployment", messages=[{"role": "user", "content": "azure"}])
        self.assertEqual(reply.choices[0].message.content, "Unit 402 is open.")
        self.assertIn("/openai/deployments/my-deployment/", Server.providers[0])
        (call,) = self.calls()
        self.assertEqual((call["api"], call["request"]["messages"][0]["content"]), ("openai-chat", "azure"))

    @unittest.skipUnless(HAS_ANTHROPIC, "anthropic is not installed")
    def test_anthropic_parse_and_beta_create_parse_stream(self):
        import anthropic
        client = anthropic.Anthropic(api_key="sk-test", base_url=self.url, max_retries=0)
        kwargs = dict(model="claude-test", max_tokens=10)
        parsed = client.messages.parse(messages=[{"role": "user", "content": "parse"}], **kwargs)
        beta = client.beta.messages.create(messages=[{"role": "user", "content": "beta"}], **kwargs)
        beta_parsed = client.beta.messages.parse(messages=[{"role": "user", "content": "bparse"}], **kwargs)
        with client.beta.messages.stream(messages=[{"role": "user", "content": "bstream"}], **kwargs) as stream:
            text = "".join(stream.text_stream)
        self.assertEqual((parsed.content[0].text, beta.content[0].text, beta_parsed.content[0].text),
                         ("Tours run daily.",) * 3)
        self.assertEqual(text, "Booking now.")
        calls = self.calls()
        self.assertEqual([c["api"] for c in calls], ["anthropic"] * 4)
        self.assertEqual([c["request"]["messages"][0]["content"] for c in calls], ["parse", "beta", "bparse", "bstream"])
        self.assertEqual(calls[3]["response"]["content"][0]["text"], "Booking now.")
        self.assertEqual(calls[3]["response"]["content"][1]["input"], {"day": "mon"})
        self.assertEqual(len(Server.providers), 4)

    @unittest.skipUnless(HAS_ANTHROPIC, "anthropic is not installed")
    def test_async_anthropic_parse_and_beta_stream(self):
        import anthropic

        async def run():
            client = anthropic.AsyncAnthropic(api_key="sk-test", base_url=self.url, max_retries=0)
            parsed = await client.messages.parse(model="claude-test", max_tokens=10, messages=[])
            beta = await client.beta.messages.parse(model="claude-test", max_tokens=10, messages=[])
            async with client.beta.messages.stream(model="claude-test", max_tokens=10, messages=[]) as stream:
                text = "".join([t async for t in stream.text_stream])
            return parsed, beta, text
        parsed, beta, text = asyncio.run(run())
        self.assertEqual((parsed.content[0].text, beta.content[0].text, text), ("Tours run daily.", "Tours run daily.", "Booking now."))
        calls = self.calls()
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[2]["response"]["content"][0]["text"], "Booking now.")

    @unittest.skipUnless(HAS_ANTHROPIC, "anthropic is not installed")
    def test_anthropic_bedrock_and_vertex_use_the_patched_classes(self):
        import anthropic
        from anthropic.resources.messages import Messages
        from anthropic.resources.beta.messages.messages import Messages as BetaMessages
        clients = [anthropic.AnthropicVertex(region="us-east5", project_id="p", access_token="t", base_url=self.url)]
        if HAS_BOTO3:
            clients.append(anthropic.AnthropicBedrock(aws_access_key="a", aws_secret_key="s", aws_region="us-east-1",
                                                      base_url=self.url))
        for client in clients:
            self.assertIsInstance(client.messages, Messages)
            self.assertTrue(getattr(client.beta.messages.create, "_generalcontext", False))
        for cls in (Messages, BetaMessages):
            for method in ("create", "parse", "stream"):
                self.assertTrue(getattr(getattr(cls, method), "_generalcontext", False), (cls, method))

    # Google Gen AI.

    def gemini(self):
        from google import genai
        return genai.Client(api_key="test-key", http_options={"base_url": self.url, "api_version": "v1beta"})

    @unittest.skipUnless(HAS_GENAI, "google-genai is not installed")
    def test_gemini_generate_content_plain_and_streamed(self):
        from google.genai import types
        client = self.gemini()
        config = types.GenerateContentConfig(system_instruction="Be brief.", temperature=0)
        with generalcontext.session("gem-1"):
            reply = client.models.generate_content(model="gemini-test", contents="hi", config=config)
            chunks = list(client.models.generate_content_stream(model="gemini-test", contents=["a", "b"]))
        self.assertEqual(reply.text, "Tours run daily.")
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].text, "Unit 402 ")
        plain, streamed = self.calls()
        self.assertEqual((plain["api"], plain["conversation"]["id"]), ("gemini", "gem-1"))
        self.assertEqual(plain["request"]["model"], "gemini-test")
        self.assertEqual(plain["request"]["contents"], "hi")
        self.assertEqual(plain["request"]["config"]["system_instruction"], "Be brief.")
        self.assertEqual(plain["response"]["candidates"][0]["content"]["parts"], [{"text": "Tours run daily."}])
        self.assertEqual(plain["response"]["usage_metadata"]["prompt_token_count"], 3)
        response = streamed["response"]
        self.assertEqual(response["candidates"][0]["content"]["parts"],
                         [{"text": "Unit 402 is open."}, {"function_call": {"name": "book", "args": {"day": "mon"}}}])
        self.assertEqual(response["candidates"][0]["finish_reason"], "STOP")
        self.assertEqual(response["usage_metadata"]["candidates_token_count"], 5)
        self.assertEqual(response["model_version"], "gemini-test")
        self.assertEqual(len(Server.providers), 2)

    @unittest.skipUnless(HAS_GENAI, "google-genai is not installed")
    def test_async_gemini_plain_and_streamed(self):
        async def run():
            client = self.gemini()
            reply = await client.aio.models.generate_content(model="gemini-test", contents="hi")
            stream = await client.aio.models.generate_content_stream(model="gemini-test", contents="hi")
            return reply, [chunk async for chunk in stream]
        reply, chunks = asyncio.run(run())
        self.assertEqual((reply.text, len(chunks)), ("Tours run daily.", 3))
        plain, streamed = self.calls()
        self.assertEqual(streamed["response"]["candidates"][0]["content"]["parts"][0], {"text": "Unit 402 is open."})

    @unittest.skipUnless(HAS_GENAI, "google-genai is not installed")
    def test_gemini_error_is_recorded_and_reraised(self):
        from google.genai import errors
        with self.assertRaises(errors.APIError):
            client = self.gemini()
            client.models.generate_content(model="missing", contents="hi")
        (call,) = self.calls()
        self.assertEqual(call["api"], "gemini")
        self.assertIsNotNone(call["error"])

    # AWS Bedrock Converse.

    def bedrock(self):
        import boto3
        return boto3.client("bedrock-runtime", region_name="us-east-1", aws_access_key_id="test",
                            aws_secret_access_key="test")

    @unittest.skipUnless(HAS_BOTO3, "boto3 is not installed")
    def test_bedrock_converse_is_recorded_with_bytes_omitted(self):
        from botocore.stub import Stubber
        client = self.bedrock()
        reply = {"output": {"message": {"role": "assistant", "content": [{"text": "Tours run daily."}]}},
                 "stopReason": "end_turn", "usage": {"inputTokens": 3, "outputTokens": 4, "totalTokens": 7},
                 "metrics": {"latencyMs": 5}}
        messages = [{"role": "user", "content": [{"text": "hi"}, {"image": {"format": "png", "source": {"bytes": b"\x89PNG"}}}]}]
        with Stubber(client) as stub, generalcontext.task("bedrock"):
            stub.add_response("converse", reply, {"modelId": "anthropic.claude-test", "messages": messages})
            result = client.converse(modelId="anthropic.claude-test", messages=messages)
        self.assertEqual(result["output"], reply["output"])
        (call,) = self.calls()
        self.assertEqual((call["api"], call["task"]), ("bedrock-converse", "bedrock"))
        self.assertEqual(call["request"]["modelId"], "anthropic.claude-test")
        self.assertEqual(call["request"]["messages"][0]["content"][1]["image"]["source"]["bytes"], {"omitted": "bytes"})
        self.assertEqual(call["response"]["output"]["message"]["content"], [{"text": "Tours run daily."}])
        self.assertEqual(call["response"]["usage"]["outputTokens"], 4)
        self.assertNotIn("ResponseMetadata", call["response"])

    @unittest.skipUnless(HAS_BOTO3, "boto3 is not installed")
    def test_bedrock_converse_stream_passes_events_and_records_one_response(self):
        from botocore.stub import Stubber
        client = self.bedrock()
        events = [
            {"messageStart": {"role": "assistant"}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Booking "}}},
            {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "now."}}},
            {"contentBlockStop": {"contentBlockIndex": 0}},
            {"contentBlockStart": {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "t1", "name": "book"}}}},
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": "{\"day\": "}}}},
            {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {"toolUse": {"input": "\"mon\"}"}}}},
            {"messageStop": {"stopReason": "tool_use"}},
            {"metadata": {"usage": {"inputTokens": 3, "outputTokens": 9, "totalTokens": 12}, "metrics": {"latencyMs": 5}}},
        ]
        with Stubber(client) as stub:
            stub._validate_operation_response = lambda *args: None  # The stubbed stream is a plain list of events.
            stub.add_response("converse_stream", {"stream": events})
            result = client.converse_stream(modelId="m", messages=[{"role": "user", "content": [{"text": "book"}]}])
            seen = list(result["stream"])
        self.assertEqual(seen, events)
        (call,) = self.calls()
        self.assertEqual(call["api"], "bedrock-converse")
        self.assertEqual(call["response"], {
            "output": {"message": {"role": "assistant", "content": [
                {"text": "Booking now."}, {"toolUse": {"toolUseId": "t1", "name": "book", "input": {"day": "mon"}}}]}},
            "stopReason": "tool_use", "usage": {"inputTokens": 3, "outputTokens": 9, "totalTokens": 12},
            "metrics": {"latencyMs": 5}})

    @unittest.skipUnless(HAS_BOTO3, "boto3 is not installed")
    def test_other_boto3_operations_are_not_recorded_and_errors_are(self):
        from botocore.stub import Stubber
        client = self.bedrock()
        with Stubber(client) as stub:
            stub.add_client_error("converse", "ValidationException", "bad model")
            with self.assertRaises(client.exceptions.ValidationException):
                client.converse(modelId="bad", messages=[])
            stub._validate_operation_response = lambda *args: None
            stub.add_response("apply_guardrail", {"action": "NONE", "outputs": [], "assessments": []})
            client.apply_guardrail(guardrailIdentifier="g", guardrailVersion="1", source="INPUT", content=[])
        (call,) = self.calls()
        self.assertTrue(call["error"].startswith("ClientError") or "ValidationException" in call["error"])



if __name__ == "__main__":
    unittest.main()


class KillSwitchWrapTest(unittest.TestCase):
    def test_wrap_does_nothing_when_disabled(self):
        os.environ["GENERALCONTEXT_DISABLED"] = "true"
        try:
            client = Client({"choices": []})
            self.assertIs(generalcontext.wrap(client), client)
            self.assertNotIn("create", vars(client.chat.completions), "wrap must not patch the client")
        finally:
            del os.environ["GENERALCONTEXT_DISABLED"]
