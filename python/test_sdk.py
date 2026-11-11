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
        generalcontext._config["base_url"] = cls.url  # another test class may have initialized first

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
        # The shared contract fixture has the shape this SDK sends, field for field.
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "contract", "calls.json")) as handle:
            fixture = json.load(handle)
        self.assertEqual(set(fixture), set(body))
        for example in fixture["calls"]:
            self.assertEqual(set(example), set(call))
            self.assertTrue(example["conversation"] is None or set(example["conversation"]) <= {"id", "user_id", "channel", "product", "metadata"})

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

    def test_credentials_are_never_sent_even_with_redaction_off(self):
        # A key pasted into a prompt, returned by a tool, or passed as a per-request option. The default
        # client here has no redaction, so nothing but the always-on masking stands between them and us.
        openai_key, aws_key, gc_key = "sk-proj-" + "a1B2" * 10, "AKIAIOSFODNN7EXAMPLE", "gck_" + "0f" * 24
        client = generalcontext.wrap(Client({"choices": [{"message": {"role": "assistant", "content": "Your key " + gc_key + " is set."}}]}))
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "here is my key " + openai_key},
                                 {"role": "tool", "tool_call_id": "t", "content": '{"AccessKeyId": "' + aws_key + '"}'}],
            extra_headers={"Helicone-Auth": "Bearer sk-helicone-" + "z" * 30}, extra_query={"api-key": "azure-" + "q" * 32})
        sent = json.dumps(self.calls()[0])
        for secret in (openai_key, aws_key, gc_key, "sk-helicone", "azure-qqq"):
            self.assertNotIn(secret, sent)
        self.assertIn("<openai-key>", sent)
        self.assertIn("<aws-key>", sent)
        self.assertIn("<generalcontext-key>", sent)
        self.assertNotIn("extra_query", sent)
        self.assertTrue(generalcontext.report_outcome("s1", "fail", "customer read out " + openai_key))
        self.assertNotIn(openai_key, json.dumps(Server.received[-1][2]))

    def test_secret_masking_is_fast_on_adversarial_input(self):
        for text in ("-----BEGIN RSA PRIVATE KEY-----\n" * 3000, "sk-" + "a" * 50_000, "Bearer " * 10_000, ("eyJ" + "a" * 10 + ".") * 5000):
            started = time.perf_counter()
            generalcontext.mask_secrets(text)
            self.assertLess(time.perf_counter() - started, 0.5)

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


# ---- OpenAI Realtime (voice) --------------------------------------------------------------------
# A fake Realtime server (websockets, plain ws://) drives a real client.realtime.connect() from the installed openai.

def _realtime_support():
    if not HAS_OPENAI:
        return "openai is not installed"
    if importlib.util.find_spec("websockets") is None:
        return "websockets is not installed (openai[realtime])"
    try:
        from openai.resources.realtime.realtime import RealtimeConnection  # noqa: F401
    except Exception:
        return "this openai has no client.realtime.connect()"
    return None


REALTIME_SKIP = _realtime_support()
AUDIO = __import__("base64").b64encode(b"\x07" * 6000).decode()  # 8000 base64 characters, never recorded


def _two_turn_server(sent, received):
    tools = [{"type": "function", "name": "get_rent", "description": "Rent for a unit",
              "parameters": {"type": "object", "properties": {"unit": {"type": "string"}}}}]
    usage = lambda i, o: {"total_tokens": i + o, "input_tokens": i, "output_tokens": o}  # noqa: E731
    fc = {"id": "item_fc", "type": "function_call", "call_id": "call_rent", "name": "get_rent",
          "arguments": "{\"unit\":\"4B\"}", "status": "completed"}
    preamble = {"id": "item_a1", "type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_audio", "transcript": "Let me check."}]}
    answer = {"id": "item_a2", "type": "message", "role": "assistant", "status": "incomplete",
              "content": [{"type": "output_audio", "transcript": "Unit 4B rents for 2100 dollars a month."}]}

    def handler(ws):
        def send(event):
            sent.append(event["type"])
            ws.send(json.dumps(event))
        send({"type": "session.created", "event_id": "e0", "session": {"type": "realtime", "model": "gpt-realtime", "instructions": "", "tools": []}})
        appended = False
        for message in ws:
            e = json.loads(message)
            received.append(e)
            kind = e["type"]
            if kind == "session.update":
                send({"type": "session.updated", "event_id": "e1", "session": dict({"type": "realtime", "model": "gpt-realtime"}, **e["session"])})
            elif kind == "input_audio_buffer.append" and not appended:
                appended = True
                for event in [
                    {"type": "input_audio_buffer.speech_started", "audio_start_ms": 0, "item_id": "item_u1"},
                    {"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 1800, "item_id": "item_u1"},
                    {"type": "input_audio_buffer.committed", "item_id": "item_u1", "previous_item_id": None},
                    {"type": "conversation.item.added", "previous_item_id": None, "item": {"id": "item_u1", "type": "message", "role": "user", "content": [{"type": "input_audio", "transcript": None}]}},
                    {"type": "conversation.item.input_audio_transcription.completed", "item_id": "item_u1", "content_index": 0, "transcript": "What is the rent for 4B?", "usage": {"type": "duration", "seconds": 2}},
                    {"type": "response.created", "response": {"id": "resp_1", "object": "realtime.response", "status": "in_progress", "output": []}},
                    {"type": "response.output_item.added", "response_id": "resp_1", "output_index": 0, "item": dict(preamble, status="in_progress", content=[])},
                    {"type": "response.output_audio.delta", "response_id": "resp_1", "item_id": "item_a1", "output_index": 0, "content_index": 0, "delta": AUDIO},
                    {"type": "response.output_audio_transcript.done", "response_id": "resp_1", "item_id": "item_a1", "output_index": 0, "content_index": 0, "transcript": "Let me check."},
                    {"type": "response.output_item.added", "response_id": "resp_1", "output_index": 1, "item": dict(fc, arguments="", status="in_progress")},
                    {"type": "response.function_call_arguments.done", "response_id": "resp_1", "item_id": "item_fc", "output_index": 1, "call_id": "call_rent", "name": "get_rent", "arguments": fc["arguments"]},
                    {"type": "response.output_item.done", "response_id": "resp_1", "output_index": 1, "item": fc},
                    {"type": "response.done", "response": {"id": "resp_1", "object": "realtime.response", "status": "completed", "output": [preamble, fc], "usage": usage(120, 15)}},
                ]:
                    send(dict(event, event_id="e"))
            elif kind == "conversation.item.create":
                send({"type": "conversation.item.added", "event_id": "e12", "previous_item_id": "item_fc", "item": dict(e["item"], id="item_fco")})
            elif kind == "response.create":
                send({"type": "response.created", "event_id": "e13", "response": {"id": "resp_2", "object": "realtime.response", "status": "in_progress", "output": []}})
                send({"type": "response.output_item.added", "event_id": "e14", "response_id": "resp_2", "output_index": 0, "item": dict(answer, status="in_progress", content=[])})
                for i in range(3):
                    send({"type": "response.output_audio.delta", "event_id": f"e15{i}", "response_id": "resp_2", "item_id": "item_a2", "output_index": 0, "content_index": 0, "delta": AUDIO})
                send({"type": "response.output_audio_transcript.done", "event_id": "e16", "response_id": "resp_2", "item_id": "item_a2", "output_index": 0, "content_index": 0, "transcript": answer["content"][0]["transcript"]})
                send({"type": "input_audio_buffer.speech_started", "event_id": "e17", "audio_start_ms": 2600, "item_id": "item_u2"})
            elif kind == "conversation.item.truncate":
                send({"type": "conversation.item.truncated", "event_id": "e18", "item_id": e["item_id"], "content_index": 0, "audio_end_ms": e["audio_end_ms"]})
            elif kind == "response.cancel":
                send({"type": "response.output_item.done", "event_id": "e19", "response_id": "resp_2", "output_index": 0, "item": answer})
                send({"type": "response.done", "event_id": "e20", "response": {"id": "resp_2", "object": "realtime.response", "status": "cancelled", "output": [answer], "usage": usage(160, 30)}})
    return handler, tools


def _voice_contract():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "contract", "voice.json")) as handle:
        return [c for c in json.load(handle)["calls"] if c["api"] == "openai-realtime"]


def _item_kind(item):
    return item.get("type") or item.get("role")


def _item_shapes(calls):
    """Item kind to its keys, and to its voice keys without the optional truncated_at_ms."""
    shapes = {}
    for call in calls:
        for item in call["request"]["input"] + call["response"]["output"]:
            voice = sorted(set(item["voice"]) - {"truncated_at_ms"}) if "voice" in item else None
            shapes.setdefault(_item_kind(item), set()).add((tuple(sorted(item)), tuple(voice) if voice else None))
    return shapes


@unittest.skipIf(REALTIME_SKIP, REALTIME_SKIP or "")
class RealtimeTest(unittest.TestCase):
    # The collector and setup from SdkTest, without inheriting its tests.
    setUpClass = classmethod(SdkTest.setUpClass.__func__)
    tearDownClass = classmethod(SdkTest.tearDownClass.__func__)
    setUp = SdkTest.setUp
    calls = SdkTest.calls

    def serve(self, handler):
        from websockets.sync.server import serve
        server = serve(handler, "127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.shutdown)
        return openai.OpenAI(api_key="sk-test", base_url=f"http://127.0.0.1:{server.socket.getsockname()[1]}/v1", max_retries=0)

    def realtime_calls(self):
        return [c for c in self.calls() if c["api"] == "openai-realtime"]

    def test_two_turn_voice_call_records_one_responses_shaped_call_per_response(self):
        sent, received = [], []
        handler, tools = _two_turn_server(sent, received)
        client = self.serve(handler)
        seen = []
        with generalcontext.session("call-1"):
            connection = client.realtime.connect(model="gpt-realtime").enter()
        with generalcontext.session("other-scope"):  # the session at connect time is the one that counts
            connection.session.update(session={"type": "realtime", "instructions": "You are a leasing agent.", "tools": tools,
                                               "audio": {"output": {"format": {"type": "audio/pcm", "rate": 24000}}}})
            connection.input_audio_buffer.append(audio=AUDIO)
            playing, done = None, 0
            for event in connection:
                seen.append(event.type)
                if event.type == "response.function_call_arguments.done":
                    connection.conversation.item.create(item={"type": "function_call_output", "call_id": event.call_id, "output": "{\"rent\":2100}"})
                    connection.response.create()
                elif event.type == "response.output_audio.delta":
                    playing = playing or event.item_id
                elif event.type == "input_audio_buffer.speech_started" and playing:
                    connection.conversation.item.truncate(item_id=playing, content_index=0, audio_end_ms=640)
                    connection.response.cancel()
                    playing = None
                elif event.type == "response.done":
                    done += 1
                    if done == 2:
                        break
            connection.close()
        calls = self.realtime_calls()

        self.assertEqual(seen, sent)  # the app saw every server event, in order
        self.assertEqual(next(e for e in received if e["type"] == "input_audio_buffer.append")["audio"], AUDIO)
        self.assertEqual([e["type"] for e in received], ["session.update", "input_audio_buffer.append", "conversation.item.create",
                                                         "response.create", "conversation.item.truncate", "response.cancel"])
        self.assertEqual(len(calls), 2)
        for c in calls:
            self.assertEqual(c["conversation"], {"id": "call-1", "channel": "voice"})
            self.assertEqual(c["request"]["model"], "gpt-realtime")
            self.assertEqual(c["request"]["instructions"], "You are a leasing agent.")
            self.assertEqual(c["request"]["tools"][0]["name"], "get_rent")
            self.assertTrue({"speech_stopped_at", "first_audio_at", "interrupted"} <= set(c["voice"]))
        first, second = calls
        user = {"role": "user", "content": [{"type": "input_text", "text": "What is the rent for 4B?"}], "audio": True,
                "at": first["request"]["input"][0].get("at")}
        self.assertIsNotNone(user["at"])
        latency = generalcontext._rt_latency(first["voice"])
        self.assertIsInstance(latency, int)
        said = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Let me check."}], "audio": True,
                "voice": {"latency_ms": latency, "interrupted": False}}
        call = {"type": "function_call", "call_id": "call_rent", "name": "get_rent", "arguments": "{\"unit\":\"4B\"}"}
        result = {"type": "function_call_output", "call_id": "call_rent", "output": "{\"rent\":2100}"}
        reply = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Unit 4B rents for 2100 dollars a month."}], "audio": True,
                 "voice": {"latency_ms": None, "interrupted": True, "truncated_at_ms": 640}}
        self.assertEqual(first["request"]["input"], [user])
        self.assertEqual(first["response"], {"id": "resp_1", "output": [said, call], "usage": {"total_tokens": 135, "input_tokens": 120, "output_tokens": 15}})
        self.assertIsNone(first["error"])
        self.assertIsNotNone(first["voice"]["speech_stopped_at"])
        self.assertIsNotNone(first["voice"]["first_audio_at"])
        self.assertFalse(first["voice"]["interrupted"])
        # The first reply's timing is still on its item in the second call's history.
        self.assertEqual(second["request"]["input"], [user, said, call, result])
        self.assertEqual(second["request"]["input"][1]["voice"], {"latency_ms": latency, "interrupted": False})
        self.assertEqual(second["response"]["output"], [reply])
        self.assertIsNotNone(second["voice"]["first_audio_at"])
        self.assertTrue(second["voice"]["interrupted"])
        self.assertEqual(second["voice"]["truncated_at_ms"], 640)
        recorded = json.dumps(calls)
        # The shared voice fixture has the shape this SDK sends, key for key. Values may differ.
        fixture = _voice_contract()
        for c in calls:
            self.assertEqual(set(c), set(fixture[0]))
            self.assertEqual(set(c["request"]), set(fixture[0]["request"]))
            self.assertEqual(set(c["response"]), set(fixture[0]["response"]))
        expected, actual = _item_shapes(fixture), _item_shapes(calls)
        self.assertEqual(set(actual), {"user", "message", "function_call", "function_call_output"})
        for kind, shapes in actual.items():
            self.assertLessEqual(shapes, expected[kind], kind)
        self.assertIsNone(__import__("re").search(r"[A-Za-z0-9+/]{100,}", recorded))
        self.assertNotIn("omitted", recorded)  # dropped at the source, not even a placeholder

    def test_error_event_and_socket_closed_mid_response_record_an_error(self):
        def handler(ws):
            ws.send(json.dumps({"type": "session.created", "event_id": "e0", "session": {"model": "gpt-realtime", "instructions": "Be brief.", "tools": []}}))
            for message in ws:
                e = json.loads(message)
                if e["type"] == "conversation.item.create":
                    ws.send(json.dumps({"type": "conversation.item.added", "event_id": "e1", "item": dict(e["item"], id="item_u")}))
                if e["type"] == "response.create":
                    ws.send(json.dumps({"type": "response.created", "event_id": "e2", "response": {"id": "resp_x", "status": "in_progress", "output": []}}))
                    ws.send(json.dumps({"type": "response.output_audio.delta", "event_id": "e3", "response_id": "resp_x", "item_id": "i", "output_index": 0, "content_index": 0, "delta": AUDIO}))
                    ws.send(json.dumps({"type": "error", "event_id": "e4", "error": {"type": "server_error", "code": "overloaded", "message": "The server is overloaded"}}))
                    ws.close()
                    return
        client = self.serve(handler)
        with generalcontext.session("call-2"):
            with client.realtime.connect(model="gpt-realtime") as connection:
                connection.conversation.item.create(item={"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]})
                connection.response.create()
                kinds = [event.type for event in connection]  # ends when the server closes
        self.assertEqual(kinds[-1], "error")
        calls = self.realtime_calls()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["conversation"], {"id": "call-2", "channel": "voice"})
        self.assertEqual(calls[0]["request"]["input"], [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}])
        self.assertRegex(calls[0]["error"], "overloaded")
        self.assertEqual(calls[0]["request"]["instructions"], "Be brief.")

    def test_the_wait_before_a_reply_counts_from_the_caller_through_a_silent_tool_call(self):
        # The caller stops speaking, the model calls a tool (no audio), then speaks after the result. The
        # caller waited the whole time, so the spoken reply's latency starts at speech_stopped.
        def handler(ws):
            for message in ws:
                e = json.loads(message)
                if e["type"] == "response.create":
                    send = lambda event: ws.send(json.dumps(event))
                    send({"type": "input_audio_buffer.speech_stopped", "event_id": "e0", "audio_end_ms": 900, "item_id": "u1"})
                    send({"type": "conversation.item.input_audio_transcription.completed", "event_id": "e1", "item_id": "u1", "content_index": 0, "transcript": "Any two bedrooms?"})
                    send({"type": "response.created", "event_id": "e2", "response": {"id": "r1", "status": "in_progress", "output": []}})
                    send({"type": "response.output_item.done", "event_id": "e3", "response_id": "r1", "output_index": 0, "item": {"id": "fc", "type": "function_call", "call_id": "c1", "name": "units", "arguments": "{}"}})
                    send({"type": "response.done", "event_id": "e4", "response": {"id": "r1", "status": "completed", "output": [], "usage": None}})
                    time.sleep(0.12)
                    send({"type": "response.created", "event_id": "e5", "response": {"id": "r2", "status": "in_progress", "output": []}})
                    send({"type": "response.output_audio.delta", "event_id": "e6", "response_id": "r2", "item_id": "a2", "output_index": 0, "content_index": 0, "delta": "AAAA"})
                    send({"type": "response.output_audio_transcript.done", "event_id": "e7", "response_id": "r2", "item_id": "a2", "output_index": 0, "content_index": 0, "transcript": "Yes, unit 204."})
                    send({"type": "response.done", "event_id": "e8", "response": {"id": "r2", "status": "completed", "output": [], "usage": None}})
                    ws.close()
                    return
        client = self.serve(handler)
        with generalcontext.session("call-4"):
            with client.realtime.connect(model="gpt-realtime") as connection:
                connection.response.create()
                for _ in connection:
                    pass
        tool_call, spoken = self.realtime_calls()
        self.assertIsNone(tool_call["voice"]["first_audio_at"])
        self.assertRegex(spoken["voice"]["speech_stopped_at"], r"^\d{4}-")
        reply = next(item for item in spoken["response"]["output"] if item.get("role") == "assistant")
        self.assertGreaterEqual(reply["voice"]["latency_ms"], 100)

    def test_a_caller_turn_with_no_transcript_still_shows_and_a_late_transcript_fills_it_in(self):
        def handler(ws):
            send = lambda event: ws.send(json.dumps(event))
            spoke = lambda i: send({"type": "conversation.item.added", "event_id": "x" + i, "previous_item_id": None, "item": {"id": i, "type": "message", "role": "user", "content": [{"type": "input_audio", "transcript": None}]}})
            def reply(r, text):
                send({"type": "response.created", "event_id": "c" + r, "response": {"id": r, "status": "in_progress", "output": []}})
                send({"type": "response.output_text.done", "event_id": "t" + r, "response_id": r, "item_id": "a" + r, "output_index": 0, "content_index": 0, "text": text})
                send({"type": "response.done", "event_id": "d" + r, "response": {"id": r, "status": "completed", "output": [], "usage": None}})
            for message in ws:
                if json.loads(message)["type"] == "response.create":
                    spoke("u1")
                    reply("r1", "Hello.")
                    spoke("u2")
                    send({"type": "conversation.item.input_audio_transcription.completed", "event_id": "tr", "item_id": "u2", "content_index": 0, "transcript": "Book me in."})
                    spoke("u2")
                    reply("r2", "Done.")
                    ws.close()
                    return
        client = self.serve(handler)
        with generalcontext.session("call-5"):
            with client.realtime.connect(model="gpt-realtime") as connection:
                connection.response.create()
                for _ in connection:
                    pass
        calls = self.realtime_calls()
        users = [item for item in calls[1]["request"]["input"] if item.get("role") == "user"]
        self.assertEqual([(item["content"][0]["text"], item.get("audio")) for item in users], [("", True), ("Book me in.", True)])

    def test_async_connection_and_bounded_state(self):
        def handler(ws):
            count = 0
            for message in ws:
                count += 1
                if count == 250:
                    ws.send(json.dumps({"type": "response.created", "response": {"id": "r"}}))
                    ws.send(json.dumps({"type": "response.output_text.done", "response_id": "r", "item_id": "a", "text": "ok"}))
                    ws.send(json.dumps({"type": "response.done", "response": {"id": "r", "status": "completed", "output": [], "usage": None}}))

        sync_client = self.serve(handler)
        client = openai.AsyncOpenAI(api_key="sk-test", base_url=str(sync_client.base_url), max_retries=0)

        async def run():
            with generalcontext.session("call-3"):
                async with client.realtime.connect(model="gpt-realtime") as connection:
                    for i in range(250):
                        await connection.conversation.item.create(item={"id": f"u{i}", "type": "message", "role": "user", "content": [{"type": "input_text", "text": f"t{i}"}]})
                    async for event in connection:
                        if event.type == "response.done":
                            break
        asyncio.run(run())
        calls = self.realtime_calls()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["conversation"], {"id": "call-3", "channel": "voice"})
        self.assertEqual(len(calls[0]["request"]["input"]), 199)  # 200 kept, one of them the reply
        self.assertEqual(calls[0]["request"]["input"][0]["content"][0]["text"], "t51")
        self.assertEqual(calls[0]["response"]["output"], [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}], "voice": {"latency_ms": None, "interrupted": False}}])
        if os.environ.get("GC_PRINT_REALTIME"):
            print(json.dumps(calls[0], indent=2))
