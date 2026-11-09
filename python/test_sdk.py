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
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import generalcontext

HAS_OPENAI = importlib.util.find_spec("openai") is not None
HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None
HAS_HTTPX = importlib.util.find_spec("httpx") is not None
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


class Server(BaseHTTPRequestHandler):
    received = []  # (path, authorization, body) for collector requests
    status = 200

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if self.path.startswith("/api/platform-collect"):
            Server.received.append((self.path, self.headers.get("Authorization"), body))
            return self.reply(Server.status, {})
        if body.get("model") == "bad":
            return self.reply(400, {"error": {"message": "bad model", "type": "invalid_request_error"}})
        if self.path.endswith("/chat/completions"):
            return self.sse([("", c) for c in CHAT_CHUNKS], done=True) if body.get("stream") else self.reply(200, CHAT)
        if self.path.endswith("/messages"):
            return self.sse([(e["type"], e) for e in MESSAGE_EVENTS]) if body.get("stream") else self.reply(200, MESSAGE)
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
        generalcontext.initialize(api_key=KEY, base_url=cls.url, app_name="leasing-bot")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()

    def setUp(self):
        Server.received.clear()
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
        with generalcontext.session("sms:thread-1", user_id="u1", channel="sms", product="leasing"):
            with generalcontext.task("qualify"):
                result = client.chat.completions.create(model="m", messages=[{"role": "user", "content": "2 beds?"}])
        self.assertIs(result, reply)
        self.assertTrue(generalcontext.flush(5))
        path, authorization, body = Server.received[0]
        self.assertEqual((path, authorization), ("/api/platform-collect/sdk", "Bearer " + KEY))
        self.assertEqual((body["sdk"], body["app"]), ("python/0.2.0", "leasing-bot"))
        call = body["calls"][0]
        self.assertEqual(call["api"], "openai-chat")
        self.assertEqual(call["conversation"], {"id": "sms:thread-1", "user_id": "u1", "channel": "sms", "product": "leasing"})
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


if __name__ == "__main__":
    unittest.main()
