"""A local fake model provider and General Context collector on one port. Never reaches the network.

Every provider route answers with the same reply: text "Unit 4B is open.", one tool call to
`get_unit`, and usage of 4200 input / 42 output tokens, so a recorded call can be checked for
completeness without knowing the provider's shape. A model name (in the body or the URL path)
of err-400, err-429, err-500 or err-timeout makes the provider fail that way instead.
"""

import base64
import json
import struct
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TEXT = "Unit 4B is open."
JSON_TEXT = '{"unit": "4B", "open": true}'
TIMEOUT_SLEEP = 2.5  # Clients use a 1 second timeout.


def chat(structured):
    message = {"role": "assistant", "content": JSON_TEXT if structured else TEXT}
    if not structured:
        message["tool_calls"] = [{"id": "call_1", "type": "function",
                                  "function": {"name": "get_unit", "arguments": '{"unit": "4B"}'}}]
    return {"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "gpt-fake",
            "choices": [{"index": 0, "finish_reason": "stop" if structured else "tool_calls", "message": message,
                         "logprobs": None}],
            "usage": {"prompt_tokens": 4200, "completion_tokens": 42, "total_tokens": 4242}}


def chat_chunks(structured):
    base = {"id": "chatcmpl-2", "object": "chat.completion.chunk", "created": 1, "model": "gpt-fake"}
    texts = [JSON_TEXT[:10], JSON_TEXT[10:]] if structured else ["Unit 4B ", "is open."]
    chunks = [{"choices": [{"index": 0, "delta": {"role": "assistant", "content": texts[0]}, "finish_reason": None}]},
              {"choices": [{"index": 0, "delta": {"content": texts[1]}, "finish_reason": None}]}]
    if not structured:
        chunks += [
            {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "call_1", "type": "function",
                                                                 "function": {"name": "get_unit", "arguments": '{"unit"'}}]},
                          "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ': "4B"}'}}]},
                          "finish_reason": None}]}]
    chunks += [{"choices": [{"index": 0, "delta": {}, "finish_reason": "stop" if structured else "tool_calls"}]},
               {"choices": [], "usage": {"prompt_tokens": 4200, "completion_tokens": 42, "total_tokens": 4242}}]
    return [dict(base, **c) for c in chunks]


def response(structured):
    message = {"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": JSON_TEXT if structured else TEXT, "annotations": []}]}
    output = [message]
    if not structured:
        output.append({"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "get_unit",
                       "arguments": '{"unit": "4B"}', "status": "completed"})
    return {"id": "resp_1", "object": "response", "created_at": 1, "model": "gpt-fake", "status": "completed",
            "output": output, "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "usage": {"input_tokens": 4200, "output_tokens": 42, "total_tokens": 4242,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}


def response_events(structured):
    final = response(structured)
    text = final["output"][0]["content"][0]["text"]
    message = final["output"][0]
    events = [
        {"type": "response.created", "response": dict(final, status="in_progress", output=[], usage=None)},
        {"type": "response.output_item.added", "output_index": 0,
         "item": dict(message, status="in_progress", content=[])},
        {"type": "response.content_part.added", "output_index": 0, "item_id": "msg_1", "content_index": 0,
         "part": {"type": "output_text", "text": "", "annotations": []}},
        {"type": "response.output_text.delta", "output_index": 0, "item_id": "msg_1", "content_index": 0,
         "delta": text[:8], "logprobs": []},
        {"type": "response.output_text.delta", "output_index": 0, "item_id": "msg_1", "content_index": 0,
         "delta": text[8:], "logprobs": []},
        {"type": "response.output_text.done", "output_index": 0, "item_id": "msg_1", "content_index": 0,
         "text": text, "logprobs": []},
        {"type": "response.content_part.done", "output_index": 0, "item_id": "msg_1", "content_index": 0,
         "part": message["content"][0]},
        {"type": "response.output_item.done", "output_index": 0, "item": message},
    ]
    if not structured:
        call = final["output"][1]
        events += [
            {"type": "response.output_item.added", "output_index": 1, "item": dict(call, arguments="", status="in_progress")},
            {"type": "response.function_call_arguments.delta", "output_index": 1, "item_id": "fc_1", "delta": call["arguments"]},
            {"type": "response.function_call_arguments.done", "output_index": 1, "item_id": "fc_1", "arguments": call["arguments"]},
            {"type": "response.output_item.done", "output_index": 1, "item": call},
        ]
    events.append({"type": "response.completed", "response": final})
    for number, event in enumerate(events):
        event["sequence_number"] = number
    return events


def message(structured):
    content = [{"type": "text", "text": JSON_TEXT if structured else TEXT}]
    if not structured:
        content.append({"type": "tool_use", "id": "toolu_1", "name": "get_unit", "input": {"unit": "4B"}})
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-fake", "content": content,
            "stop_reason": "end_turn" if structured else "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 4200, "output_tokens": 42}}


def message_events(structured):
    text = JSON_TEXT if structured else TEXT
    events = [
        {"type": "message_start", "message": {"id": "msg_2", "type": "message", "role": "assistant", "model": "claude-fake",
                                              "content": [], "stop_reason": None, "stop_sequence": None,
                                              "usage": {"input_tokens": 4200, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text[:8]}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text[8:]}},
        {"type": "content_block_stop", "index": 0},
    ]
    if not structured:
        events += [
            {"type": "content_block_start", "index": 1,
             "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_unit", "input": {}}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"unit": '}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '"4B"}'}},
            {"type": "content_block_stop", "index": 1},
        ]
    events += [{"type": "message_delta", "delta": {"stop_reason": "end_turn" if structured else "tool_use",
                                                   "stop_sequence": None}, "usage": {"output_tokens": 42}},
               {"type": "message_stop"}]
    return events


GEMINI = {"candidates": [{"content": {"role": "model", "parts": [
    {"text": TEXT}, {"functionCall": {"name": "get_unit", "args": {"unit": "4B"}}}]}, "finishReason": "STOP", "index": 0}],
    "usageMetadata": {"promptTokenCount": 4200, "candidatesTokenCount": 42, "totalTokenCount": 4242},
    "modelVersion": "gemini-fake", "responseId": "gem_1"}
GEMINI_CHUNKS = [
    {"candidates": [{"content": {"role": "model", "parts": [{"text": "Unit 4B "}]}, "index": 0}], "modelVersion": "gemini-fake",
     "responseId": "gem_2"},
    {"candidates": [{"content": {"role": "model", "parts": [{"text": "is open."}]}, "index": 0}], "modelVersion": "gemini-fake",
     "responseId": "gem_2"},
    {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "get_unit", "args": {"unit": "4B"}}}]},
                     "finishReason": "STOP", "index": 0}],
     "usageMetadata": {"promptTokenCount": 4200, "candidatesTokenCount": 42, "totalTokenCount": 4242},
     "modelVersion": "gemini-fake", "responseId": "gem_2"},
]
CONVERSE = {"output": {"message": {"role": "assistant", "content": [
    {"text": TEXT}, {"toolUse": {"toolUseId": "tu_1", "name": "get_unit", "input": {"unit": "4B"}}}]}},
    "stopReason": "tool_use", "usage": {"inputTokens": 4200, "outputTokens": 42, "totalTokens": 4242},
    "metrics": {"latencyMs": 5}}
CONVERSE_EVENTS = [
    ("messageStart", {"role": "assistant"}),
    ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Unit 4B "}}),
    ("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "is open."}}),
    ("contentBlockStop", {"contentBlockIndex": 0}),
    ("contentBlockStart", {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "tu_1", "name": "get_unit"}}}),
    ("contentBlockDelta", {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"unit": '}}}),
    ("contentBlockDelta", {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '"4B"}'}}}),
    ("contentBlockStop", {"contentBlockIndex": 1}),
    ("messageStop", {"stopReason": "tool_use"}),
    ("metadata", {"usage": {"inputTokens": 4200, "outputTokens": 42, "totalTokens": 4242}, "metrics": {"latencyMs": 5}}),
]


def aws_event(event_type, payload):
    """One AWS event stream message (binary framing with CRC32s)."""
    def header(name, value):
        name, value = name.encode(), value.encode()
        return bytes([len(name)]) + name + bytes([7]) + struct.pack(">H", len(value)) + value
    headers = header(":event-type", event_type) + header(":content-type", "application/json") + header(":message-type", "event")
    body = json.dumps(payload).encode()
    prelude = struct.pack(">II", 12 + len(headers) + len(body) + 4, len(headers))
    message = prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + body
    return message + struct.pack(">I", zlib.crc32(message))


ERRORS = {"err-400": 400, "err-429": 429, "err-500": 500}
AWS_ERRORS = {400: "ValidationException", 429: "ThrottlingException", 500: "InternalServerException"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    calls = []  # Calls received by the fake collector.
    provider_requests = []
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def do_GET(self):
        """GET /__calls drains what the fake collector received (used by the Node harness)."""
        if self.path == "/__calls":
            with Handler.lock:
                calls = list(Handler.calls)
                Handler.calls.clear()
            return self.send_json(200, calls)
        self.send_json(404, {})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        path = self.path.split("?")[0]
        if path.startswith("/api/platform-collect"):
            if path.endswith("/sdk"):
                with Handler.lock:
                    Handler.calls.extend(json.loads(raw).get("calls") or [])
            return self.send_json(200, {})
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = {}
        with Handler.lock:
            Handler.provider_requests.append(path)
        model = str(body.get("model") or "") + " " + path
        for trigger in ("err-400", "err-429", "err-500", "err-timeout"):
            if trigger in model:
                return self.fail(trigger, path)
        structured = bool(body.get("response_format") or (body.get("text") or {}).get("format")
                          or body.get("output_format") or body.get("output_config"))
        stream = bool(body.get("stream"))
        if path.endswith("/chat/completions"):
            return self.sse([("", c) for c in chat_chunks(structured)], done=True) if stream else self.send_json(200, chat(structured))
        if path.endswith("/responses"):
            return self.sse([(e["type"], e) for e in response_events(structured)]) if stream else self.send_json(200, response(structured))
        if path.endswith("/messages") or path.endswith(":rawPredict") or path.endswith(":streamRawPredict"):
            if stream or path.endswith(":streamRawPredict"):
                return self.sse([(e["type"], e) for e in message_events(structured)])
            return self.send_json(200, message(structured))
        if path.endswith("/invoke"):  # AnthropicBedrock
            return self.send_json(200, message(structured))
        if path.endswith("/invoke-with-response-stream"):
            return self.aws_stream([("chunk", {"bytes": base64.b64encode(json.dumps(e).encode()).decode()})
                                    for e in message_events(structured)])
        if path.endswith(":generateContent"):
            return self.send_json(200, GEMINI)
        if path.endswith(":streamGenerateContent"):
            return self.sse([("", c) for c in GEMINI_CHUNKS])
        if path.endswith("/converse"):
            return self.send_json(200, CONVERSE)
        if path.endswith("/converse-stream"):
            return self.aws_stream(CONVERSE_EVENTS)
        self.send_json(404, {"error": {"message": "fake: no route " + path}})

    def fail(self, trigger, path):
        if trigger == "err-timeout":
            time.sleep(TIMEOUT_SLEEP)
            try:
                self.send_json(200, {})
            except Exception:
                pass
            return
        status = ERRORS[trigger]
        if path.endswith("/converse") or path.endswith("/converse-stream") or "/model/" in path:
            return self.send_json(status, {"message": f"fake {status}"},
                                  {"x-amzn-ErrorType": AWS_ERRORS[status] + ":http://internal.amazon.com/coral/"})
        if ":generateContent" in path or ":streamGenerateContent" in path:
            names = {400: "INVALID_ARGUMENT", 429: "RESOURCE_EXHAUSTED", 500: "INTERNAL"}
            return self.send_json(status, {"error": {"code": status, "message": f"fake {status}", "status": names[status]}})
        if path.endswith("/messages") or "Predict" in path:
            kinds = {400: "invalid_request_error", 429: "rate_limit_error", 500: "api_error"}
            return self.send_json(status, {"type": "error", "error": {"type": kinds[status], "message": f"fake {status}"}})
        return self.send_json(status, {"error": {"message": f"fake {status}", "type": "fake_error", "code": None, "param": None}})

    def send_json(self, status, body, headers=None):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def sse(self, events, done=False):
        text = "".join((f"event: {name}\n" if name else "") + f"data: {json.dumps(data)}\n\n" for name, data in events)
        data = (text + ("data: [DONE]\n\n" if done else "")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def aws_stream(self, events):
        data = b"".join(aws_event(name, payload) for name, payload in events)
        self.send_response(200)
        self.send_header("Content-Type", "application/vnd.amazon.eventstream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass  # Clients that time out close the socket mid-reply; that is the point.


def start():
    server = Server(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}"


if __name__ == "__main__":
    # Standalone: print the URL, serve until stdin closes (the parent process exits).
    import sys
    _, url = start()
    print(url, flush=True)
    sys.stdin.read()
