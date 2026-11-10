"""Local fakes for the Layer 2 robustness stress tests. Standard library only.

    python3 fakes.py PORTFILE

Starts two servers on 127.0.0.1 and writes {"provider": port, "collector": port, "dead": port}
to PORTFILE:

- provider: a fake OpenAI (/v1/chat/completions, JSON or SSE) and Anthropic (/v1/messages)
  endpoint that answers instantly and echoes the last user message back as the reply text.
  It is also the control plane: POST /_mode {"mode": ...}, GET /_stats, GET /_calls,
  POST /_reset, POST /_scan {"needles": [...]}.
- collector: a fake General Context collector whose behavior follows the mode:
  healthy | dead (connection refused: the socket is closed) | blackhole (accepts, never answers)
  | slow (answers after 30 s) | 500 | 429 | 401 | garbage (non-HTTP bytes, then close).
  In healthy mode the body must be strict JSON (NaN/Infinity rejected with 400, like the real
  server), and every call is stored.
- dead: a port that was bound then closed, so connecting is refused.

No real model API is ever contacted. Exits when stdin closes, so it never outlives its parent.
"""

import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.setrecursionlimit(100_000)  # Deeply nested payloads must parse, like a JS server would.
STATE = {"mode": "healthy", "calls": [], "bodies": [], "posts": 0, "rejected": 0}
LOCK = threading.Lock()
STOP = threading.Event()


def _strict(text):
    def bad(value):
        raise ValueError(f"non-standard JSON constant {value}")
    return json.loads(text, parse_constant=bad)


def _reply_text(request):
    messages = request.get("messages") or []
    last = messages[-1] if messages else {}
    content = last.get("content") if isinstance(last, dict) else None
    if isinstance(content, list):
        content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "ok")[:2000]


class Provider(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, status, body, content_type="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        length = int(self.headers.get("content-length") or 0)
        return self.rfile.read(length) if length else b""

    def do_GET(self):
        if self.path == "/_stats":
            with LOCK:
                return self._send(200, {"mode": STATE["mode"], "calls": len(STATE["calls"]),
                                        "posts": STATE["posts"], "rejected": STATE["rejected"]})
        if self.path == "/_reset":
            with LOCK:
                STATE.update(calls=[], bodies=[], posts=0, rejected=0)
            return self._send(200, {})
        if self.path == "/_calls":
            with LOCK:
                return self._send(200, STATE["calls"])
        self._send(404, {})

    def do_POST(self):
        raw = self._body()
        path = self.path.split("?")[0]
        if path == "/_mode":
            mode = json.loads(raw)["mode"]
            with LOCK:
                STATE["mode"] = mode
            COLLECTOR.set_mode(mode)
            return self._send(200, {"mode": mode})
        if path == "/_reset":
            with LOCK:
                STATE.update(calls=[], bodies=[], posts=0, rejected=0)
            return self._send(200, {})
        if path == "/_scan":
            needles = json.loads(raw)["needles"]
            with LOCK:
                bodies = list(STATE["bodies"])
            hits = sorted({n for n in needles for b in bodies if n in b})
            return self._send(200, {"bodies": len(bodies), "hits": hits})
        try:
            request = json.loads(raw or b"{}")
        except ValueError:
            request = {}
        text = _reply_text(request)
        if path.endswith("/chat/completions"):
            if request.get("stream"):
                return self._sse([
                    {"id": "c2", "object": "chat.completion.chunk", "created": 1, "model": "fake",
                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}]},
                    {"id": "c2", "object": "chat.completion.chunk", "created": 1, "model": "fake",
                     "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ], openai=True)
            return self._send(200, {"id": "c1", "object": "chat.completion", "created": 1, "model": "fake",
                                    "choices": [{"index": 0, "finish_reason": "stop",
                                                 "message": {"role": "assistant", "content": text}}],
                                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        if path.endswith("/messages"):
            message = {"id": "m1", "type": "message", "role": "assistant", "model": "fake",
                       "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
                       "usage": {"input_tokens": 1, "output_tokens": 1}}
            if request.get("stream"):
                return self._sse([
                    {"type": "message_start", "message": dict(message, content=[])},
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
                    {"type": "content_block_stop", "index": 0},
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
                    {"type": "message_stop"},
                ], openai=False)
            return self._send(200, message)
        self._send(404, {"error": "unknown path"})

    def _sse(self, events, openai):
        out = []
        for event in events:
            prefix = "" if openai else f"event: {event['type']}\n"
            out.append(f"{prefix}data: {json.dumps(event)}\n\n")
        if openai:
            out.append("data: [DONE]\n\n")
        self._send(200, "".join(out).encode(), "text/event-stream")


class Collector:
    """A raw socket server, so dead / blackhole / garbage can be modeled exactly."""

    def __init__(self):
        self.mode = "healthy"
        self.listener = None
        self.port = None
        self.lock = threading.Lock()
        self._listen()

    def _listen(self, port=0):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen(512)
        self.listener, self.port = listener, listener.getsockname()[1]
        threading.Thread(target=self._accept, args=(listener,), daemon=True).start()

    def set_mode(self, mode):
        with self.lock:
            previous, self.mode = self.mode, mode
            if mode == "dead" and previous != "dead":
                self.listener.close()
            elif mode != "dead" and previous == "dead":
                self._listen(self.port)

    def _accept(self, listener):
        while not STOP.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        try:
            conn.settimeout(120)
            reader = conn.makefile("rb")
            while True:
                line = reader.readline()
                if not line:
                    return
                headers = {}
                while True:
                    header = reader.readline()
                    if header in (b"\r\n", b"\n", b""):
                        break
                    key, _, value = header.decode("latin-1").partition(":")
                    headers[key.strip().lower()] = value.strip()
                body = reader.read(int(headers.get("content-length") or 0))
                mode = self.mode
                if mode == "blackhole":
                    STOP.wait(3600)
                    return
                if mode == "garbage":
                    conn.sendall(b"\x00\xffthis is not http\r\n\r\n{{{")
                    return
                if mode == "slow":
                    if STOP.wait(30):
                        return
                status, payload = 200, b'{"ok":true}'
                if mode in ("500", "429", "401"):
                    status, payload = int(mode), b'{"error":"fake"}'
                else:
                    try:
                        parsed = _strict(body.decode("utf-8"))
                        calls = parsed.get("calls", []) if isinstance(parsed, dict) else []
                        with LOCK:
                            STATE["posts"] += 1
                            STATE["calls"].extend(calls)
                            STATE["bodies"].append(body.decode("utf-8"))
                    except Exception as error:
                        with LOCK:
                            STATE["rejected"] += 1
                        status, payload = 400, json.dumps({"error": str(error)[:200]}).encode()
                reason = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 429: "Too Many Requests",
                          500: "Internal Server Error"}[status]
                conn.sendall(f"HTTP/1.1 {status} {reason}\r\ncontent-type: application/json\r\n"
                             f"content-length: {len(payload)}\r\n\r\n".encode() + payload)
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass


def main():
    global COLLECTOR
    COLLECTOR = Collector()
    provider = ThreadingHTTPServer(("127.0.0.1", 0), Provider)
    provider.daemon_threads = True
    provider.request_queue_size = 512
    dead = socket.socket()
    dead.bind(("127.0.0.1", 0))
    dead_port = dead.getsockname()[1]
    dead.close()
    threading.Thread(target=provider.serve_forever, daemon=True).start()
    with open(sys.argv[1], "w") as handle:
        json.dump({"provider": provider.server_address[1], "collector": COLLECTOR.port, "dead": dead_port}, handle)
    try:
        sys.stdin.read()  # Parent closed our stdin: shut down.
    finally:
        STOP.set()
        provider.shutdown()


COLLECTOR = None

if __name__ == "__main__":
    main()
