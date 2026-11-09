# Copyright 2026 General Context Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Record your agent's model calls for General Context, without changing how it runs.

    import generalcontext
    generalcontext.initialize()            # once per process, reads GENERALCONTEXT_API_KEY

    with generalcontext.session(conversation_id, user_id="u_42"):
        client.chat.completions.create(...)   # any OpenAI or Anthropic client, anywhere

`initialize()` patches the installed `openai` and `anthropic` client classes, so every
client is recorded: chat.completions, responses and messages, sync and async, streamed or
not. With `instrument_http=True` it also records raw `httpx` and `requests` POSTs to
`/chat/completions`. Calls are sent from a background thread. Your agent never waits on
General Context, and a failure on our side is logged and dropped, never raised.

`redact=True` replaces emails, phone numbers and card or SSN-like numbers with stable
placeholders (<phone:3f2a1c>) before anything leaves this process. Set
GENERALCONTEXT_REDACT_SALT to one secret in every process so placeholders match.

Standard library only. `wrap(client)` and `conversation(...)` from 0.1 still work.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import functools
import hashlib
import hmac
import inspect
import json
import logging
import os
import queue
import re
import threading
import urllib.request
import uuid

__all__ = ["initialize", "session", "with_session", "conversation", "task", "flush",
           "report_outcome", "log_event", "wrap"]
__version__ = "0.2.0"

DEFAULT_BASE_URL = "https://generalcontextlabs.com"
BATCH = 20
log = logging.getLogger("generalcontext")

_session = ContextVar("generalcontext_session", default=None)
_task = ContextVar("generalcontext_task", default=None)
# Set while a patched SDK call runs, so the call is recorded once even when it is wrapped
# twice or its HTTP request passes through a patched transport.
_inside = ContextVar("generalcontext_inside", default=False)

_config = {"initialized": False, "api_key": None, "base_url": None, "app": None, "scrub": None, "batch": BATCH}


def _api_key():
    return _config["api_key"] or os.environ.get("GENERALCONTEXT_API_KEY")


def _base_url():
    return (_config["base_url"] or os.environ.get("GENERALCONTEXT_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _now():
    return datetime.now(timezone.utc).isoformat()


# ---- Setup -------------------------------------------------------------------------------

def initialize(api_key=None, base_url=None, app_name=None, redact=False, instrument_http=True, disable_batch=False):
    """Record every OpenAI and Anthropic call in this process. Call once, in every process."""
    if _config["initialized"]:
        log.warning("generalcontext.initialize() was already called; this call is ignored")
        return
    _config.update(initialized=True, api_key=api_key, base_url=base_url, app=app_name,
                   scrub=_redactor(redact), batch=1 if disable_batch else BATCH)
    if not _api_key():
        log.warning("GENERALCONTEXT_API_KEY is not set; model calls will not be recorded")
    _patch_sdks()
    if instrument_http:
        _patch_http()


def _patch_sdks():
    targets = [
        ("openai.resources.chat.completions", "Completions", "AsyncCompletions", "openai-chat"),
        ("openai.resources.responses", "Responses", "AsyncResponses", "openai-responses"),
        ("anthropic.resources.messages", "Messages", "AsyncMessages", "anthropic"),
    ]
    for module_name, sync_name, async_name, api in targets:
        try:
            module = __import__(module_name, fromlist=[sync_name])
        except Exception:
            continue  # Package not installed.
        for name, is_async in ((sync_name, False), (async_name, True)):
            cls = getattr(module, name, None)
            if cls is None or getattr(cls.create, "_generalcontext", False):
                continue
            cls.create = _recording(cls.create, api, is_async)
            if api == "anthropic" and hasattr(cls, "stream"):
                cls.stream = _recording_stream_manager(cls.stream, is_async)


def wrap(client, *, redact=False):
    """Record every model call `client` makes. Returns the same client. Prefer initialize()."""
    scrub = _redactor(redact)
    targets = []
    if hasattr(client, "chat") and hasattr(getattr(client.chat, "completions", None), "create"):
        targets.append((client.chat.completions, "openai-chat"))
    if hasattr(getattr(client, "responses", None), "create"):
        targets.append((client.responses, "openai-responses"))
    if hasattr(getattr(client, "messages", None), "create") and not hasattr(client, "chat"):
        targets.append((client.messages, "anthropic"))
    if not targets:
        raise TypeError("generalcontext.wrap expects an OpenAI or Anthropic client")
    for owner, api in targets:
        create = owner.create
        if not getattr(create, "_generalcontext_instance", False):
            owner.create = _recording(create, api, inspect.iscoroutinefunction(inspect.unwrap(create)), scrub)
            owner.create._generalcontext_instance = True
    return client


# ---- Sessions and tasks --------------------------------------------------------------------

@contextmanager
def session(session_id, user_id=None, channel=None, product=None, metadata=None):
    """Tag every call inside the block as part of conversation `session_id`. Safe across threads and asyncio."""
    tags = {key: value for key, value in dict(id=str(session_id), user_id=user_id, channel=channel,
                                               product=product, metadata=metadata).items() if value is not None}
    token = _session.set(tags)
    try:
        yield
    finally:
        _session.reset(token)


with_session = session
conversation = session


@contextmanager
def task(name):
    """Mark every call inside the block as part of the unit of work `name`."""
    token = _task.set(str(name))
    try:
        yield
    finally:
        _task.reset(token)


# ---- Sending -----------------------------------------------------------------------------

class _Sender:
    """One daemon thread that batches recorded calls and posts them."""

    def __init__(self):
        self.queue = queue.Queue(maxsize=5000)
        self.thread = None
        self.lock = threading.Lock()
        self.warned = False

    def put(self, call):
        if not _api_key():
            if not self.warned:
                log.warning("GENERALCONTEXT_API_KEY is not set; model calls are not recorded")
                self.warned = True
            return
        with self.lock:
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._run, name="generalcontext", daemon=True)
                self.thread.start()
        try:
            self.queue.put_nowait(call)
        except queue.Full:
            log.warning("General Context queue is full; a model call was dropped")

    def _run(self):
        while True:
            batch = [self.queue.get()]
            while len(batch) < _config["batch"]:
                try:
                    batch.append(self.queue.get(timeout=0.5))
                except queue.Empty:
                    break
            try:
                _post("/api/platform-collect/sdk", {"calls": batch, "sdk": f"python/{__version__}", "app": _config["app"]})
            except Exception as error:  # Never let our failure reach the agent.
                log.warning("General Context did not accept %d calls: %s", len(batch), error)
            finally:
                for _ in batch:
                    self.queue.task_done()


def _post(path, body):
    data = json.dumps(body, default=str).encode()
    request = urllib.request.Request(_base_url() + path, data=data, headers={
        "Authorization": "Bearer " + (_api_key() or ""), "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        response.read()


_sender = _Sender()


def flush(timeout=10):
    """Wait until recorded calls are sent. Call before a short-lived process exits. True if all were sent."""
    try:
        done = threading.Event()
        threading.Thread(target=lambda: (_sender.queue.join(), done.set()), daemon=True).start()
        return done.wait(timeout)
    except Exception as error:
        log.warning("General Context flush failed: %s", error)
        return False


def _send_after_flush(path, body, what):
    """Flush first, so the conversation exists when the server receives `body`."""
    try:
        flush()
        _post(path, body)
        return True
    except Exception as error:
        log.warning("General Context did not accept the %s: %s", what, error)
        return False


def report_outcome(session_id, outcome, note=None):
    """Mark a conversation as a success or a failure. True if General Context accepted it."""
    if outcome not in ("success", "fail"):
        log.warning("report_outcome expects 'success' or 'fail', got %r", outcome)
        return False
    return _send_after_flush("/api/platform-collect/outcomes",
                             {"session_id": str(session_id), "outcome": outcome, "note": note}, "outcome")


def log_event(session_id, event_type, properties=None, event_id=None, timestamp=None):
    """Record something that happened in your systems (order.refunded, ticket.escalated). Pass a stable event_id so retries dedupe."""
    event = {"event_id": event_id or str(uuid.uuid4()), "event_type": event_type, "timestamp": timestamp or _now(),
             "session_id": str(session_id), "properties": properties or {}}
    return _send_after_flush("/api/platform-collect/events", {"events": [event]}, "event")


def _record(meta, response=None, error=None):
    """Enqueue one finished call. Never raises."""
    try:
        call = dict(meta["call"], response=_plain(response) if response is not None else None,
                    ended_at=_now(), error=f"{type(error).__name__}: {error}" if error is not None else None)
        _sender.put(_scrubbed(call, meta["scrub"]))
    except Exception as failure:
        log.warning("General Context could not record a model call: %s", failure)


def _start(api, request, scrub=None):
    return {"scrub": scrub or _config["scrub"], "call": {
        "api": api, "conversation": _session.get(), "task": _task.get(),
        "request": _plain(request), "started_at": _now()}}


# ---- SDK patching --------------------------------------------------------------------------

def _recording(create, api, is_async, scrub=None):
    """Wrap a `create` method so each call is recorded. The caller gets exactly what `create` returns."""
    if is_async:
        @functools.wraps(create)
        async def wrapped(*args, **kwargs):
            if _inside.get():
                return await create(*args, **kwargs)
            token = _inside.set(True)
            meta = _start(api, kwargs, scrub)
            try:
                reply = await create(*args, **kwargs)
            except Exception as error:
                _record(meta, error=error)
                raise
            finally:
                _inside.reset(token)
            if kwargs.get("stream") is True:
                return _AsyncStream(reply, meta)
            _record(meta, reply)
            return reply
    else:
        @functools.wraps(create)
        def wrapped(*args, **kwargs):
            if _inside.get():
                return create(*args, **kwargs)
            token = _inside.set(True)
            meta = _start(api, kwargs, scrub)
            try:
                reply = create(*args, **kwargs)
            except Exception as error:
                _record(meta, error=error)
                raise
            finally:
                _inside.reset(token)
            if kwargs.get("stream") is True:
                return _SyncStream(reply, meta)
            _record(meta, reply)
            return reply
    wrapped._generalcontext = True
    return wrapped


class _StreamBase:
    """Passes chunks through unchanged and records the assembled reply once the stream ends."""

    def __init__(self, inner, meta):
        self._inner, self._meta, self._chunks, self._done = inner, meta, [], False

    def _finish(self, error=None):
        if not self._done:
            self._done = True
            _record(self._meta, None if error else _assemble(self._meta["call"]["api"], self._chunks), error)

    def _keep(self, chunk):
        try:
            self._chunks.append(_plain(chunk))
        except Exception:
            pass
        return chunk

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _SyncStream(_StreamBase):
    def __iter__(self):
        return self

    def __next__(self):
        if not hasattr(self, "_iterator"):
            self._iterator = iter(self._inner)
        try:
            return self._keep(next(self._iterator))
        except StopIteration:
            self._finish()
            raise
        except Exception as error:
            self._finish(error)
            raise

    def close(self):
        try:
            self._inner.close()
        finally:
            self._finish()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class _AsyncStream(_StreamBase):
    def __aiter__(self):
        return self

    async def __anext__(self):
        if not hasattr(self, "_iterator"):
            self._iterator = self._inner.__aiter__()
        try:
            return self._keep(await self._iterator.__anext__())
        except StopAsyncIteration:
            self._finish()
            raise
        except Exception as error:
            self._finish(error)
            raise

    async def close(self):
        try:
            await self._inner.close()
        finally:
            self._finish()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()


def _recording_stream_manager(stream, is_async):
    """Wrap Anthropic `messages.stream(...)`: record the SDK's final message snapshot on exit."""

    @functools.wraps(stream)
    def wrapped(*args, **kwargs):
        return _StreamManager(stream(*args, **kwargs), _start("anthropic", dict(kwargs, stream=True)))
    wrapped._generalcontext = True
    return wrapped


class _StreamManager:
    def __init__(self, inner, meta):
        self._inner, self._meta, self._stream = inner, meta, None

    def _finish(self, error):
        try:
            snapshot = self._stream.current_message_snapshot if self._stream is not None else None
        except Exception:
            snapshot = None
        _record(self._meta, snapshot, error)

    def __enter__(self):
        self._stream = self._inner.__enter__()
        return self._stream

    def __exit__(self, kind, error, trace):
        self._finish(error)
        return self._inner.__exit__(kind, error, trace)

    async def __aenter__(self):
        self._stream = await self._inner.__aenter__()
        return self._stream

    async def __aexit__(self, kind, error, trace):
        self._finish(error)
        return await self._inner.__aexit__(kind, error, trace)


def _assemble(api, chunks):
    """Turn streamed chunks into the reply a non-streamed call would have returned."""
    if api == "openai-chat":
        return _assemble_chat(chunks)
    if api == "anthropic":
        return _assemble_anthropic(chunks)
    finals = [c["response"] for c in chunks if isinstance(c, dict) and c.get("type") in
              ("response.completed", "response.incomplete", "response.failed")]
    return finals[-1] if finals else {"events": chunks}


def _assemble_chat(chunks):
    text, tools, model, usage, finish = [], {}, None, None, None
    for chunk in chunks:
        model = chunk.get("model") or model
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            if choice.get("index", 0) != 0:
                continue
            delta = choice.get("delta") or {}
            text.append(delta.get("content") or "")
            for call in delta.get("tool_calls") or []:
                slot = tools.setdefault(call.get("index", 0), {"id": None, "type": "function",
                                                                "function": {"name": "", "arguments": ""}})
                slot["id"] = call.get("id") or slot["id"]
                function = call.get("function") or {}
                slot["function"]["name"] += function.get("name") or ""
                slot["function"]["arguments"] += function.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    message = {"role": "assistant", "content": "".join(text) or None}
    if tools:
        message["tool_calls"] = [tools[index] for index in sorted(tools)]
    return {"model": model, "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}


def _assemble_anthropic(events):
    message, blocks = {}, {}
    for event in events:
        kind = event.get("type")
        if kind == "message_start":
            message = dict(event.get("message") or {})
        elif kind == "content_block_start":
            blocks[event["index"]] = dict(event.get("content_block") or {})
        elif kind == "content_block_delta":
            block, delta = blocks.setdefault(event["index"], {}), event.get("delta") or {}
            for field, key in (("text", "text"), ("partial_json", "_json"), ("thinking", "thinking")):
                if field in delta:
                    block[key] = block.get(key, "") + delta[field]
            if "signature" in delta:
                block["signature"] = delta["signature"]
        elif kind == "message_delta":
            message.update(event.get("delta") or {})
            message["usage"] = {**(message.get("usage") or {}), **(event.get("usage") or {})}
    for block in blocks.values():
        if "_json" in block:
            raw = block.pop("_json")
            try:
                block["input"] = json.loads(raw) if raw else {}
            except ValueError:
                block["input"] = raw
    message["content"] = [blocks[index] for index in sorted(blocks)]
    return message


# ---- Raw HTTP capture ----------------------------------------------------------------------

def _patch_http():
    try:
        import httpx
    except ImportError:
        httpx = None
    if httpx is not None and not getattr(httpx.Client.send, "_generalcontext", False):
        sync_send, async_send = httpx.Client.send, httpx.AsyncClient.send

        def send(self, request, *args, **kwargs):
            meta = _http_start(request.url, lambda: request.content)
            if meta is None:
                return sync_send(self, request, *args, **kwargs)
            return _http_finish(meta, lambda: sync_send(self, request, *args, **kwargs), kwargs.get("stream"),
                                lambda r: r.text)

        async def asend(self, request, *args, **kwargs):
            meta = _http_start(request.url, lambda: request.content)
            if meta is None:
                return await async_send(self, request, *args, **kwargs)
            try:
                response = await async_send(self, request, *args, **kwargs)
            except Exception as error:
                _record(meta, error=error)
                raise
            return _http_finish(meta, lambda: response, kwargs.get("stream"), lambda r: r.text)

        send._generalcontext = asend._generalcontext = True
        httpx.Client.send, httpx.AsyncClient.send = send, asend
    try:
        import requests
    except ImportError:
        return
    if not getattr(requests.Session.send, "_generalcontext", False):
        requests_send = requests.Session.send

        def rsend(self, request, **kwargs):
            meta = _http_start(request.url, lambda: request.body)
            if meta is None:
                return requests_send(self, request, **kwargs)
            return _http_finish(meta, lambda: requests_send(self, request, **kwargs), kwargs.get("stream"),
                                lambda r: r.text)

        rsend._generalcontext = True
        requests.Session.send = rsend


def _http_start(url, body):
    """Meta for a raw chat completions POST, or None when it is not one to record."""
    try:
        url = str(url).split("?")[0]
        if _inside.get() or not url.endswith("/chat/completions") or url.startswith(_base_url() + "/api/platform-collect"):
            return None
        raw = body()
        return _start("openai-chat", json.loads(raw.decode() if isinstance(raw, bytes) else raw))
    except Exception:
        return None


def _http_finish(meta, send, streamed, text):
    try:
        response = send()
    except Exception as error:
        _record(meta, error=error)
        raise
    if streamed:
        return response  # The body is still unread; reading it here would change what the caller gets.
    try:
        body = text(response)
        if body.lstrip().startswith("data:"):
            chunks = [json.loads(line[5:]) for line in body.splitlines()
                      if line.startswith("data:") and line[5:].strip() != "[DONE]"]
            _record(meta, _assemble_chat(chunks))
        else:
            _record(meta, json.loads(body))
    except Exception as error:
        log.warning("General Context could not read a chat completions response: %s", error)
    return response


# ---- Redaction -----------------------------------------------------------------------------

PATTERNS = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("phone", re.compile(r"(?<![\w])\+?1?[\s.-]?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?![\w])")),
]


def _redactor(redact):
    if callable(redact):
        return redact
    if not redact:
        return None
    # A secret salt, the same across your processes, keeps placeholders stable everywhere while
    # making them unlinkable to the raw value for anyone without it.
    salt = os.environ.get("GENERALCONTEXT_REDACT_SALT", "").encode()
    if not salt:
        log.warning("GENERALCONTEXT_REDACT_SALT is not set; placeholders will differ between processes")
        salt = os.urandom(16)

    def placeholder(kind, value):
        # Hash the normalized value, so (612) 555-0177 and +1 612-555-0177 read as the same number.
        digits = "".join(ch for ch in value if ch.isdigit())
        normal = value.lower() if kind == "email" else digits[-10:] if kind == "phone" else digits
        return f"<{kind}:{hmac.new(salt, normal.encode(), hashlib.sha256).hexdigest()[:6]}>"

    def scrub(text):
        for kind, pattern in PATTERNS:
            text = pattern.sub(lambda match: placeholder(kind, match.group()), text)
        return text
    return scrub


def _scrubbed(value, scrub):
    if scrub is None:
        return value
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {key: _scrubbed(item, scrub) for key, item in value.items()}
    if isinstance(value, list):
        return [_scrubbed(item, scrub) for item in value]
    return value


def _plain(value, depth=0):
    """SDK objects (pydantic models) as plain JSON. Request bodies may hold anything; keep what serializes."""
    if depth > 40:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(key): _plain(item, depth + 1) for key, item in value.items()
                if not str(key).startswith(("extra_headers", "timeout"))}
    if isinstance(value, (list, tuple)):
        return [_plain(item, depth + 1) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
