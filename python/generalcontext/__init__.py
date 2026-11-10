# Copyright 2026 General Context Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Record your agent's model calls for General Context, without changing how it runs.

    import generalcontext
    generalcontext.initialize()            # once per process, reads GENERALCONTEXT_API_KEY

    with generalcontext.session(conversation_id, user_id="u_42"):
        client.chat.completions.create(...)   # any OpenAI or Anthropic client, anywhere

`initialize()` patches the installed client classes, so every client is recorded, sync and
async, streamed or not:

- OpenAI (also AzureOpenAI): chat.completions create, parse and stream, responses create,
  parse and stream.
- Anthropic (also AnthropicBedrock and AnthropicVertex): messages and beta.messages create,
  parse and stream.
- Google Gen AI (`google-genai`): models.generate_content and generate_content_stream.
- AWS Bedrock (boto3 `bedrock-runtime`): converse and converse_stream.

Raw `httpx` and `requests` POSTs to `/chat/completions` are recorded only with
`instrument_http=True`. Calls are sent from a background thread. Your agent never waits on
General Context, and a failure on our side is logged and dropped, never raised.

`redact=True` replaces emails, phone numbers and card or SSN-like numbers with stable
placeholders (<phone:3f2a1c>) before anything leaves this process. Set
GENERALCONTEXT_REDACT_SALT to one secret in every process so placeholders match.

Standard library only. `wrap(client)` and `conversation(...)` from 0.1 still work.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import atexit
import functools
import hashlib
import hmac
import inspect
import json
import logging
import math
import os
import queue
import re
import threading
import time
import urllib.parse
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

_config = {"initialized": False, "api_key": None, "base_url": None, "app": None, "scrub": None, "batch": BATCH,
           "before_send": None}


def _api_key():
    return _config["api_key"] or os.environ.get("GENERALCONTEXT_API_KEY")


def _base_url():
    return (_config["base_url"] or os.environ.get("GENERALCONTEXT_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _now():
    return datetime.now(timezone.utc).isoformat()


def _disabled():
    """GENERALCONTEXT_DISABLED=1 turns the SDK off without a code change."""
    return os.environ.get("GENERALCONTEXT_DISABLED", "").strip().lower() in ("1", "true", "yes")


# ---- Setup -------------------------------------------------------------------------------

def initialize(api_key=None, base_url=None, app_name=None, redact=False, before_send=None,
               instrument_http=False, disable_batch=False):
    """Record every OpenAI and Anthropic call in this process. Call once, in every process.

    SDK clients are always recorded. Raw httpx/requests POSTs to /chat/completions are
    recorded only with instrument_http=True. before_send(call) runs on the sender thread and
    returns the dict to send, or None to drop the call.
    """
    if _disabled():
        log.info("GENERALCONTEXT_DISABLED is set; General Context is off")
        return
    if _config["initialized"]:
        log.warning("generalcontext.initialize() was already called; this call is ignored")
        return
    _config.update(initialized=True, api_key=api_key, base_url=base_url, app=app_name,
                   scrub=_redactor(redact), before_send=before_send, batch=1 if disable_batch else BATCH)
    if not _api_key():
        log.warning("GENERALCONTEXT_API_KEY is not set; model calls will not be recorded")
    if not _secure(_base_url()):
        _warn_insecure()
    atexit.register(_flush_at_exit)
    _patch_sdks()
    _patch_gemini()
    _patch_bedrock()
    if instrument_http:
        _patch_http()


def _patch_sdks():
    # `stream` helpers of OpenAI call self.create, so create records them. `parse` posts directly.
    # Anthropic's `stream` helpers post directly too, so they are wrapped on their own.
    targets = [
        ("openai.resources.chat.completions", "Completions", "AsyncCompletions", "openai-chat", ()),
        # openai 1.40 to 1.92 have parse only under beta.chat.completions, and it posts directly.
        ("openai.resources.beta.chat.completions", "Completions", "AsyncCompletions", "openai-chat", ()),
        ("openai.resources.responses", "Responses", "AsyncResponses", "openai-responses", ()),
        ("anthropic.resources.messages", "Messages", "AsyncMessages", "anthropic", ("stream",)),
        ("anthropic.resources.beta.messages.messages", "Messages", "AsyncMessages", "anthropic", ("stream",)),
        # AnthropicBedrock and AnthropicVertex beta copy the first party methods at import.
        ("anthropic.lib.bedrock._beta_messages", "Messages", "AsyncMessages", "anthropic", ("stream",)),
        ("anthropic.lib.vertex._beta_messages", "Messages", "AsyncMessages", "anthropic", ("stream",)),
    ]
    for module_name, sync_name, async_name, api, managers in targets:
        try:
            module = __import__(module_name, fromlist=[sync_name])
        except Exception:
            continue  # Package not installed.
        for name, is_async in ((sync_name, False), (async_name, True)):
            cls = getattr(module, name, None)
            if cls is None:
                continue
            for method in ("create", "parse"):
                original = vars(cls).get(method)
                if original is not None and not getattr(original, "_generalcontext", False):
                    setattr(cls, method, _recording(original, api, is_async))
            for method in managers:
                original = vars(cls).get(method)
                if original is not None and not getattr(original, "_generalcontext", False):
                    setattr(cls, method, _recording_stream_manager(original, is_async, api))


def wrap(client, *, redact=False):
    """Record every model call `client` makes. Returns the same client. Prefer initialize()."""
    if _disabled():
        return client
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

MAX_CALL_BYTES = 4_000_000     # The server rejects larger calls.
MAX_QUEUE_BYTES = 64_000_000
MAX_BATCH_BYTES = 4_000_000
MAX_QUEUE_CALLS = 5000
_warned_at = {}


def _warn(key, message, *args):
    """Log `message` at most once per 60 seconds per `key`."""
    now = time.monotonic()
    if now - _warned_at.get(key, -1e9) >= 60:
        _warned_at[key] = now
        log.warning(message, *args)


def _secure(url):
    """HTTPS, or plain HTTP to this machine for local development."""
    parts = urllib.parse.urlsplit(url)
    return parts.scheme == "https" or (parts.scheme == "http" and parts.hostname in ("localhost", "127.0.0.1", "::1"))


def _warn_insecure():
    _warn("http", "General Context base_url %s is not https; nothing will be sent", _base_url())


def _size(value, limit):
    """Cheap size estimate of a snapshot; stops counting past `limit`."""
    total, stack = 0, [value]
    while stack and total <= limit:
        item = stack.pop()
        total += 8
        if isinstance(item, str):
            total += len(item)
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return total


class _Sender:
    """One daemon thread that redacts, encodes, batches and posts recorded calls."""

    def __init__(self):
        self._reset()
        self.warned = False

    def _reset(self):
        # Also run in a forked child: the parent's thread and lock do not exist there.
        self.queue = queue.Queue(maxsize=MAX_QUEUE_CALLS)
        self.lock = threading.Lock()
        self.bytes = 0
        self.thread = None
    def put(self, call, scrub):
        """Runs on the caller's thread, so it only measures and enqueues."""
        if not _api_key():
            if not self.warned:
                log.warning("GENERALCONTEXT_API_KEY is not set; model calls are not recorded")
                self.warned = True
            return
        if not _secure(_base_url()):
            return _warn_insecure()
        size = _size(call, MAX_CALL_BYTES)
        if size > MAX_CALL_BYTES:
            return _warn("oversized", "General Context dropped a model call larger than %d bytes", MAX_CALL_BYTES)
        with self.lock:
            if self.bytes + size > MAX_QUEUE_BYTES:
                return _warn("full", "General Context queue is full; a model call was dropped")
            try:
                self.queue.put_nowait((call, scrub, size))
            except queue.Full:
                return _warn("full", "General Context queue is full; a model call was dropped")
            self.bytes += size
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._run, name="generalcontext", daemon=True)
                self.thread.start()

    def _take(self, pending, timeout):
        """Move one call from the queue into `pending` as JSON bytes. False when the queue stayed empty."""
        try:
            call, scrub, size = self.queue.get(timeout=timeout)
        except queue.Empty:
            return False
        with self.lock:
            self.bytes -= size
        data = _encode(call, scrub)
        if data is None:
            self.queue.task_done()
        else:
            pending.append(data)
        return True

    def _run(self):
        pending = []  # Encoded calls taken from the queue; each still owes one task_done.
        while True:
            if not pending:
                self._take(pending, None)
                continue
            while (len(pending) < _config["batch"] and sum(map(len, pending)) < MAX_BATCH_BYTES
                   and self._take(pending, 0.5)):
                pass
            count, size = 0, 0
            for data in pending:
                if count and (count >= _config["batch"] or size + len(data) + 1 > MAX_BATCH_BYTES):
                    break
                count, size = count + 1, size + len(data) + 1
            batch, pending = pending[:count], pending[count:]
            try:
                head = json.dumps({"sdk": f"python/{__version__}", "app": _config["app"]})[:-1]
                _post("/api/platform-collect/sdk", (head + ', "calls": [').encode() + b",".join(batch) + b"]}", 15)
            except Exception as error:  # Never let our failure reach the agent.
                _warn("send", "General Context did not accept %d calls: %s", len(batch), error)
            finally:
                for _ in batch:
                    self.queue.task_done()


def _encode(call, scrub):
    """before_send, redaction and JSON encoding, on the sender thread. None drops the call."""
    try:
        before_send = _config["before_send"]
        if before_send is not None:
            try:
                call = before_send(call)
            except Exception as error:
                _warn("before_send", "General Context before_send raised; a call was dropped: %s", error)
                return None
            if call is None:
                return None
        # allow_nan=False: a NaN that before_send adds drops this call only, not the whole batch.
        data = json.dumps(_scrubbed(call, scrub), default=str, allow_nan=False).encode()
    except Exception as error:
        _warn("encode", "General Context could not encode a model call: %s", error)
        return None
    if len(data) > MAX_CALL_BYTES:
        _warn("oversized", "General Context dropped a model call larger than %d bytes", MAX_CALL_BYTES)
        return None
    return data


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects, so the Authorization header is never forwarded anywhere else."""

    def redirect_request(self, *args, **kwargs):
        return None  # urllib then raises HTTPError for the 3xx.


_opener = urllib.request.build_opener(_NoRedirect)  # Honors HTTPS_PROXY and SSL_CERT_FILE.


def _post(path, body, timeout):
    url = _base_url() + path
    if not _secure(url):
        _warn_insecure()
        raise ValueError("refusing to send to a non-https base_url")
    data = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
    request = urllib.request.Request(url, data=data, headers={
        "Authorization": "Bearer " + (_api_key() or ""), "Content-Type": "application/json"})
    with _opener.open(request, timeout=timeout) as response:
        response.read()


_sender = _Sender()
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_sender._reset)


def flush(timeout=10):
    """Wait until recorded calls are sent. Call before a short-lived process exits. True if all were sent."""
    try:
        done = threading.Event()
        threading.Thread(target=lambda: (_sender.queue.join(), done.set()), daemon=True).start()
        return done.wait(timeout)
    except Exception as error:
        log.warning("General Context flush failed: %s", error)
        return False


def _flush_at_exit():
    try:
        flush(2)
    except Exception:
        pass


def _send_after_flush(path, body, what):
    """Flush first, so the conversation exists when the server receives `body`."""
    try:
        if not _api_key():
            return False
        flush()
        _post(path, body, 10)
        return True
    except Exception as error:
        log.warning("General Context did not accept the %s: %s", what, error)
        return False


def _scrub_text(value):
    scrub = _config["scrub"]
    return scrub(value) if scrub is not None and isinstance(value, str) else value


def report_outcome(session_id, outcome, note=None):
    """Mark a conversation as a success or a failure. True if General Context accepted it."""
    try:
        if outcome not in ("success", "fail"):
            log.warning("report_outcome expects 'success' or 'fail', got %r", outcome)
            return False
        body = {"session_id": _scrub_text(str(session_id)), "outcome": outcome, "note": _scrub_text(note)}
    except Exception as error:
        log.warning("General Context could not build the outcome: %s", error)
        return False
    return _send_after_flush("/api/platform-collect/outcomes", body, "outcome")


def log_event(session_id, event_type, properties=None, event_id=None, timestamp=None):
    """Record something that happened in your systems (order.refunded, ticket.escalated). Pass a stable event_id so retries dedupe."""
    try:
        event = {"event_id": event_id or str(uuid.uuid4()), "event_type": _scrub_text(event_type),
                 "timestamp": timestamp or _now(), "session_id": _scrub_text(str(session_id)),
                 "properties": {key: _scrub_text(value) for key, value in (properties or {}).items()}}
    except Exception as error:
        log.warning("General Context could not build the event: %s", error)
        return False
    return _send_after_flush("/api/platform-collect/events", {"events": [event]}, "event")


CLIP_OVER = 200_000   # Strings longer than this many characters are clipped.
CLIP_HEAD = 100_000
CLIP_TAIL = 20_000
_BINARY_MIN = 256     # Shorter base64 payloads are left alone.
_BASE64 = re.compile(r"[A-Za-z0-9+/_-]+={0,2}")
_DATA_URL = re.compile(r"data:[^,;]*(?:;[^,;]*)*;base64,")
_BINARY_KEYS = ("data", "result", "b64_json")


def _omitted(payload):
    digest = hashlib.sha256(payload.encode("utf-8", "surrogatepass")).hexdigest()[:12]
    return f"[omitted {len(payload)} bytes sha256:{digest}]"


def compact_for_recording(value, key=None, depth=0):
    """The copy of a call that is recorded. Never changes `value` or what the caller sends to the provider.

    Agents resend the whole run on every call, so without this, later calls pass the per-call
    limit and are lost.

    - Inline binary becomes a placeholder that keeps the shape: a base64 data URL keeps its prefix
      (`data:image/png;base64,[omitted N bytes sha256:ab12cd34ef56]`), and a base64 `data`, `result`
      or `b64_json` field (Anthropic `source.data`, Gemini `inline_data.data`, OpenAI `input_audio.data`,
      Responses `image_generation_call.result`) becomes `[omitted N bytes sha256:...]`. N is the length
      of the base64 text and the hash is the first 12 hex digits of the sha256 of that text, so the same
      image gets the same placeholder in every call. Media types stay. Binary is never recorded: it does
      not help review and it is a privacy risk.
    - Any string over 200,000 characters keeps its first 100,000 and last 20,000 characters around
      `\\n[... clipped N characters ...]\\n`.

    Runs before before_send and redaction. A call still over the limit afterwards is dropped.
    """
    if isinstance(value, str):
        if len(value) > _BINARY_MIN:
            url = _DATA_URL.match(value[:512])
            if url:
                return value[:url.end()] + _omitted(value[url.end():])
            if key in _BINARY_KEYS and _BASE64.fullmatch(value):
                return _omitted(value)
        if len(value) > CLIP_OVER:
            clipped = len(value) - CLIP_HEAD - CLIP_TAIL
            return f"{value[:CLIP_HEAD]}\n[... clipped {clipped} characters ...]\n{value[-CLIP_TAIL:]}"
        return value
    if depth > 64:
        return value
    if isinstance(value, dict):
        return {k: compact_for_recording(v, k, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [compact_for_recording(item, None, depth + 1) for item in value]
    return value


def _record(meta, response=None, error=None):
    """Snapshot and enqueue one finished call. Never raises."""
    try:
        call = dict(meta["call"], response=_plain(response) if response is not None else None,
                    ended_at=_now(), error=f"{type(error).__name__}: {error}" if error is not None
else None)
        _sender.put(compact_for_recording(call), meta["scrub"])
    except Exception as failure:
        log.warning("General Context could not record a model call: %s", failure)


def _start(api, request, scrub=None):
    return {"scrub": scrub or _config["scrub"], "call": {
        "api": api, "conversation": _session.get(), "task": _task.get(),
        "request": _plain(request), "started_at": _now()}}


# ---- SDK patching --------------------------------------------------------------------------

def _safe_start(api, request, scrub=None):
    try:
        return _start(api, request, scrub)
    except Exception as error:
        _warn("start", "General Context could not snapshot a model call; it runs unrecorded: %s", error)
        return None


def _recording(create, api, is_async, scrub=None):
    """Wrap a `create` method so each call is recorded. The caller gets exactly what `create` returns."""
    if is_async:
        @functools.wraps(create)
        async def wrapped(*args, **kwargs):
            if _inside.get():
                return await create(*args, **kwargs)
            meta = _safe_start(api, kwargs, scrub)
            if meta is None:
                return await create(*args, **kwargs)
            token = _inside.set(True)
            try:
                reply = await create(*args, **kwargs)
            except Exception as error:
                _record(meta, error=error)
                raise
            finally:
                _inside.reset(token)
            if _is_raw_response(reply):
                body = _raw_body(reply, kwargs)
                if inspect.isawaitable(body):
                    try:
                        body = await body
                    except Exception as error:
                        _warn("raw", "General Context could not read a raw response: %s", error)
                        body = None
                _record(meta, body)
                return reply
            if kwargs.get("stream") is True:
                return _AsyncStream(reply, meta)
            _record(meta, reply)
            return reply
    else:
        @functools.wraps(create)
        def wrapped(*args, **kwargs):
            if _inside.get():
                return create(*args, **kwargs)
            meta = _safe_start(api, kwargs, scrub)
            if meta is None:
                return create(*args, **kwargs)
            token = _inside.set(True)
            try:
                reply = create(*args, **kwargs)
            except Exception as error:
                _record(meta, error=error)
                raise
            finally:
                _inside.reset(token)
            if _is_raw_response(reply):
                _record(meta, _raw_body(reply, kwargs))
                return reply
            if kwargs.get("stream") is True:
                return _SyncStream(reply, meta)
            _record(meta, reply)
            return reply
    wrapped._generalcontext = True
    return wrapped


def _is_raw_response(reply):
    """`with_raw_response` / `with_streaming_response` hand back a response wrapper, not the model."""
    return hasattr(reply, "http_response") and callable(getattr(reply, "parse", None))


def _raw_body(reply, kwargs):
    """The parsed reply of a raw response, or None when reading it would change what the caller gets.

    parse() caches its result, so the caller's own parse() returns the same object afterwards.
    Streamed and `with_streaming_response` bodies are left unread.
    """
    try:
        headers = kwargs.get("extra_headers") or {}
        if kwargs.get("stream") is True or headers.get("X-Stainless-Raw-Response") == "stream":
            return None
        return reply.parse()  # A coroutine for async raw responses; the async wrapper awaits it.
    except Exception as error:
        _warn("raw", "General Context could not read a raw response: %s", error)
        return None


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
            close = getattr(self._inner, "close", None) or getattr(self._inner, "aclose", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result
        finally:
            self._finish()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()


def _recording_stream_manager(stream, is_async, api="anthropic"):
    """Wrap Anthropic `messages.stream(...)`: record the SDK's final message snapshot on exit."""

    @functools.wraps(stream)
    def wrapped(*args, **kwargs):
        if _inside.get():
            return stream(*args, **kwargs)
        meta = _safe_start(api, dict(kwargs, stream=True))
        inner = stream(*args, **kwargs)
        return inner if meta is None else _StreamManager(inner, meta)
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
        token = _inside.set(True)
        try:
            self._stream = self._inner.__enter__()
        except Exception as error:
            _record(self._meta, error=error)
            raise
        finally:
            _inside.reset(token)
        return self._stream

    def __exit__(self, kind, error, trace):
        self._finish(error)
        return self._inner.__exit__(kind, error, trace)

    async def __aenter__(self):
        token = _inside.set(True)
        try:
            self._stream = await self._inner.__aenter__()
        except Exception as error:
            _record(self._meta, error=error)
            raise
        finally:
            _inside.reset(token)
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
    if api == "gemini":
        return _assemble_gemini(chunks)
    if api == "bedrock-converse":
        return _assemble_converse(chunks)
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


# ---- Google Gen AI and AWS Bedrock -----------------------------------------------------------

def _assemble_gemini(chunks):
    """Streamed GenerateContentResponse chunks as one response: text of candidate 0 joined in order."""
    parts, finish, usage, version, role = [], None, None, None, "model"
    for chunk in chunks:
        usage = chunk.get("usage_metadata") or usage
        version = chunk.get("model_version") or version
        for candidate in chunk.get("candidates") or []:
            if candidate.get("index", 0) != 0:
                continue
            finish = candidate.get("finish_reason") or finish
            content = candidate.get("content") or {}
            role = content.get("role") or role
            for part in content.get("parts") or []:
                if set(part) == {"text"} and parts and set(parts[-1]) == {"text"}:
                    parts[-1] = {"text": parts[-1]["text"] + part["text"]}
                else:
                    parts.append(dict(part))
    candidate = {"content": {"role": role, "parts": parts}, "index": 0}
    if finish:
        candidate["finish_reason"] = finish
    response = {"candidates": [candidate]}
    if usage:
        response["usage_metadata"] = usage
    if version:
        response["model_version"] = version
    return response


def _patch_gemini():
    try:
        from google.genai import models
    except Exception:
        return  # google-genai is not installed.
    for cls, is_async in ((getattr(models, "Models", None), False), (getattr(models, "AsyncModels", None), True)):
        if cls is None:
            continue
        original = vars(cls).get("generate_content")
        if original is not None and not getattr(original, "_generalcontext", False):
            cls.generate_content = _recording(original, "gemini", is_async)
        original = vars(cls).get("generate_content_stream")
        if original is not None and not getattr(original, "_generalcontext", False):
            cls.generate_content_stream = _recording_iterator(original, "gemini", is_async)


def _recording_iterator(method, api, is_async):
    """Wrap a method that returns (or, async, resolves to) an iterator of chunks."""
    if is_async:
        @functools.wraps(method)
        async def wrapped(*args, **kwargs):
            if _inside.get():
                return await method(*args, **kwargs)
            meta = _safe_start(api, kwargs)
            if meta is None:
                return await method(*args, **kwargs)
            token = _inside.set(True)
            try:
                inner = await method(*args, **kwargs)
            except Exception as error:
                _record(meta, error=error)
                raise
            finally:
                _inside.reset(token)
            return _AsyncStream(inner, meta)
    else:
        @functools.wraps(method)
        def wrapped(*args, **kwargs):
            if _inside.get():
                return method(*args, **kwargs)
            meta = _safe_start(api, kwargs)
            if meta is None:
                return method(*args, **kwargs)
            try:
                inner = method(*args, **kwargs)
            except Exception as error:
                _record(meta, error=error)
                raise
            return _SyncStream(_inside_iterator(inner), meta)
    wrapped._generalcontext = True
    return wrapped


def _inside_iterator(inner):
    """Iterate a lazy generator with _inside set, so its internal calls are not recorded twice."""
    iterator = iter(inner)
    while True:
        token = _inside.set(True)
        try:
            item = next(iterator)
        except StopIteration:
            return
        finally:
            _inside.reset(token)
        yield item


def _no_bytes(value, depth=0):
    """Bedrock params with images and documents replaced by {"omitted": "bytes"}."""
    if depth > 40:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)) or hasattr(value, "read"):
        return {"omitted": "bytes"}
    if isinstance(value, dict):
        return {key: _no_bytes(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_no_bytes(item, depth + 1) for item in value]
    return value


def _patch_bedrock():
    try:
        from botocore.client import BaseClient
    except Exception:
        return  # boto3 is not installed.
    original = BaseClient._make_api_call
    if getattr(original, "_generalcontext", False):
        return

    def _make_api_call(self, operation_name, api_params):
        if operation_name not in ("Converse", "ConverseStream") or _inside.get():
            return original(self, operation_name, api_params)
        meta = _safe_start("bedrock-converse", _no_bytes(api_params))
        if meta is None:
            return original(self, operation_name, api_params)
        token = _inside.set(True)
        try:
            result = original(self, operation_name, api_params)
        except Exception as error:
            _record(meta, error=error)
            raise
        finally:
            _inside.reset(token)
        if operation_name == "Converse":
            try:
                reply = {key: value for key, value in result.items() if key != "ResponseMetadata"}
            except Exception:
                reply = result
            _record(meta, _no_bytes(reply))
            return result
        try:
            if isinstance(result, dict) and result.get("stream") is not None:
                return dict(result, stream=_ConverseStream(result["stream"], meta))
        except Exception as error:
            _warn("bedrock", "General Context could not follow a Bedrock stream: %s", error)
        return result

    _make_api_call._generalcontext = True
    BaseClient._make_api_call = _make_api_call


class _ConverseStream(_SyncStream):
    """Passes Bedrock ConverseStream events through and records them assembled."""

    def _keep(self, chunk):
        try:
            self._chunks.append(_no_bytes(chunk))
        except Exception:
            pass
        return chunk


def _assemble_converse(events):
    """ConverseStream events as the response Converse would have returned."""
    role, blocks, stop, usage, metrics = "assistant", {}, None, None, None
    for event in events:
        if not isinstance(event, dict):
            continue
        if "messageStart" in event:
            role = (event["messageStart"] or {}).get("role") or role
        elif "contentBlockStart" in event:
            start = event["contentBlockStart"]
            tool = (start.get("start") or {}).get("toolUse")
            if tool:
                blocks[start.get("contentBlockIndex", 0)] = {"toolUse": dict(tool, _input="")}
        elif "contentBlockDelta" in event:
            body = event["contentBlockDelta"]
            index, delta = body.get("contentBlockIndex", 0), body.get("delta") or {}
            if "text" in delta:
                block = blocks.setdefault(index, {"text": ""})
                block["text"] = block.get("text", "") + delta["text"]
            elif "toolUse" in delta:
                block = blocks.setdefault(index, {"toolUse": {"_input": ""}})
                block["toolUse"]["_input"] += (delta["toolUse"] or {}).get("input") or ""
            elif "reasoningContent" in delta:
                block = blocks.setdefault(index, {"reasoningContent": {"reasoningText": {"text": ""}}})
                text = block["reasoningContent"].setdefault("reasoningText", {"text": ""})
                text["text"] = text.get("text", "") + ((delta["reasoningContent"] or {}).get("text") or "")
        elif "messageStop" in event:
            stop = (event["messageStop"] or {}).get("stopReason") or stop
        elif "metadata" in event:
            usage = (event["metadata"] or {}).get("usage") or usage
            metrics = (event["metadata"] or {}).get("metrics") or metrics
    for block in blocks.values():
        tool = block.get("toolUse")
        if tool is not None and "_input" in tool:
            raw = tool.pop("_input")
            try:
                tool["input"] = json.loads(raw) if raw else {}
            except ValueError:
                tool["input"] = raw
    response = {"output": {"message": {"role": role, "content": [blocks[index] for index in sorted(blocks)]}},
                "stopReason": stop}
    if usage:
        response["usage"] = usage
    if metrics:
        response["metrics"] = metrics
    return response


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
    ("email", re.compile(r"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63}){0,8}\.[A-Za-z]{2,24}")),
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


PLAIN_MAX_NODES = 200_000  # Past this many values a snapshot is cut, so a huge or shared structure stays cheap.


def _plain(value, depth=0, _path=None, _budget=None):
    """SDK objects (pydantic models) as plain JSON. Request bodies may hold anything; keep what serializes.

    Circular references become "<circular>" and NaN or Infinity become strings, so the snapshot
    is always strict JSON and always finishes in time linear in its size.
    """
    if _budget is None:
        _path, _budget = set(), [PLAIN_MAX_NODES]
    _budget[0] -= 1
    if depth > 40 or _budget[0] < 0:
        return None
    if inspect.isclass(value):  # e.g. response_format=SomeModel
        return f"{value.__module__}.{value.__qualname__}"
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (dict, list, tuple)):
        if id(value) in _path:
            return "<circular>"
        _path.add(id(value))
        try:
            if isinstance(value, dict):
                return {str(key): _plain(item, depth + 1, _path, _budget) for key, item in value.items()
                        if not str(key).startswith(("extra_headers", "timeout"))}
            return [_plain(item, depth + 1, _path, _budget) for item in value]
        finally:
            _path.discard(id(value))
    return str(value)
