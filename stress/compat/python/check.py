"""Layer 1 compatibility check for one installed provider package. Prints one JSON line.

    python check.py openai|anthropic|google-genai|boto3

Runs every call style twice against the local fake provider: first without
generalcontext.initialize(), then with it. For each style it asserts:
  - the agent's result is identical in both phases,
  - the style is recorded exactly once (and nothing is recorded before initialize),
  - the recorded request and response are complete (messages, tools, reply text, tool calls, usage),
  - provider errors (400, 429, 500, timeout) reach the agent unchanged and are recorded.
A style the installed version does not offer is reported as "n/a".
"""

import asyncio
import inspect
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fake  # noqa: E402

os.environ.pop("GENERALCONTEXT_DISABLED", None)
SERVER, URL = fake.start()
USER = "MARKER-user-asks-about-unit-4B"
SYSTEM = "MARKER-system-be-brief"


class NA(Exception):
    """This call style does not exist in the installed version."""


def need(condition, why):
    if not condition:
        raise NA(why)


def plain(value):
    """The agent-visible result as JSON-comparable data, without per-request noise (dates, headers)."""
    if hasattr(value, "model_dump"):
        try:
            value = value.model_dump(mode="json")
        except TypeError:
            value = value.model_dump()
    elif hasattr(value, "to_dict") and not isinstance(value, dict):
        value = value.to_dict()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()
                if k not in ("sdk_http_response", "ResponseMetadata", "headers", "_request_id")}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.decode("latin-1")
    return repr(type(value))


def error_signature(error):
    return {"type": f"{type(error).__module__}.{type(error).__name__}", "text": str(error),
            "status": getattr(error, "status_code", None) or getattr(error, "code", None)}


# ---- Call styles per package ---------------------------------------------------------------
# Each style: (name, run(phase_clients) -> result, expectations). Expectations name what the
# recorded call must contain: tools sent/returned, structured (JSON) text, streamed.

STRICT_PARAMS = {"type": "object", "properties": {"unit": {"type": "string"}}, "required": ["unit"], "additionalProperties": False}
# strict, so openai's stream helpers (which auto-parse tool arguments) accept it.
TOOLS_OPENAI = [{"type": "function", "function": {"name": "get_unit", "description": "Look up a unit",
                                                  "parameters": STRICT_PARAMS, "strict": True}}]
TOOLS_RESPONSES = [{"type": "function", "name": "get_unit", "description": "Look up a unit",
                    "parameters": {"type": "object", "properties": {"unit": {"type": "string"}}}}]
TOOLS_ANTHROPIC = [{"name": "get_unit", "description": "Look up a unit",
                    "input_schema": {"type": "object", "properties": {"unit": {"type": "string"}}}}]


def accepts(method, name):
    """Whether the installed SDK method takes keyword `name` (old releases predate tools, stream_options)."""
    try:
        return name in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return True


TOOLS = {"on": True}  # Whether this package version accepts tools; recorded calls must then carry them.


def openai_styles():
    import openai
    import pydantic

    class Unit(pydantic.BaseModel):
        unit: str
        open: bool

    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}]
    sync = openai.OpenAI(api_key="sk-fake", base_url=URL + "/v1", max_retries=0, timeout=1.0)
    TOOLS["on"] = accepts(type(sync.chat.completions).create, "tools")
    chat = dict(model="gpt-fake", messages=messages, **({"tools": TOOLS_OPENAI} if TOOLS["on"] else {}))
    usage = {"stream_options": {"include_usage": True}} if accepts(type(sync.chat.completions).create, "stream_options") else {}
    asy = lambda: openai.AsyncOpenAI(api_key="sk-fake", base_url=URL + "/v1", max_retries=0, timeout=1.0)

    def helper(client, name):
        """chat.completions.<name>, or beta.chat.completions.<name> in older versions."""
        method = getattr(client.chat.completions, name, None)
        if method is None:
            beta = getattr(getattr(getattr(client, "beta", None), "chat", None), "completions", None)
            method = getattr(beta, name, None)
        need(method is not None, f"no chat.completions.{name}")
        return method

    def stream_sync():
        return [plain(c) for c in sync.chat.completions.create(stream=True, **usage, **chat)]

    async def stream_async():
        client = asy()
        return [plain(c) async for c in await client.chat.completions.create(stream=True, **usage, **chat)]

    def helper_stream_sync():
        with helper(sync, "stream")(**usage, **chat) as stream:
            events = [e.type for e in stream]
            return events, plain(stream.get_final_completion())

    async def helper_stream_async():
        client = asy()
        async with helper(client, "stream")(**usage, **chat) as stream:
            events = [e.type async for e in stream]
            return events, plain(await stream.get_final_completion())

    def parse_sync():
        reply = helper(sync, "parse")(model="gpt-fake", messages=messages, response_format=Unit)
        return plain(reply), plain(reply.choices[0].message.parsed)

    async def parse_async():
        reply = await helper(asy(), "parse")(model="gpt-fake", messages=messages, response_format=Unit)
        return plain(reply), plain(reply.choices[0].message.parsed)

    def raw_sync():
        # A fresh client: with_raw_response is a cached_property that binds create when first read.
        client = openai.OpenAI(api_key="sk-fake", base_url=URL + "/v1", max_retries=0, timeout=1.0)
        raw = client.chat.completions.with_raw_response.create(**chat)
        return raw.status_code, plain(raw.parse())

    def responses(client):
        need(hasattr(client, "responses"), "no responses API")
        return client.responses

    rsp = dict(model="gpt-fake", input=USER, instructions=SYSTEM, tools=TOOLS_RESPONSES)

    def responses_stream():
        return [plain(e) for e in responses(sync).create(stream=True, **rsp)]

    async def responses_async():
        return plain(await responses(asy()).create(**rsp))

    async def responses_stream_async():
        return [plain(e) async for e in await responses(asy()).create(stream=True, **rsp)]

    def responses_helper():
        method = getattr(responses(sync), "stream", None)
        need(method is not None, "no responses.stream")
        with method(**rsp) as stream:
            events = [e.type for e in stream]
            return events, plain(stream.get_final_response())

    def responses_parse():
        method = getattr(responses(sync), "parse", None)
        need(method is not None, "no responses.parse")
        reply = method(model="gpt-fake", input=USER, instructions=SYSTEM, text_format=Unit)
        return plain(reply), plain(reply.output_parsed)

    def azure():
        return openai.AzureOpenAI(api_key="sk-fake", api_version="2024-10-21", azure_endpoint=URL, max_retries=0, timeout=1.0)

    async def azure_async():
        client = openai.AsyncAzureOpenAI(api_key="sk-fake", api_version="2024-10-21", azure_endpoint=URL, max_retries=0, timeout=1.0)
        return plain(await client.chat.completions.create(**chat))

    styles = [
        ("chat.create", lambda: plain(sync.chat.completions.create(**chat)), {"tools": True}),
        ("chat.create async", lambda: asyncio.run(_coro(lambda: asy().chat.completions.create(**chat))), {"tools": True}),
        ("chat.create stream", stream_sync, {"tools": True}),
        ("chat.create stream async", lambda: asyncio.run(stream_async()), {"tools": True}),
        ("chat.stream helper", helper_stream_sync, {"tools": True}),
        ("chat.stream helper async", lambda: asyncio.run(helper_stream_async()), {"tools": True}),
        ("chat.parse", parse_sync, {"structured": True}),
        ("chat.parse async", lambda: asyncio.run(parse_async()), {"structured": True}),
        ("chat.with_raw_response", raw_sync, {"tools": True}),
        ("responses.create", lambda: plain(responses(sync).create(**rsp)), {"tools": True}),
        ("responses.create async", lambda: asyncio.run(responses_async()), {"tools": True}),
        ("responses.create stream", responses_stream, {"tools": True}),
        ("responses.create stream async", lambda: asyncio.run(responses_stream_async()), {"tools": True}),
        ("responses.stream helper", responses_helper, {"tools": True}),
        ("responses.parse", responses_parse, {"structured": True}),
        ("azure chat.create", lambda: plain(azure().chat.completions.create(**chat)), {"tools": True}),
        ("azure chat.create async", lambda: asyncio.run(azure_async()), {"tools": True}),
    ]

    def failing(model):
        return lambda: sync.chat.completions.create(model=model, messages=messages)

    def failing_async(model):
        return lambda: asyncio.run(_coro(lambda: asy().chat.completions.create(model=model, messages=messages)))

    def failing_stream(model):
        return lambda: list(sync.chat.completions.create(model=model, messages=messages, stream=True))

    return styles, [("chat.create", failing), ("chat.create async", failing_async), ("chat.create stream", failing_stream)]


async def _coro(make):
    return plain(await make())


def anthropic_styles():
    import anthropic
    import pydantic

    class Unit(pydantic.BaseModel):
        unit: str
        open: bool

    sync = anthropic.Anthropic(api_key="sk-fake", base_url=URL, max_retries=0, timeout=1.0)
    TOOLS["on"] = accepts(type(sync.messages).create, "tools")
    req = dict(model="claude-fake", max_tokens=64, system=SYSTEM, messages=[{"role": "user", "content": USER}],
               **({"tools": TOOLS_ANTHROPIC} if TOOLS["on"] else {}))
    asy = lambda: anthropic.AsyncAnthropic(api_key="sk-fake", base_url=URL, max_retries=0, timeout=1.0)

    def stream_sync():
        return [plain(e) for e in sync.messages.create(stream=True, **req)]

    async def stream_async():
        return [plain(e) async for e in await asy().messages.create(stream=True, **req)]

    def helper_sync(owner):
        with owner.stream(**req) as stream:
            text = "".join(stream.text_stream)
            return text, plain(stream.get_final_message())

    async def helper_async():
        async with asy().messages.stream(**req) as stream:
            text = "".join([t async for t in stream.text_stream])
            return text, plain(await stream.get_final_message())

    def beta():
        owner = getattr(getattr(sync, "beta", None), "messages", None)
        need(owner is not None, "no beta.messages")
        return owner

    def parse(owner, label):
        method = getattr(owner, "parse", None)
        need(method is not None, f"no {label}.parse")
        reply = method(model="claude-fake", max_tokens=64, messages=[{"role": "user", "content": USER}],
                       output_format=Unit)
        return plain(reply), plain(getattr(reply, "parsed_output", None))

    def raw_sync():
        client = anthropic.Anthropic(api_key="sk-fake", base_url=URL, max_retries=0, timeout=1.0)
        raw = client.messages.with_raw_response.create(**req)
        return raw.status_code, plain(raw.parse())

    def bedrock():
        need(hasattr(anthropic, "AnthropicBedrock"), "no AnthropicBedrock")
        try:
            import boto3  # noqa: F401
        except ImportError:
            raise NA("boto3 not installed")
        return anthropic.AnthropicBedrock(aws_access_key="fake", aws_secret_key="fake", aws_region="us-east-1",
                                          base_url=URL, max_retries=0, timeout=1.0)

    def vertex():
        need(hasattr(anthropic, "AnthropicVertex"), "no AnthropicVertex")
        return anthropic.AnthropicVertex(region="us-east5", project_id="p", access_token="fake", base_url=URL,
                                         max_retries=0, timeout=1.0)

    def bedrock_stream():
        with bedrock().messages.stream(**req) as stream:
            text = "".join(stream.text_stream)
            return text, plain(stream.get_final_message())

    def vertex_stream():
        with vertex().messages.stream(**req) as stream:
            text = "".join(stream.text_stream)
            return text, plain(stream.get_final_message())

    styles = [
        ("messages.create", lambda: plain(sync.messages.create(**req)), {"tools": True}),
        ("messages.create async", lambda: asyncio.run(_coro(lambda: asy().messages.create(**req))), {"tools": True}),
        ("messages.create stream", stream_sync, {"tools": True}),
        ("messages.create stream async", lambda: asyncio.run(stream_async()), {"tools": True}),
        ("messages.stream helper", lambda: helper_sync(sync.messages), {"tools": True}),
        ("messages.stream helper async", lambda: asyncio.run(helper_async()), {"tools": True}),
        ("messages.parse", lambda: parse(sync.messages, "messages"), {"structured": True}),
        ("messages.with_raw_response", raw_sync, {"tools": True}),
        ("beta.messages.create", lambda: plain(beta().create(**req)), {"tools": True}),
        ("beta.messages.stream helper", lambda: helper_sync(beta()), {"tools": True}),
        ("beta.messages.parse", lambda: parse(beta(), "beta.messages"), {"structured": True}),
        ("bedrock messages.create", lambda: plain(bedrock().messages.create(**req)), {"tools": True}),
        ("bedrock messages.stream helper", bedrock_stream, {"tools": True}),
        ("vertex messages.create", lambda: plain(vertex().messages.create(**req)), {"tools": True}),
        ("vertex messages.stream helper", vertex_stream, {"tools": True}),
    ]

    def failing(model):
        return lambda: sync.messages.create(**dict(req, model=model, timeout=1.0))

    def failing_async(model):
        return lambda: asyncio.run(_coro(lambda: asy().messages.create(**dict(req, model=model, timeout=1.0))))

    def failing_stream(model):
        def run():
            with sync.messages.stream(**dict(req, model=model, timeout=1.0)) as stream:
                return list(stream)
        return run

    return styles, [("messages.create", failing), ("messages.create async", failing_async),
                    ("messages.stream helper", failing_stream)]


def genai_styles():
    from google import genai
    from google.genai import types

    def client(**extra):
        options = {"base_url": URL, "api_version": "v1beta", "timeout": 1000}
        return genai.Client(api_key="fake", http_options=types.HttpOptions(**options), **extra)

    sync = client()
    tools = [types.Tool(function_declarations=[types.FunctionDeclaration(
        name="get_unit", description="Look up a unit",
        parameters=types.Schema(type="OBJECT", properties={"unit": types.Schema(type="STRING")}))])]
    # Function declarations (not Python callables), so automatic function calling never runs.
    config = types.GenerateContentConfig(system_instruction=SYSTEM, tools=tools)
    req = dict(model="gemini-fake", contents=USER, config=config)

    async def stream_async():
        return [plain(c) async for c in await client().aio.models.generate_content_stream(**req)]

    def vertex():
        try:
            from google.oauth2.credentials import Credentials
        except ImportError:
            raise NA("google-auth not installed")
        options = types.HttpOptions(base_url=URL, api_version="v1beta1", timeout=1000)
        return genai.Client(vertexai=True, project="p", location="us-central1",
                            credentials=Credentials(token="fake"), http_options=options)

    def chat():
        need(hasattr(sync, "chats"), "no client.chats")
        session = sync.chats.create(model="gemini-fake", config=config)
        return plain(session.send_message(USER))

    styles = [
        ("models.generate_content", lambda: plain(sync.models.generate_content(**req)), {"tools": True}),
        ("models.generate_content async", lambda: asyncio.run(_coro(lambda: client().aio.models.generate_content(**req))), {"tools": True}),
        ("models.generate_content_stream", lambda: [plain(c) for c in sync.models.generate_content_stream(**req)], {"tools": True}),
        ("models.generate_content_stream async", lambda: asyncio.run(stream_async()), {"tools": True}),
        ("chats.send_message", chat, {"tools": True}),
        ("vertex models.generate_content", lambda: (lambda c: plain(c.models.generate_content(**req)))(vertex()), {"tools": True}),
        ("vertex models.generate_content_stream", lambda: (lambda v: [plain(c) for c in v.models.generate_content_stream(**req)])(vertex()), {"tools": True}),
    ]

    def failing(model):
        return lambda: sync.models.generate_content(**dict(req, model=model))

    def failing_async(model):
        return lambda: asyncio.run(_coro(lambda: client().aio.models.generate_content(**dict(req, model=model))))

    def failing_stream(model):
        return lambda: list(sync.models.generate_content_stream(**dict(req, model=model)))

    return styles, [("models.generate_content", failing), ("models.generate_content async", failing_async),
                    ("models.generate_content_stream", failing_stream)]


def boto3_styles():
    import boto3
    from botocore.config import Config

    client = boto3.client("bedrock-runtime", region_name="us-east-1", endpoint_url=URL,
                          aws_access_key_id="fake", aws_secret_access_key="fake",
                          config=Config(read_timeout=1, connect_timeout=1, retries={"total_max_attempts": 1}))
    tool_config = {"tools": [{"toolSpec": {"name": "get_unit", "description": "Look up a unit",
                                           "inputSchema": {"json": {"type": "object", "properties": {"unit": {"type": "string"}}}}}}]}
    req = dict(modelId="anthropic.claude-fake", system=[{"text": SYSTEM}],
               messages=[{"role": "user", "content": [{"text": USER}]}], toolConfig=tool_config)

    def stream():
        result = client.converse_stream(**req)
        return [plain(e) for e in result["stream"]]

    styles = [
        ("converse", lambda: plain(client.converse(**req)), {"tools": True}),
        ("converse_stream", stream, {"tools": True}),
    ]

    def failing(model):
        return lambda: client.converse(**dict(req, modelId=model))

    def failing_stream(model):
        return lambda: list(client.converse_stream(**dict(req, modelId=model))["stream"])

    return styles, [("converse", failing), ("converse_stream", failing_stream)]


PACKAGES = {"openai": openai_styles, "anthropic": anthropic_styles, "google-genai": genai_styles, "boto3": boto3_styles}


# ---- Checks -------------------------------------------------------------------------------

def run(fn):
    try:
        return "ok", fn()
    except NA as reason:
        return "na", str(reason)
    except Exception as error:
        return "error", error


def drain():
    import generalcontext
    if generalcontext._config["initialized"]:
        generalcontext.flush(5)
    time.sleep(0.05)
    with fake.Handler.lock:
        calls = list(fake.Handler.calls)
        fake.Handler.calls.clear()
    return calls


def complete(call, expect, session):
    """Problems with one recorded successful call (empty when complete)."""
    problems = []
    expect = dict(expect, tools=expect.get("tools") and TOOLS["on"])
    request, response = json.dumps(call.get("request")), json.dumps(call.get("response"))
    if call.get("error"):
        problems.append(f"error recorded on success: {call['error'][:120]}")
    if USER not in request:
        problems.append("request lacks the user message")
    if expect.get("tools") and "get_unit" not in request:
        problems.append("request lacks the tools")
    if "-fake" not in request:
        problems.append("request lacks the model")
    if call.get("response") is None:
        problems.append("no response")
    else:
        if expect.get("structured"):
            if '\\"unit\\": \\"4B\\"' not in response and '"unit": "4B"' not in response:
                problems.append("response lacks the reply text")
        elif "Unit 4B" not in response:
            problems.append("response lacks the reply text")
        if expect.get("tools") and "get_unit" not in response:
            problems.append("response lacks the tool call")
        if "4200" not in response or "42" not in response.replace("4200", ""):
            problems.append("response lacks usage")
    if (call.get("conversation") or {}).get("id") != session:
        problems.append("wrong or missing session")
    return problems


ERROR_MODELS = {"400": "err-400", "429": "err-429", "500": "err-500", "timeout": "err-timeout"}


def main(package):
    import generalcontext
    styles, error_styles = PACKAGES[package]()
    error_cases = [(f"{name} {label}", make(model)) for name, make in error_styles for label, model in ERROR_MODELS.items()]

    # Phase 1: no initialize().
    baseline = {name: run(fn) for name, fn, _ in styles}
    baseline_errors = {name: run(fn) for name, fn in error_cases}
    stray = drain()

    # Phase 2: initialize() patches the installed classes, including clients created above.
    generalcontext.initialize(api_key="gck_" + "f" * 48, base_url=URL, app_name="compat", disable_batch=True)
    results = {}
    for name, fn, expect in styles:
        session = "s-" + name
        with generalcontext.session(session):
            status, value = run(fn)
        calls = drain()
        base_status, base_value = baseline[name]
        if base_status == "na":
            results[name] = {"status": "n/a", "detail": base_value}
            continue
        problems = []
        if base_status == "error":
            # The package cannot do this against the fake even without us: not an SDK result.
            results[name] = {"status": "env", "detail": f"fails without the SDK: {type(base_value).__name__}: {str(base_value)[:200]}"
                             + (f"; with the SDK: {status}" if status != "error" else "")}
            continue
        if status != base_status:
            problems.append(f"with the SDK: {status}: {value if status != 'error' else repr(value)[:300]}")
        elif status == "ok" and json.dumps(value, sort_keys=True, default=str) != json.dumps(base_value, sort_keys=True, default=str):
            problems.append("agent result differs with initialize()")
        if len(calls) != 1:
            problems.append(f"recorded {len(calls)} times")
        for call in calls[:1]:
            problems += complete(call, expect, session)
        results[name] = {"status": "pass" if not problems else "fail", "detail": "; ".join(problems)}

    for name, fn in error_cases:
        session = "s-" + name
        with generalcontext.session(session):
            status, value = run(fn)
        calls = drain()
        base_status, base_value = baseline_errors[name]
        problems = []
        if base_status == "na":
            results[name] = {"status": "n/a", "detail": base_value}
            continue
        if base_status != "error":
            problems.append(f"fake did not fail without the SDK ({base_status})")
        elif status != "error":
            problems.append(f"with the SDK the agent got {status}, not the error")
        elif error_signature(value) != error_signature(base_value):
            problems.append(f"error changed: {error_signature(value)} vs {error_signature(base_value)}")
        if len(calls) != 1:
            problems.append(f"recorded {len(calls)} times")
        elif not calls[0].get("error") or type(base_value).__name__ not in calls[0]["error"]:
            problems.append(f"recorded error missing or wrong: {calls[0].get('error')!r:.120}")
        elif USER not in json.dumps(calls[0].get("request")):
            problems.append("request of failed call incomplete")
        results[name] = {"status": "pass" if not problems else "fail", "detail": "; ".join(problems)}

    if stray:
        results["no recording before initialize"] = {"status": "fail", "detail": f"{len(stray)} calls recorded"}
    return results


if __name__ == "__main__":
    name = sys.argv[1]
    try:
        out = {"package": name, "results": main(name)}
    except Exception:
        out = {"package": name, "crash": traceback.format_exc()[-2000:]}
    print("RESULT " + json.dumps(out))
    sys.stdout.flush()
    os._exit(0)  # Do not wait on the fake server's sleeping timeout threads.
