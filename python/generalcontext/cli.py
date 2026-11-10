# Copyright 2026 General Context Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The `generalcontext` command.

    generalcontext verify [--session ID] [--json]
    generalcontext benchmark pull [--out benchmark.json]
    generalcontext evals run BENCHMARK --agent ... [--upload]

Uses GENERALCONTEXT_API_KEY and GENERALCONTEXT_BASE_URL. Exit 0 on success, 1 on a failed check,
2 on a usage or configuration error.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

COLLECT = "/api/platform-collect"


class ConfigError(Exception):
    """Missing key or an insecure base URL: exit 2."""


class CheckFailed(Exception):
    """The server said no: exit 1."""


def _settings():
    import generalcontext
    key = os.environ.get("GENERALCONTEXT_API_KEY")
    base = (os.environ.get("GENERALCONTEXT_BASE_URL") or generalcontext.DEFAULT_BASE_URL).rstrip("/")
    if not key:
        raise ConfigError("GENERALCONTEXT_API_KEY is not set. Create a key on the API keys page.")
    if not generalcontext._secure(base):
        raise ConfigError(f"GENERALCONTEXT_BASE_URL {base} is not https.")
    return key, base


def _get(path):
    import generalcontext
    key, base = _settings()
    request = urllib.request.Request(base + COLLECT + path, headers={
        "Authorization": "Bearer " + key, "Accept": "application/json"})
    try:
        with generalcontext._opener.open(request, timeout=60) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        try:
            message = json.loads(error.read() or b"{}").get("error")
        except Exception:
            message = None
        if error.code == 401:
            message = message or "General Context did not accept this key. Check GENERALCONTEXT_API_KEY."
        raise CheckFailed(message or f"General Context answered {error.code}.")
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise CheckFailed(f"Could not reach General Context at {base}: {getattr(error, 'reason', error)}")


SCOPES = {"ingest": "sends traffic", "evals": "runs the benchmark"}


def verify(args):
    path = "/status" + ("?" + urllib.parse.urlencode({"session": args.session}) if args.session else "")
    status = _get(path)
    session = status.get("session")
    if args.json:
        print(json.dumps(status))
        return 1 if args.session and not (session or {}).get("found") else 0
    key = status.get("key") or {}
    scopes = ", ".join(SCOPES.get(s, s) for s in key.get("scopes") or []) or "no scopes"
    print(f"Workspace {status.get('workspace')}: key {key.get('label') or '(unnamed)'} ({scopes}).")
    if status.get("last_call_at"):
        print(f"Last call arrived {status['last_call_at']} from {status.get('last_source') or 'unknown'}.")
    else:
        print("No calls have arrived yet.")
    if args.session:
        if session and session.get("found"):
            models = ", ".join(session.get("models") or []) or "no model"
            print(f"Conversation {args.session}: arrived, {session.get('calls', 0)} calls ({models}).")
        else:
            print(f"Conversation {args.session}: not found yet.")
            return 1
    return 0


def benchmark_pull(args):
    data = _get("/benchmark")
    from . import evals
    if not isinstance(data, dict) or data.get("format") != evals.BENCHMARK_FORMAT:
        raise CheckFailed("General Context did not return a benchmark.")
    Path(args.out).write_text(json.dumps(data, indent=2))
    tasks = data.get("tasks") or []
    splits = Counter(task.get("split") or "unsplit" for task in tasks)
    detail = ", ".join(f"{count} {name}" for name, count in sorted(splits.items()))
    print(f"Wrote {args.out}: {len(tasks)} tasks" + (f" ({detail})." if detail else "."))
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # evals run takes its own flags: hand them over untouched.
    if argv[:2] == ["evals", "run"]:
        from . import evals
        return _exit_code(lambda: evals.main(argv[2:], prog="generalcontext evals run"))
    parser = argparse.ArgumentParser(prog="generalcontext", description="General Context from the command line.")
    commands = parser.add_subparsers(dest="command", metavar="command")
    check = commands.add_parser("verify", help="check the key and that calls are arriving")
    check.add_argument("--session", help="also check that this conversation arrived")
    check.add_argument("--json", action="store_true", help="print the raw status as JSON")
    bench = commands.add_parser("benchmark", help="download the benchmark")
    bench_commands = bench.add_subparsers(dest="action", metavar="action")
    pull = bench_commands.add_parser("pull", help="download the ready tasks")
    pull.add_argument("--out", default="benchmark.json")
    run = commands.add_parser("evals", help="run the benchmark (generalcontext evals run --help)")
    run.add_argument("action", choices=["run"])
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit:
        return exit.code if isinstance(exit.code, int) else 2
    if args.command == "verify":
        handler = verify
    elif args.command == "benchmark" and args.action == "pull":
        handler = benchmark_pull
    else:
        parser.print_help(sys.stderr)
        return 2
    try:
        return handler(args)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except CheckFailed as error:
        print(str(error), file=sys.stderr)
        return 1


def _exit_code(run):
    try:
        code = run()
    except SystemExit as exit:
        code = exit.code
    return code if isinstance(code, int) else (0 if code is None else 2)


def console():
    sys.exit(main())


if __name__ == "__main__":
    console()
