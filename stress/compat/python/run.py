"""Run the layer 1 compatibility matrix for the Python SDK. One command:

    python3 stress/compat/python/run.py            # full matrix
    python3 stress/compat/python/run.py --quick    # latest version of each package, one Python

Needs `uv`. Virtualenvs go under $STRESS_SCRATCH (default $JCODE_SCRATCH_DIR or a temp dir),
never in the repository. At most --jobs (default 4) environments run at once. Writes
results-python.json and matrix-python.md next to this file, and exits 1 on any failure.
No network calls except package downloads: providers and the collector are a local fake.
"""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

PYTHONS = ["3.9", "3.10", "3.11", "3.12", "3.13"]
# Oldest supported, two in the middle, latest (checked on PyPI 2026-11-09). Each entry:
# version -> lowest Python it installs on. Extra pins keep an old release installable the way
# its users run it (old openai/anthropic break with httpx 0.28, which dropped `proxies`).
MATRIX = {
    "openai": {"1.0.0": ("3.9", ["httpx<0.28"]), "1.66.0": ("3.9", ["httpx<0.28"]),
               "2.20.0": ("3.9", []), "3.48.0": ("3.10", [])},
    "anthropic": {"0.20.0": ("3.9", ["httpx<0.28"]), "0.60.0": ("3.9", []),
                  "1.8.0": ("3.10", []), "2.7.0": ("3.10", [])},
    "google-genai": {"1.0.0": ("3.9", []), "1.30.0": ("3.9", []),
                     "2.15.0": ("3.10", []), "3.0.0": ("3.11", [])},
    "boto3": {"1.34.116": ("3.9", []), "1.38.46": ("3.9", []),
              "1.42.97": ("3.9", []), "1.43.127": ("3.10", [])},
}
EXTRAS = {"openai": ["pydantic"], "anthropic": ["pydantic", "boto3", "botocore"], "google-genai": ["google-auth"],
          "boto3": []}


def key(version):
    return tuple(int(part) for part in version.split("."))


def scratch():
    root = os.environ.get("STRESS_SCRATCH") or os.environ.get("JCODE_SCRATCH_DIR") or tempfile.gettempdir()
    path = os.path.join(root, "gc-compat-python")
    os.makedirs(path, exist_ok=True)
    return path


def cell(package, version, python, pins):
    env_dir = os.path.join(scratch(), f"{package}-{version}-py{python}")
    log = []

    def sh(*cmd, timeout=600):
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              env=dict(os.environ, UV_CACHE_DIR=os.path.join(scratch(), "uv-cache")))
        log.append(done.stdout[-3000:] + done.stderr[-3000:])
        return done

    if not os.path.exists(os.path.join(env_dir, "bin", "python")):
        if sh("uv", "venv", "-q", "--python", python, env_dir).returncode:
            return {"status": "install", "detail": log[-1][-500:]}
    py = os.path.join(env_dir, "bin", "python")
    install = sh("uv", "pip", "install", "-q", "--python", py, f"{package}=={version}", *pins,
                 *EXTRAS[package], "-e", os.path.join(REPO, "python"))
    if install.returncode:
        return {"status": "install", "detail": install.stderr[-800:]}
    try:
        done = sh(py, os.path.join(HERE, "check.py"), package, timeout=600)
    except subprocess.TimeoutExpired:
        return {"status": "crash", "detail": "check.py timed out"}
    line = next((l for l in done.stdout.splitlines() if l.startswith("RESULT ")), None)
    if line is None:
        return {"status": "crash", "detail": (done.stdout + done.stderr)[-1500:]}
    out = json.loads(line[7:])
    if "crash" in out:
        return {"status": "crash", "detail": out["crash"]}
    return {"status": "ran", "results": out["results"]}


def summary(outcome):
    if outcome["status"] != "ran":
        return outcome["status"], []
    failed = [name for name, r in outcome["results"].items() if r["status"] == "fail"]
    return ("fail" if failed else "pass"), failed


def markdown(rows):
    lines = ["| Package | Version | " + " | ".join(f"Python {p}" for p in PYTHONS) + " |",
             "| - | - | " + " | ".join("-" for _ in PYTHONS) + " |"]
    notes = []
    for package, versions in MATRIX.items():
        for version in sorted(versions, key=key):
            cells = []
            for python in PYTHONS:
                outcome = rows.get((package, version, python))
                if outcome is None:
                    cells.append("n/a")
                    continue
                status, failed = summary(outcome)
                if status == "pass":
                    results = outcome["results"]
                    passed = sum(r["status"] == "pass" for r in results.values())
                    na = sum(r["status"] in ("n/a", "env") for r in results.values())
                    cells.append(f"pass ({passed}" + (f", {na} n/a" if na else "") + ")")
                elif status == "fail":
                    cells.append("FAIL: " + ", ".join(failed))
                else:
                    cells.append(status.upper())
                    notes.append(f"- {package} {version} on Python {python}: {status}: {outcome['detail'][:300]}")
            lines.append(f"| {package} | {version} | " + " | ".join(cells) + " |")
    return "\n".join(lines + ([""] + notes if notes else [])) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--python", action="append", help="limit to these Pythons")
    parser.add_argument("--package", action="append", help="limit to these packages")
    args = parser.parse_args()
    pythons = args.python or PYTHONS
    work = []
    for package, versions in MATRIX.items():
        if args.package and package not in args.package:
            continue
        chosen = [max(versions, key=key)] if args.quick else sorted(versions, key=key)
        for version in chosen:
            lowest, pins = versions[version]
            for python in ([pythons[-1]] if args.quick else pythons):
                if key(python) >= key(lowest):
                    work.append((package, version, python, pins))
    rows = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.jobs, 4)) as pool:
        futures = {pool.submit(cell, *item): item[:3] for item in work}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                rows[name] = future.result()
            except Exception as error:
                rows[name] = {"status": "crash", "detail": repr(error)}
            status, failed = summary(rows[name])
            print(f"{name[0]:>13} {name[1]:>9} py{name[2]:<5} {status}" + (f": {', '.join(failed)}" if failed else ""), flush=True)
    with open(os.path.join(HERE, "results-python.json"), "w") as out:
        json.dump([{"package": p, "version": v, "python": py, **o} for (p, v, py), o in sorted(rows.items())], out, indent=1)
    table = markdown(rows)
    with open(os.path.join(HERE, "matrix-python.md"), "w") as out:
        out.write(table)
    print(table)
    bad = [name for name, outcome in rows.items() if summary(outcome)[0] != "pass"]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
