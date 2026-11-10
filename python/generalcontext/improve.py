# Copyright 2026 General Context Labs, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Propose better instructions for your agent from its benchmark.

    generalcontext improve benchmark.json --agent openai:gpt-6.1-terra --rounds 2 --trials 2

It never edits your code. It runs the reference agent on the benchmark's train tasks with your
current instructions, asks a proposer model for the smallest general change that fixes the
failures, and keeps a candidate only if the pass rate rises. Held-out tasks are never shown to
the proposer: they are run once at the end, with the current and the best instructions, so the
comparison is honest. The result is a proposal for a person to review: instructions.md,
instructions.diff, report.json and runs.json in --out.
"""

from collections import Counter
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import re
import sys

from . import evals

DEFAULT_PROPOSER = "openai:gpt-6.1-sol@high"

PROPOSER = """You improve the instructions (system prompt) of an AI agent. You get its current instructions
and runs where it failed: what it should have done, the policies, the criteria it failed with the
grader's reasons, and part of the conversation.

Make the smallest general change that fixes these failures. Keep everything else verbatim. Never copy
names, numbers, addresses or other facts from a specific conversation into the instructions: write
rules that hold for every conversation.

Answer with JSON only: {"instructions": "the full revised instructions", "rationale": "one or two sentences"}"""


def base_instructions(tasks, path=None):
    """--instructions FILE, else the most common non-empty instructions recorded in the tasks."""
    if path:
        return Path(path).read_text()
    found = Counter(text for text in ((task.get("agent") or {}).get("instructions") for task in tasks)
                    if isinstance(text, str) and text.strip())
    if not found:
        raise ValueError("No instructions found in the benchmark. Pass --instructions FILE.")
    return found.most_common(1)[0][0]


def evaluate(tasks, agent_model, instructions, person_model, judge_model, trials, label):
    """Run every task `trials` times. Returns (summary, results) where results hold task and run."""
    results = []
    for task in tasks:
        for trial in range(1, trials + 1):
            try:
                result = evals.run_task(task, evals.reference_agent(agent_model, task, instructions), person_model, judge_model)
            except Exception as failure:  # The person or judge failed: not a result.
                print(f"SKIP  {task.get('slug') or task['id']}  #{trial}  harness error: {type(failure).__name__}: {failure}"[:400], flush=True)
                continue
            results.append({"task": task, "trial": trial, "label": label, "result": result})
            # One line per trial, so a long run shows progress (and CI logs show where time went).
            print(f"{'PASS' if result['passed'] else 'FAIL'}  {task.get('slug') or task['id']}  #{trial}  {label}  {evals.note(result)}"[:300], flush=True)
    return summarize(results), results


def summarize(results):
    if not results:
        return {"runs": 0, "pass_rate": 0.0, "reward": 0.0}
    return {"runs": len(results),
            "pass_rate": round(sum(r["result"]["passed"] for r in results) / len(results), 4),
            "reward": round(sum(r["result"]["reward"] for r in results) / len(results), 4)}


def better(candidate, current):
    """Higher pass rate wins; a tie is broken by mean reward."""
    return (candidate["pass_rate"], candidate["reward"]) > (current["pass_rate"], current["reward"]) and \
        candidate["pass_rate"] >= current["pass_rate"]


def failure_brief(results, limit=12):
    """What the proposer sees about failed runs. Only optimize-set runs are ever passed in."""
    parts = []
    for item in [r for r in results if not r["result"]["passed"]][:limit]:
        task, result = item["task"], item["result"]
        failed = [c for c in result["criteria"] if not c["passed"]]
        lines = [f"Task: {task.get('title') or task.get('slug') or task['id']}",
                 "Expected: " + (task.get("expected") or "(not given)")]
        if task.get("policies"):
            lines.append("Policies:\n" + "\n".join(f"- {p.get('title')}: {p.get('text')}" for p in task["policies"]))
        if result.get("error"):
            lines.append("Error: " + result["error"])
        lines.append("Failed criteria:\n" + ("\n".join(f"- {c['claim']} ({c['reason']})" for c in failed) or "- (none graded)"))
        lines.append("Transcript:\n" + "\n".join(f"{t['from']}: {t['text'][:300]}" for t in result["transcript"][:8]))
        parts.append("\n".join(lines))
    return "\n\n---\n\n".join(parts)


def propose(proposer_model, instructions, results):
    content = "Current instructions:\n<<<\n" + instructions + "\n>>>\n\nFailed runs:\n\n" + failure_brief(results)
    reply = proposer_model(PROPOSER, [{"role": "user", "content": content}], [])
    found = re.search(r"\{.*\}", reply.get("text") or "", re.S)
    data = json.loads(found.group(0)) if found else {}
    text = data.get("instructions")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("the proposer did not return instructions")
    return text, str(data.get("rationale") or "")


def improve(tasks, agent_spec, agent_model, proposer_model, person_model, judge_model, instructions,
            rounds=2, trials=2, optimize_on="train", log=print):
    """Returns the report, the best instructions, and every result for runs.json."""
    optimize = tasks if optimize_on == "all" else [t for t in tasks if t.get("split") == "train"]
    held_out = [] if optimize_on == "all" else [t for t in tasks if t.get("split") == "held_out"]
    if optimize_on == "all":
        log("Warning: optimizing on every task, so held-out scores are no longer an honest check.")
    if not optimize:
        raise ValueError("No train tasks to optimize on. Use --optimize-on all to use every task.")
    current_label, proposed_label = f"{agent_spec} (current instructions)", f"{agent_spec} (proposed instructions)"
    baseline, results = evaluate(optimize, agent_model, instructions, person_model, judge_model, trials, current_label)
    report = {"optimize_on": optimize_on, "optimize_tasks": len(optimize), "held_out_tasks": len(held_out),
              "baseline": baseline, "rounds": [], "held_out": None}
    log(f"Baseline on {len(optimize)} optimize tasks: {baseline['pass_rate']:.0%} pass, reward {baseline['reward']:.2f}")
    all_results = list(results)
    best, best_summary, best_results = instructions, baseline, results
    for number in range(1, rounds + 1):
        if best_summary["pass_rate"] >= 1:
            break
        try:
            candidate, rationale = propose(proposer_model, best, best_results)
        except Exception as failure:
            log(f"Round {number}: no proposal ({type(failure).__name__}: {failure})")
            report["rounds"].append({"round": number, "kept": False, "error": f"{type(failure).__name__}: {failure}"})
            break
        summary, results = evaluate(optimize, agent_model, candidate, person_model, judge_model, trials, proposed_label)
        kept = better(summary, best_summary)
        report["rounds"].append({"round": number, **summary, "rationale": rationale, "kept": kept})
        log(f"Round {number}: {summary['pass_rate']:.0%} pass, reward {summary['reward']:.2f}, {'kept' if kept else 'rejected'}")
        if not kept:
            break
        all_results.extend(results)
        best, best_summary, best_results = candidate, summary, results
    report["optimize"] = {"before": baseline, "after": best_summary}
    report["changed"] = best != instructions
    if held_out:
        before, results = evaluate(held_out, agent_model, instructions, person_model, judge_model, trials, current_label)
        all_results.extend(results)
        after = before
        if best != instructions:
            after, results = evaluate(held_out, agent_model, best, person_model, judge_model, trials, proposed_label)
            all_results.extend(results)
        report["held_out"] = {"before": before, "after": after}
    return report, best, all_results


def runs_body(results):
    at = datetime.now(timezone.utc).isoformat()
    return {"format": evals.RUNS_FORMAT, "runs": [
        {"task": r["task"]["id"], "agent": r["label"], "passed": r["result"]["passed"],
         "reward": round(r["result"]["reward"], 3), "at": at, "note": evals.note(r["result"])} for r in results]}


def write_outputs(out, base, best, report, results):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "instructions.md").write_text(best)
    diff = "".join(difflib.unified_diff(base.splitlines(True), best.splitlines(True),
                                        "instructions (current)", "instructions (proposed)"))
    (out / "instructions.diff").write_text(diff)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    body = runs_body(results)
    (out / "runs.json").write_text(json.dumps(body, indent=2))
    return body


def main(argv=None, prog="generalcontext improve"):
    import argparse
    parser = argparse.ArgumentParser(prog=prog, description="Propose better instructions for your agent from its benchmark.")
    parser.add_argument("benchmark", help="a generalcontext.benchmark/v1 file")
    parser.add_argument("--agent", required=True, help="openai:<model> or claude-code:<model>")
    parser.add_argument("--instructions", help="file with your agent's current instructions")
    parser.add_argument("--proposer", default=DEFAULT_PROPOSER, help="model that proposes new instructions")
    parser.add_argument("--person", default="openai:gpt-6.1-terra@none", help="model that plays the person")
    parser.add_argument("--judge", default="openai:gpt-6.1-sol@high", help="model that grades the rubric")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--optimize-on", choices=["train", "all"], default="train")
    parser.add_argument("--out", default="improve", help="directory for the proposal")
    parser.add_argument("--upload", action="store_true", help="send runs.json to General Context")
    args = parser.parse_args(argv)
    import os
    if args.agent.split(":", 1)[0] not in ("openai", "claude-code", "anthropic"):
        parser.error("--agent must be openai:<model> or claude-code:<model>")
    if args.upload and not os.environ.get("GENERALCONTEXT_API_KEY"):
        parser.error("--upload needs GENERALCONTEXT_API_KEY")
    try:
        tasks = evals.load_tasks(args.benchmark)
        base = base_instructions(tasks, args.instructions)
    except (OSError, ValueError) as failure:
        print(f"error: {failure}", file=sys.stderr)
        return 2
    try:
        report, best, results = improve(
            tasks, args.agent, evals.model(args.agent), evals.model(args.proposer), evals.model(args.person),
            evals.model(args.judge), base, rounds=args.rounds, trials=args.trials, optimize_on=args.optimize_on)
    except ValueError as failure:
        print(f"error: {failure}", file=sys.stderr)
        return 2
    body = write_outputs(args.out, base, best, report, results)
    before, after = report["optimize"]["before"], report["optimize"]["after"]
    print(f"Optimize set: {before['pass_rate']:.0%} -> {after['pass_rate']:.0%} pass")
    if report["held_out"]:
        print(f"Held out: {report['held_out']['before']['pass_rate']:.0%} -> {report['held_out']['after']['pass_rate']:.0%} pass")
    out = Path(args.out)
    print(f"Proposal: {out / 'instructions.diff'}" if report["changed"] else
          "No change beat the current instructions. Report: " + str(out / "report.json"))
    if args.upload:
        try:
            result = evals.upload(body)
        except Exception as failure:
            print(f"Upload failed: {type(failure).__name__}: {failure}", file=sys.stderr)
            return 1
        print(f"Uploaded {result.get('recorded', 0)} runs to General Context.")
    return 0
