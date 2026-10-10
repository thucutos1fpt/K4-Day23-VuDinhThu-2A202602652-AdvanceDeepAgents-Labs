"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
from collections import Counter  # noqa: F401
from pathlib import Path

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^\w]+", "-", (topic or "").strip().lower(), flags=re.UNICODE)
    slug = slug.strip("-_")[:60].strip("-_")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return f"""Research this topic: {topic}

Produce a sourced survey report. Use this required structure in the report body:
# <title>
## TL;DR
## Background
## <thematic sections that compare approaches>
## Trends and open problems

Do not write ## References yourself: the finalizer creates it. Complete the entire workflow in your system
instructions, including delegated research, sources.json, citation finalization, validation, and spot checks."""


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    counts = Counter()
    subagent_calls = 0
    input_tokens = 0
    output_tokens = 0
    for message in messages or []:
        tool_calls = getattr(message, "tool_calls", None)
        usage = getattr(message, "usage_metadata", None)
        if isinstance(message, dict):
            tool_calls = tool_calls if tool_calls is not None else message.get("tool_calls", [])
            usage = usage if usage is not None else message.get("usage_metadata", {})
        for call in tool_calls or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            if name:
                counts[name] += 1
                subagent_calls += name == "task"
        usage = usage or {}
        if isinstance(usage, dict):
            input_tokens += int(usage.get("input_tokens") or 0)
            output_tokens += int(usage.get("output_tokens") or 0)
    return {
        "model": str(model_name),
        "elapsed_s": round(elapsed, 1),
        "subagent_calls": subagent_calls,
        "tool_calls": dict(sorted(counts.items())),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes = files.get(REPORT_PATH)
    sources_bytes = files.get(SOURCES_PATH)
    if not report_bytes or not report_bytes.strip():
        raise RuntimeError("agent did not create a non-empty report.md")
    if not sources_bytes:
        raise RuntimeError("agent did not create sources.json")
    try:
        report = report_bytes.decode("utf-8")
        sources = json.loads(sources_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid downloaded output: {exc}") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("sources.json must be a non-empty JSON array")

    summary = summarize(messages, elapsed, model_name)
    summary.update({
        "topic": topic,
        "n_sources": len(sources),
        "source_families": sorted({entry.get("source") for entry in sources if isinstance(entry, dict)
                                   and isinstance(entry.get("source"), str)}),
    })
    reports_dir.mkdir(parents=True, exist_ok=True)
    stem = slugify(topic)
    report_path = reports_dir / f"{stem}.md"
    sources_path = reports_dir / f"{stem}.sources.json"
    meta_path = reports_dir / f"{stem}.meta.json"
    # All downloaded content is validated before any target is overwritten.
    report_path.write_text(report, encoding="utf-8")
    sources_path.write_text(json.dumps(sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    if not (topic or "").strip():
        print('usage: python research.py "research topic"', file=sys.stderr)
        return 2
    try:
        model = make_model()
        model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__
        start = time.monotonic()
        with open_sandbox() as backend:
            backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })
            agent = build_lead_agent(backend, model)
            result = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic.strip())}]},
                config={"recursion_limit": 1000},
            )
            messages = result.get("messages", []) if isinstance(result, dict) else getattr(result, "messages", [])
            report_path = save_outputs(backend, topic.strip(), messages, time.monotonic() - start, model_name)
    except Exception as exc:  # Includes sandbox/model/agent failures; no outputs are saved after a failed run.
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"saved report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
