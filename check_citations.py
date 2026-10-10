"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]

    by_number = {}
    seen_urls = set()
    for index, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            problems.append(f"source entry {index} is not an object")
            continue
        number = source.get("n")
        url = source.get("url")
        if not isinstance(number, int) or isinstance(number, bool):
            problems.append(f"source entry {index} has non-integer n")
        elif number in by_number:
            problems.append(f"source number [{number}] appears more than once")
        else:
            by_number[number] = source
        if not isinstance(url, str) or not re.match(r"https?://", url):
            problems.append(f"source [{number}] has invalid url")
        elif url in seen_urls:
            problems.append(f"duplicate source url: {url}")
        else:
            seen_urls.add(url)

    # Split only at a proper Markdown heading. Everything afterwards is references,
    # so numbers there must not be mistaken for in-text citations.
    match = re.search(r"(?m)^## References\s*$", report_text)
    if not match:
        return problems + ["report is missing ## References"]
    body = report_text[:match.start()]
    references = report_text[match.end():]

    # Remove fenced code blocks and Markdown links before recognizing [n] citations.
    # This supports grouped citations such as [1, 2] and ranges such as [3-5].
    body = re.sub(r"```.*?```", "", body, flags=re.S)
    body = re.sub(r"\[[^\]]*\]\([^\)]*\)", "", body)
    cited = set()
    for group in re.findall(r"\[([0-9][0-9,\s-]*)\]", body):
        for piece in group.split(","):
            piece = piece.strip()
            range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", piece)
            if range_match:
                start, end = map(int, range_match.groups())
                if start <= end:
                    cited.update(range(start, end + 1))
                else:
                    problems.append(f"invalid citation range [{piece}]")
            elif piece.isdigit():
                cited.add(int(piece))

    for number in sorted(cited):
        if number not in by_number:
            problems.append(f"[{number}] cited but missing from sources.json")
    for number in sorted(by_number):
        if number not in cited:
            problems.append(f"source [{number}] never cited")

    reference_lines = re.findall(r"(?m)^\s*\[(\d+)\]\s*(.*)$", references)
    lines_by_number = {}
    for number_text, line in reference_lines:
        number = int(number_text)
        lines_by_number.setdefault(number, []).append(line)

    for number in sorted(by_number):
        count = len(lines_by_number.get(number, []))
        if count == 0:
            problems.append(f"source [{number}] missing from References")
        elif count > 1:
            problems.append(f"source [{number}] has {count} reference lines")
    for number in sorted(lines_by_number):
        if number not in by_number:
            problems.append(f"References contains [{number}] not in sources.json")

    url_pattern = re.compile(r"https?://[^\s)>\]}]+")
    for number, lines in lines_by_number.items():
        if number not in by_number or len(lines) != 1:
            continue
        urls = url_pattern.findall(lines[0])
        if len(urls) != 1:
            problems.append(f"reference [{number}] must contain exactly one URL")
        elif urls[0] != by_number[number].get("url"):
            problems.append(f"reference [{number}] URL does not match sources.json")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
