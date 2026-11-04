"""PreToolUse: the agent may not edit the judge (metrics script, baseline, these hooks)
or overwrite the baseline. Only humans move the goalposts."""
import json, sys

PROTECTED = ("scripts/metrics.py", "metrics/baseline.json", ".claude/")
d = json.load(sys.stdin)
ti = d.get("tool_input", {})
target = ti.get("file_path") or ti.get("notebook_path") or ""
cmd = ti.get("command", "")
hit = any(p in target for p in PROTECTED) or "--save" in cmd and "metrics.py" in cmd \
    or any(p in cmd for p in ("metrics/baseline.json", "scripts/metrics.py")) and any(
        w in cmd for w in (">", "sed -i", "rm ", "mv ", "cp ", "tee", "git checkout", "git restore"))
if hit:
    print("Protected: the judge (scripts/metrics.py, metrics/baseline.json, .claude/) is "
          "human-owned. Ask the user to change it.", file=sys.stderr)
    sys.exit(2)
