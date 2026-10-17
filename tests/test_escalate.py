"""The escalation stage, driven by a fake `claude` CLI (the real one is never called in tests)."""

import json
import subprocess
import time
import sys
import textwrap
from pathlib import Path

from iris import paths, selfstore, trace
from iris.improve import report, review
from tests.test_loops import turn, write_turns

FAKE_CLAUDE = textwrap.dedent('''\
    #!{python}
    """Pretends to be `claude -p`: reads the prompt, edits files, prints a JSON result."""
    import json, os, sys, pathlib
    assert "ANTHROPIC_API_KEY" not in os.environ      # must use the subscription, never the API
    args = sys.argv[1:]
    prompt = sys.stdin.read()
    assert "--model" in args and "-p" in args
    dossier = pathlib.Path(args[args.index("--add-dir") + 1])
    assert (dossier / "conversations.md").read_text().count("question") >= 3
    persona = dossier / "self/persona/iris.yaml"
    persona.write_text(persona.read_text().replace("Hello. IRIS online.", "Hey. I'm here."))
    (dossier / "self/policy.yaml").write_text("tampered: true\\n")          # must be refused
    (dossier / "analysis.md").write_text("# Analysis\\n\\nReplies were fine. Changed the greeting.\\n")
    pathlib.Path("NOTES_FROM_CLAUDE.md").write_text("a code change\\n")    # cwd = code worktree
    print(json.dumps({{"type": "assistant", "message": {{"content": [{{"type": "tool_use", "name": "Edit",
          "input": {{"file_path": str(persona)}}}}]}}}}), flush=True)
    print(json.dumps({{"type": "result", "result": "Changed the greeting.", "is_error": False,
                      "session_id": "abc", "total_cost_usd": 0.0, "num_turns": 3}}))
''')


def _code_repo(tmp_path, monkeypatch) -> Path:
    """A throwaway clone of the code repo, so tests never create branches in the real one."""
    src = paths.REPO
    dst = tmp_path / "code"
    subprocess.run(["git", "init", "-q", "-b", "main", str(dst)], check=True)
    (dst / "README.md").write_text("code\n")
    (dst / "self_template").symlink_to(src / "self_template")
    subprocess.run(["git", "-C", str(dst), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(dst), "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"],
                   check=True)
    monkeypatch.setattr(paths, "REPO", dst)
    return dst


def test_escalation_makes_proposals(tmp_path, monkeypatch):
    fake = tmp_path / "claude"
    fake.write_text(FAKE_CLAUDE.format(python=sys.executable))
    fake.chmod(0o755)
    monkeypatch.setenv("IRIS_CLAUDE_COMMAND", str(fake))
    monkeypatch.setenv("IRIS_TEST_COMMAND", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-be-removed")
    repo = _code_repo(tmp_path, monkeypatch)
    write_turns([turn(i) for i in range(1, 5)])

    from iris.improve.night import Night
    r = Night(only={"metrics", "escalate", "report"}).run()
    esc = r["stages"]["escalate"]
    assert esc["status"] == "ok", esc
    res = esc["result"]
    assert res["claude"]["ok"] and res["turns"] == 4
    assert res["refused_self_paths"] == ["policy.yaml"]

    # self changes became one proposal branch; the live self store is untouched
    prop = res["self"]["proposal"]
    assert prop in selfstore.proposals()
    assert "Hello. IRIS online." in (paths.self_dir() / "persona/iris.yaml").read_text()
    assert "policy.yaml" not in selfstore.proposal_files(prop)

    # code changes became a branch in the code repo, with an evidence file
    code = res["code"]
    branches = subprocess.run(["git", "-C", str(repo), "branch", "--list"], capture_output=True, text=True).stdout
    assert code["branch"] in branches and code["tests_passed"]
    ids = [p["id"] for p in report.pending_proposals()]
    assert code["id"] in ids and prop.split("/", 1)[1] in ids

    html = (paths.reports_dir() / f"{r['date']}.html").read_text()
    assert "Escalation" in html and "Changed the greeting" in html

    # approving the code proposal merges it into main
    review.approve(code["id"])
    assert (repo / "NOTES_FROM_CLAUDE.md").exists()
    assert code["id"] not in [p["id"] for p in report.pending_proposals()]

    # a second run with nothing new is skipped
    r2 = Night(only={"metrics", "escalate"}).run()
    assert "skipped" in r2["stages"]["escalate"]["result"]


def test_escalation_skips_without_cli(monkeypatch):
    from iris.improve.night import Night
    write_turns([turn(1)])
    r = Night(only={"escalate"}).run()
    assert "not found" in r["stages"]["escalate"]["result"]["skipped"]


SLOW_CLAUDE = textwrap.dedent('''\
    #!{python}
    import os, subprocess, sys, time, pathlib
    sys.stdin.read()
    args = sys.argv[1:]
    dossier = pathlib.Path(args[args.index("--add-dir") + 1])
    (dossier / "self/memory/lessons.md").write_text("- half-finished lesson\\n")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    pathlib.Path(os.environ["CHILD_PID_FILE"]).write_text(str(child.pid))
    time.sleep(600)
''')


def test_escalation_timeout_stops_claude_and_its_children(tmp_path, monkeypatch):
    import os
    from iris.improve import escalate
    fake = tmp_path / "claude"
    fake.write_text(SLOW_CLAUDE.format(python=sys.executable))
    fake.chmod(0o755)
    pidfile = tmp_path / "child.pid"
    monkeypatch.setenv("IRIS_CLAUDE_COMMAND", str(fake))
    monkeypatch.setenv("IRIS_TEST_COMMAND", "true")
    monkeypatch.setenv("CHILD_PID_FILE", str(pidfile))
    orig = escalate.config
    monkeypatch.setattr(escalate, "config", lambda st: {**orig(st), "max_minutes": 0.05})   # 3 seconds
    _code_repo(tmp_path, monkeypatch)
    write_turns([turn(i) for i in range(1, 3)])
    from iris.improve.night import Night
    r = Night(only={"metrics", "escalate"}).run()
    res = r["stages"]["escalate"]["result"]
    assert res["claude"]["timed_out"] and not res["claude"]["ok"]
    assert "INCOMPLETE" in json.loads(next(paths.proposals_dir().glob("proposal_claude-*.json")).read_text())["title"]
    child = int(pidfile.read_text())
    time.sleep(0.5)
    try:
        os.kill(child, 0)
        alive = Path(f"/proc/{child}/stat").read_text().split()[2] != "Z"
    except ProcessLookupError:
        alive = False
    assert not alive, "a process started by Claude survived the timeout"
