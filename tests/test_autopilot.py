"""The autopilot supervisor (iris/autopilot/): fence, referees, redaction, ledger, a whole night with a fake worker."""

import json
import subprocess
import sys
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import yaml

from iris import paths
from iris.autopilot import config, digest, fence, gateway, ledger, night, redact, referees, sandbox, timer, workspace


def sh(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def write(repo, rel, text):
    p = Path(repo) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


@pytest.fixture
def repo(tmp_path):
    """A main repo with an `iris` package, a frozen golden file and a variant branch, plus the worker's clone."""
    r = tmp_path / "repo"
    r.mkdir()
    sh(r, "init", "-q", "-b", "main")
    write(r, "iris/core.py", "def f():\n    return 1\n")
    write(r, "iris/llm.py", "import requests\n\ndef ask():\n    return requests.get('http://127.0.0.1:11434')\n")
    write(r, "iris/autopilot/x.py", "A = 1\n")
    write(r, "self_template/evals/golden/core.yaml", "cases: []\n")
    write(r, "docs/variants.yaml", "variants: []\n")
    write(r, "docs/variants/autopilot/CHANGELOG.md", "# log\n")
    sh(r, "add", "-A")
    sh(r, "commit", "-qm", "init")
    sh(r, "branch", "variant/autopilot", "main")
    return r


@pytest.fixture
def work(repo, iris_home):
    w = config.work_dir()
    w.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "-q", str(repo), str(w)], check=True, capture_output=True)
    sh(w, "checkout", "-q", "-B", "variant/autopilot", "origin/variant/autopilot")
    return w


# -- redaction -----------------------------------------------------------------------------------------------------
def test_redaction_hides_what_the_assistant_knows_about_you():
    facts = ["The user's sister Ana lives in Durban.", "The user works at Acme Robotics.", "Likes cumin."]
    terms = redact.personal_terms(facts, ["Yugen Naidoo"])
    assert {"Ana", "Durban", "Acme", "Robotics", "Yugen Naidoo"} <= set(terms) and "Likes" not in terms and "The" not in terms
    r = redact.Redactor(terms, home=Path("/home/yugen"))
    out = r.text("Ana called from Durban (+27 82 123 4567) on ana@example.com about /home/yugen/x, 12 Long Street, acct 12345678901 "
                 "see https://example.com/private/page")
    for secret in ("Ana", "Durban", "27 82", "ana@example.com", "/home/yugen", "Long Street", "12345678901", "private/page"):
        assert secret not in out, out
    assert "[email]" in out and "[phone]" in out and "[address]" in out and "~/x" in out and "https://example.com/[path]" in out
    assert r.text("Ana and ANA and durban") == "[person-or-place-1] and [person-or-place-1] and [person-or-place-2]"   # stable placeholders


def test_a_folder_of_markdown_is_redacted_into_another(tmp_path):
    src, dst = tmp_path / "a", tmp_path / "b"
    src.mkdir()
    (src / "x.md").write_text("Ana said hi")
    (src / "ignore.txt").write_text("Ana")
    assert redact.Redactor(["Ana"]).folder(src, dst) == 1
    assert (dst / "x.md").read_text() == "[person-or-place-1] said hi" and not (dst / "ignore.txt").exists()


# -- ledger ------------------------------------------------------------------------------------------------------------
def test_the_ledger_counts_tokens_and_stops_on_budget_or_repeated_errors():
    n = ledger.Night("2026-10-19")
    budget = {"max_tokens_per_night": 1000, "stop_after_consecutive_errors": 3}
    n.feed({"type": "turn.completed", "usage": {"input_tokens": 400, "output_tokens": 50}})
    n.feed({"type": "item.completed", "item": {"type": "error", "message": "Model metadata for `x` not found"}})      # harmless
    assert (n.tokens, n.errors, n.exhausted(budget)) == (450, 0, None)
    for _ in range(3):
        n.feed({"type": "item.completed", "item": {"type": "error", "message": "429 rate limited"}})
    assert "3 provider errors in a row" in n.exhausted(budget)
    n.feed({"type": "turn.completed", "usage": {"input_tokens": 600, "output_tokens": 0}})
    assert "token budget used" in n.exhausted(budget) and n.consecutive_errors == 0


# -- fence ---------------------------------------------------------------------------------------------------------------
def test_a_clean_change_passes_the_fence(work):
    start = sh(work, "rev-parse", "HEAD")
    write(work, "iris/core.py", "def f():\n    return 2\n")
    v = fence.check(work, start)
    assert v.ok and v.changed == ["iris/core.py"]


@pytest.mark.parametrize("rel", ["self_template/evals/golden/core.yaml", "iris/autopilot/x.py", "docs/variants.yaml", "scripts/autopilot_run.sh"])
def test_frozen_files_cannot_change(work, rel):
    start = sh(work, "rev-parse", "HEAD")
    write(work, rel, "tampered\n")
    sh(work, "add", "-A")
    sh(work, "commit", "-qm", "x")
    v = fence.check(work, start)
    assert not v.ok and any(rel in m for m in v.violations)


def test_new_outbound_network_in_runtime_code_is_refused(work):
    start = sh(work, "rev-parse", "HEAD")
    write(work, "iris/phone_home.py", "import requests\n\ndef go():\n    requests.post('https://api.openai.com/v1/chat')\n")
    write(work, "iris/llm.py", "import requests\n\ndef ask():\n    return requests.get('http://127.0.0.1:11434')\n\nURL = 'https://api.anthropic.com/v1'\n")
    v = fence.check(work, start)
    text = "\n".join(v.violations)
    assert not v.ok and "phone_home.py: new network use" in text and "api.openai.com" in text and "api.anthropic.com" in text


def test_network_use_where_iris_already_talks_to_the_network_is_fine(work):
    start = sh(work, "rev-parse", "HEAD")
    write(work, "iris/llm.py", "import requests\n\ndef ask():\n    return requests.post('http://localhost:11434/api/chat')\n\ndef ps():\n    return requests.get('http://localhost:11434/api/ps')\n")
    write(work, "tests/test_net.py", "import requests\n")                       # tests and scripts are not runtime code
    assert fence.check(work, start).ok


def test_secrets_in_the_diff_are_refused(work):
    start = sh(work, "rev-parse", "HEAD")
    write(work, "iris/core.py", "KEY = 'freellmapi-" + "a" * 48 + "'\n")
    assert any("secret" in m for m in fence.check(work, start).violations)


def test_the_main_repo_and_your_agent_must_not_move():
    assert fence.isolation({"repo:main": "abc12345", "agent:iris": "def"}, {"repo:main": "abc12345", "agent:iris": "def"}) == []
    assert "repo:main changed" in fence.isolation({"repo:main": "abc12345"}, {"repo:main": "ffff0000"})[0]


# -- referees ----------------------------------------------------------------------------------------------------------------
def test_holdout_may_drop_only_a_little():
    assert referees.compare(0.98, 0.97, 2.0)[0] and referees.compare(0.98, 0.99, 2.0)[0]
    ok, msg = referees.compare(0.98, 0.90, 2.0)
    assert not ok and "-8.0 points" in msg and not referees.compare(None, 0.9, 2.0)[0]


def test_code_changes_need_the_variant_changelog(work):
    start = sh(work, "rev-parse", "HEAD")
    write(work, "iris/core.py", "def f():\n    return 3\n")
    assert not referees.changelog_ok(work, start)[0]
    write(work, "docs/variants/autopilot/CHANGELOG.md", "# log\n- f returns 3\n")
    assert referees.changelog_ok(work, start)[0]
    sh(work, "checkout", "-q", "--", ".")
    write(work, "docs/other.md", "x")
    assert referees.changelog_ok(work, start)[0]                          # docs only: no entry needed


# -- sandbox -------------------------------------------------------------------------------------------------------------------
def test_the_sandbox_hides_home_and_binds_only_the_workspace(tmp_path):
    w, h = tmp_path / "w", tmp_path / "h"
    argv = sandbox.wrap(["python", "-V"], work=w, scratch_home=h, net=False, env={"X": "1"})
    s = " ".join(argv)
    assert argv[0] == "bwrap" and "--unshare-net" in argv and f"--tmpfs {Path.home()}" in s and f"--bind {w} {w}" in s
    assert "--ro-bind / /" in s and "--setenv IRIS_HOME" in s and argv[-2:] == ["python", "-V"]
    assert "--unshare-net" not in sandbox.wrap(["x"], work=w, scratch_home=h, net=True)
    plain = sandbox.wrap(["x"], work=w, scratch_home=h, enabled=False)
    assert plain[0] == "env" and "bwrap" not in plain


@pytest.mark.skipif(not sandbox.available(), reason="bubblewrap not installed")
def test_the_real_sandbox_really_hides_your_files(tmp_path):
    secret = Path.home() / ".iris-sandbox-test-secret"
    secret.write_text("x")
    try:
        w, h = tmp_path / "w", tmp_path / "h"
        argv = sandbox.wrap([sys.executable, "-c", f"import os; print(os.path.exists({str(secret)!r}))"], work=w, scratch_home=h, net=False)
        out = subprocess.run(argv, capture_output=True, text=True)
        assert out.stdout.strip() == "False", out
        probe = sandbox.wrap(["sh", "-c", f"echo ok > {w}/f && ! touch /etc/x 2>/dev/null && echo contained"], work=w, scratch_home=h, net=False)
        r = subprocess.run(probe, capture_output=True, text=True)
        assert "contained" in r.stdout and (w / "f").read_text().strip() == "ok", r
    finally:
        secret.unlink()


# -- workspace -------------------------------------------------------------------------------------------------------------------
def test_main_is_merged_into_the_variant_and_conflicts_are_left_for_the_worker(repo, work):
    assert workspace.sync(work)[0] == "current"
    write(repo, "iris/new.py", "N = 1\n")
    sh(repo, "add", "-A"), sh(repo, "commit", "-qm", "main moves")
    assert workspace.sync(work) == ("merged", "1 commit(s) from main") and (work / "iris/new.py").exists()
    write(work, "iris/core.py", "def f():\n    return 'variant'\n")
    sh(work, "commit", "-qam", "variant edit")
    write(repo, "iris/core.py", "def f():\n    return 'main'\n")
    sh(repo, "commit", "-qam", "main edit")
    status, files = workspace.sync(work)
    assert status == "conflict" and files == "iris/core.py"


def test_a_rejected_night_leaves_no_trace_and_an_accepted_one_reaches_the_main_repo_only_through_the_supervisor(repo, work):
    start = sh(work, "rev-parse", "HEAD")
    write(work, "iris/core.py", "def f():\n    return 9\n")
    write(work, "iris/junk.py", "J = 1\n")
    sh(work, "add", "iris/core.py"), sh(work, "commit", "-qm", "edit")
    workspace.reject(work, start)
    assert sh(work, "rev-parse", "HEAD") == start and not (work / "iris/junk.py").exists() and sh(work, "status", "--porcelain") == ""
    write(work, "iris/core.py", "def f():\n    return 10\n")                       # uncommitted work is committed on accept
    workspace.accept(work, repo, "autopilot/2026-10-19")
    assert sh(repo, "show", "variant/autopilot:iris/core.py").endswith("return 10")
    assert sh(repo, "rev-parse", "autopilot/2026-10-19") == sh(work, "rev-parse", "HEAD")


def test_the_scorer_is_pinned_outside_the_repo_with_absolute_imports(repo, iris_home):
    write(repo, "iris/replay.py", "from . import paths\nimport os\n\ndef f():\n    from .config import Settings\n    from . import trace, selfstore\n")
    d = workspace.pin_runner(repo)
    src = (d / "frozen_replay.py").read_text()
    assert "from iris import paths" in src and "from iris.config import Settings" in src and "from iris import trace, selfstore" in src
    assert "from ." not in src and (d / "score.py").exists()


# -- the gateway -------------------------------------------------------------------------------------------------------------------
class _Gateway(BaseHTTPRequestHandler):
    models = {"good": True, "limited": False}

    def log_message(self, *a): pass

    def _send(self, code, obj, headers=None):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.headers.get("Authorization") != "Bearer freellmapi-test":
            return self._send(401, {"error": "bad key"})
        self._send(200, {"data": [{"id": m} for m in self.models]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if not self.models.get(body["model"]):
            return self._send(429, {"error": {"message": "all rate-limited"}})
        msg = {"content": "def is_palindrome(s):\n    return True"}
        if body.get("tools"):
            msg = {"content": None, "tool_calls": [{"function": {"name": "run_tests"}}]}
        self._send(200, {"choices": [{"message": msg}]}, {"X-Routed-Via": "fake/" + body["model"]})


@pytest.fixture
def fake_gateway(iris_home):
    srv = HTTPServer(("127.0.0.1", 0), _Gateway)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    config.root().mkdir(parents=True, exist_ok=True)
    (config.root() / "freellmapi.key").write_text("freellmapi-test\n")
    (config.root() / "config.yaml").write_text(yaml.safe_dump({"gateway": {"url": f"http://127.0.0.1:{srv.server_address[1]}/v1"}}))
    yield srv
    srv.shutdown()


def test_doctor_says_which_models_can_answer_and_call_tools(fake_gateway):
    d = gateway.doctor(["good", "limited", "absent"])
    assert d["key_ok"] and d["catalog"] == 2 and d["usable"] == ["good"]
    by = {p["model"]: p for p in d["probes"]}
    assert by["good"]["via"] == "fake/good" and by["good"]["code"] and by["limited"]["status"] == 429 and by["absent"]["error"] == "not in the catalog"


def test_doctor_explains_a_bad_key_and_a_gateway_that_is_down(fake_gateway):
    (config.root() / "freellmapi.key").write_text("wrong\n")
    assert "key rejected" in gateway.doctor(["good"])["problem"]
    (config.root() / "config.yaml").write_text(yaml.safe_dump({"gateway": {"url": "http://127.0.0.1:9/v1"}}))
    assert "not reachable" in gateway.doctor(["good"])["problem"]


# -- one whole night with a fake worker ----------------------------------------------------------------------------------------------
AGENT = textwrap.dedent('''
    import json, os, sys
    sys.stdin.read()
    work = os.getcwd()
    mode = os.environ.get("FAKE_MODE", "improve")
    model = os.environ.get("AUTOPILOT_MODEL")
    def ev(**k): print(json.dumps(k), flush=True)
    if mode == "quota" and model == "m1":
        for _ in range(9):
            ev(type="item.completed", item={"type": "error", "message": "429 rate limited"})
        sys.exit(1)
    if mode in ("improve", "quota"):
        open("iris/core.py", "w").write("def f():\\n    return 2\\n")
        open("docs/variants/autopilot/CHANGELOG.md", "a").write("- f returns 2\\n")
    elif mode == "tamper":
        open("self_template/evals/golden/core.yaml", "w").write("cases: [pass-everything]\\n")
    elif mode == "nothing":
        pass
    ev(type="turn.completed", usage={"input_tokens": 1000, "output_tokens": 100})
    open(os.environ.get("AUTOPILOT_LASTMSG", "/dev/null"), "w").write("I changed f.")
''')


@pytest.fixture
def agent_script(tmp_path):
    p = tmp_path / "agent.py"
    p.write_text(AGENT)
    return [sys.executable, str(p)]


@pytest.fixture
def prepared(repo, work, fake_gateway, monkeypatch, tmp_path):
    monkeypatch.setattr(night, "build_dossier", lambda w: {"agents": []})
    monkeypatch.setattr(night, "skip_reason", lambda cfg=None, force=False: None)
    monkeypatch.setattr(referees, "run_tests", lambda w, enabled=True: {"ok": True, "passed": 10, "failed": 0, "tail": ""})
    scores = {"rate": 0.98}
    monkeypatch.setattr(referees, "holdout_score", lambda w, enabled=True: {"ok": True, "pass_rate": scores["rate"], "passed": 49, "cases": 50, "by_route": {}})
    config.save_state({"baseline": {"holdout": 0.98, "sha": "x", "date": "2026-10-18"}})
    (config.root() / "config.yaml").write_text(yaml.safe_dump({
        "gateway": {"url": fake_gateway and f"http://127.0.0.1:{fake_gateway.server_address[1]}/v1"},
        "models": ["m1", "m2"], "agent": {"timebox_min": 1, "command": "codex"}}))
    monkeypatch.setattr(workspace, "write_codex_config", lambda model: tmp_path / "codexhome")
    return scores


def go(repo, argv, monkeypatch, mode):
    monkeypatch.setenv("FAKE_MODE", mode)
    return night.run(repo=repo, use_sandbox=False, argv=argv, notify=False)


def test_a_good_night_is_kept_tagged_published_and_reported(repo, work, prepared, agent_script, monkeypatch):
    r = go(repo, agent_script, monkeypatch, "improve")
    assert r["accepted"] and r["tag"].startswith("autopilot/")
    assert "return 2" in sh(repo, "show", "variant/autopilot:iris/core.py")
    assert sh(repo, "tag", "-l", r["tag"]) == r["tag"]
    assert config.state()["baseline"]["sha"] == r["end"] and config.state()["nights"][-1]["accepted"]
    d = Path(r["digest"]).read_text()
    assert "KEPT" in d and "I changed f." in d and "holdout: 49/50" in d and "tokens" in d


def test_a_night_that_tampers_with_the_referees_is_discarded(repo, work, prepared, agent_script, monkeypatch):
    start = sh(work, "rev-parse", "HEAD")
    r = go(repo, agent_script, monkeypatch, "tamper")
    assert not r["accepted"] and any("frozen file changed" in x for x in r["reasons"])
    assert sh(work, "rev-parse", "HEAD") == start and "cases: []" in (work / "self_template/evals/golden/core.yaml").read_text()
    assert "DISCARDED" in Path(r["digest"]).read_text()


def test_a_night_that_makes_the_holdout_worse_is_discarded(repo, work, prepared, agent_script, monkeypatch):
    prepared["rate"] = 0.90
    r = go(repo, agent_script, monkeypatch, "improve")
    assert not r["accepted"] and any("-8.0 points" in x for x in r["reasons"])
    assert "return 1" in (work / "iris/core.py").read_text()               # reverted


def test_failing_tests_discard_the_night(repo, work, prepared, agent_script, monkeypatch):
    monkeypatch.setattr(referees, "run_tests", lambda w, enabled=True: {"ok": False, "passed": 9, "failed": 1, "tail": ""})
    r = go(repo, agent_script, monkeypatch, "improve")
    assert not r["accepted"] and any("tests fail" in x for x in r["reasons"])


def test_a_night_with_no_change_is_not_a_success(repo, work, prepared, agent_script, monkeypatch):
    r = go(repo, agent_script, monkeypatch, "nothing")
    assert not r["accepted"] and "nothing changed" in r["reasons"]


def test_a_model_with_no_quota_left_hands_over_to_the_next(repo, work, prepared, agent_script, monkeypatch):
    config.save_state({"baseline": {"holdout": 0.98}})
    (config.root() / "config.yaml").write_text(yaml.safe_dump({
        "gateway": {"url": "http://127.0.0.1:9/v1"}, "models": ["m1", "m2"], "agent": {"timebox_min": 5, "command": "codex"},
        "budget": {"stop_after_consecutive_errors": 4}}))
    r = go(repo, agent_script, monkeypatch, "quota")
    assert r["ledger"]["models"] == ["m1", "m2"] and r["accepted"] and r["agent"]["model"] == "m2"


def test_the_stop_file_skips_the_night(iris_home):
    config.root().mkdir(parents=True, exist_ok=True)
    config.stop_file().write_text("stop\n")
    assert "STOP file" in night.skip_reason({})


def test_skip_reasons(iris_home, monkeypatch):
    assert "not initialised" in night.skip_reason({})
    (config.work_dir() / ".git").mkdir(parents=True)
    assert "no gateway key" in night.skip_reason({})
    config.root().mkdir(parents=True, exist_ok=True)
    (config.root() / "freellmapi.key").write_text("k\n")
    assert night.skip_reason({}) is None
    from iris import jobs
    jobs.pause(None)
    assert "paused" in night.skip_reason({}) and night.skip_reason({}, force=True) is None


# -- timer and CLI -------------------------------------------------------------------------------------------------------------------
def test_the_timer_is_only_written_by_enable_and_removed_by_disable(iris_home, monkeypatch, tmp_path):
    from iris import jobs
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setattr(jobs, "_systemctl", lambda *a: subprocess.CompletedProcess(a, 0, "", ""))
    assert not (tmp_path / "cfg/systemd/user/iris-autopilot.timer").exists()
    timer.enable()
    t = (tmp_path / "cfg/systemd/user/iris-autopilot.timer").read_text()
    assert "OnCalendar=*-*-* 02:30" in t and "iris autopilot run" in (tmp_path / "cfg/systemd/user/iris-autopilot.service").read_text()
    assert timer.disable() == ["iris-autopilot.timer", "iris-autopilot.service"] and not (tmp_path / "cfg/systemd/user/iris-autopilot.timer").exists()


def test_the_cli_reports_status_and_stops(iris_home, capsys):
    from iris import cli
    sys.argv = ["iris", "autopilot", "status"]
    cli.main()
    assert "baseline" in capsys.readouterr().out
    sys.argv = ["iris", "autopilot", "stop"]
    cli.main()
    assert config.stop_file().exists()
    sys.argv = ["iris", "autopilot", "resume"]
    cli.main()
    assert not config.stop_file().exists()
