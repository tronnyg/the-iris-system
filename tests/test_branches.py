"""Branch workflow (iris/branches.py, docs/BRANCHING.md): the rules refuse, variants sync."""

import subprocess
import sys

import pytest
import yaml

from iris import branches as b


def sh(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def commit(repo, name, text, msg=None):
    (repo / name).write_text(text)
    sh(repo, "add", "-A")
    sh(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", msg or f"change {name}")


@pytest.fixture
def repo(tmp_path):
    """A repo with an origin: main with one commit, the registry file, and nothing else."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    r = tmp_path / "work"
    r.mkdir()
    sh(r, "init", "-q", "-b", "main")
    sh(r, "remote", "add", "origin", str(origin))
    (r / "docs").mkdir()
    (r / "docs" / "variants.yaml").write_text("variants: []\n")
    commit(r, "a.txt", "one\n")
    sh(r, "push", "-q", "origin", "main")
    sh(r, "fetch", "-q")
    return r


def register(repo, *variants):
    (repo / "docs" / "variants.yaml").write_text(yaml.safe_dump({"variants": list(variants)}))
    sh(repo, "add", "-A")
    sh(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "registry")
    sh(repo, "push", "-q", "origin", "main")


def test_start_creates_from_main_and_refuses_a_second_branch(repo):
    assert b.start(repo, "fix/lists") == "fix/lists"
    sh(repo, "checkout", "main")
    with pytest.raises(b.BranchError, match="still open"):
        b.start(repo, "feat/other")


def test_start_refuses_bad_names_dirty_tree_and_non_main(repo):
    with pytest.raises(b.BranchError, match="name it"):
        b.start(repo, "improve/x")
    with pytest.raises(b.BranchError, match="name it"):
        b.start(repo, "fix/Bad Name")
    (repo / "a.txt").write_text("dirty\n")
    with pytest.raises(b.BranchError, match="uncommitted"):
        b.start(repo, "fix/x")
    sh(repo, "checkout", "--", "a.txt")
    sh(repo, "checkout", "-q", "-b", "scratch")
    with pytest.raises(b.BranchError, match="start from main"):
        b.start(repo, "fix/x")


def test_start_refuses_when_main_is_behind_or_ahead_of_origin(repo, tmp_path):
    other = tmp_path / "other"
    subprocess.run(["git", "clone", "-q", str(tmp_path / "origin.git"), str(other)], check=True)
    commit(other, "b.txt", "x\n")
    sh(other, "push", "-q", "origin", "main")
    with pytest.raises(b.BranchError, match="behind"):
        b.start(repo, "fix/x")
    sh(repo, "pull", "-q", "--ff-only", "origin", "main")
    commit(repo, "c.txt", "direct commit to main\n")
    with pytest.raises(b.BranchError, match="committed to main directly"):
        b.start(repo, "fix/x")


def test_finish_needs_approval_then_merges_pushes_and_deletes(repo):
    b.start(repo, "fix/lists")
    commit(repo, "l.txt", "fixed\n")
    with pytest.raises(b.BranchError, match="approval"):
        b.finish(repo)
    assert b.finish(repo, approved=True) == "fix/lists"
    assert b.current(repo) == "main"
    assert b.improvement_branches(repo) == []
    assert "l.txt" in sh(repo, "ls-tree", "--name-only", "origin/main")
    assert len(sh(repo, "log", "-1", "--format=%P").split()) == 2        # --no-ff: a merge commit
    b.start(repo, "feat/next")                                          # the slot is free again


def test_finish_refuses_a_branch_that_lags_main_or_is_not_an_improvement(repo):
    with pytest.raises(b.BranchError, match="not an improvement"):
        b.finish(repo, approved=True)
    sh(repo, "checkout", "-q", "-b", "fix/old")
    commit(repo, "o.txt", "x\n")
    sh(repo, "checkout", "-q", "main")
    commit(repo, "m.txt", "moved on\n")
    sh(repo, "checkout", "-q", "fix/old")
    with pytest.raises(b.BranchError, match="not cut from the tip"):
        b.finish(repo, approved=True)


def test_registry_is_validated(repo):
    register(repo, {"name": "Bad Name", "mode": "auto"})
    with pytest.raises(b.BranchError, match="bad variant name"):
        b.load_registry(repo)
    register(repo, {"name": "x", "mode": "sometimes"})
    with pytest.raises(b.BranchError, match="mode"):
        b.load_registry(repo)
    register(repo, {"name": "x", "mode": "auto"}, {"name": "x", "mode": "manual"})
    with pytest.raises(b.BranchError, match="duplicate"):
        b.load_registry(repo)


def variant(repo, name, mode, **files):
    register(repo, *(b.load_registry(repo) + [{"name": name, "mode": mode, "purpose": "test"}]))
    sh(repo, "branch", f"variant/{name}")
    wt = b.worktree_for(name)
    wt.parent.mkdir(parents=True, exist_ok=True)
    sh(repo, "worktree", "add", "-q", str(wt), f"variant/{name}")
    for f, text in files.items():
        commit(wt, f, text)
    return wt


def test_variants_do_not_count_toward_the_single_branch_rule(repo):
    variant(repo, "trial", "auto")
    variant(repo, "other", "manual")
    assert b.start(repo, "fix/x") == "fix/x"
    s = b.status(repo)
    assert s["improvements"] == ["fix/x"] and {v["name"] for v in s["variants"]} == {"trial", "other"}


def test_auto_variant_follows_main_with_a_clean_merge(repo):
    wt = variant(repo, "trial", "auto", **{"v.txt": "variant only\n"})
    commit(repo, "m.txt", "main moved\n")
    sh(repo, "push", "-q", "origin", "main")
    assert b.sync(repo, test_cmd=[sys.executable, "-c", "pass"]) == {"trial": "synced"}
    assert (wt / "m.txt").exists() and (wt / "v.txt").exists()
    assert "[auto-sync]" in sh(repo, "log", "-1", "--format=%s", "variant/trial")
    assert b.status(repo)["variants"][0]["behind_main"] == 0
    assert b.sync(repo) == {"trial": "current"}


def test_conflict_aborts_blocks_and_stops_touching_the_variant(repo):
    wt = variant(repo, "trial", "auto", **{"a.txt": "variant edit\n"})
    before = sh(repo, "rev-parse", "variant/trial")
    commit(repo, "a.txt", "main edit\n")
    sh(repo, "push", "-q", "origin", "main")
    assert b.sync(repo, test_cmd=[sys.executable, "-c", "pass"]) == {"trial": "blocked"}
    assert sh(repo, "rev-parse", "variant/trial") == before
    assert not sh(wt, "status", "--porcelain")                       # no half-merged worktree
    assert b.status(repo)["variants"][0]["blocked"]
    assert b.sync(repo) == {"trial": "blocked"}                       # stays untouched until unblocked
    b.unblock("trial")
    assert not b.status(repo)["variants"][0]["blocked"]


def test_resolver_can_fix_a_conflict(repo):
    wt = variant(repo, "trial", "auto", **{"a.txt": "variant edit\n"})
    commit(repo, "a.txt", "main edit\n")
    sh(repo, "push", "-q", "origin", "main")
    seen = []

    def resolver(path, files, attempt):
        seen.append((files, attempt))
        (path / "a.txt").write_text("resolved\n")
        return True

    assert b.sync(repo, resolver=resolver, test_cmd=[sys.executable, "-c", "pass"]) == {"trial": "synced"}
    assert seen == [(["a.txt"], 1)] and (wt / "a.txt").read_text() == "resolved\n"


def test_red_tests_after_a_clean_merge_block_it(repo):
    variant(repo, "trial", "auto", **{"v.txt": "x\n"})
    commit(repo, "m.txt", "main moved\n")
    sh(repo, "push", "-q", "origin", "main")
    assert b.sync(repo, test_cmd=[sys.executable, "-c", "raise SystemExit(1)"]) == {"trial": "blocked"}


def test_manual_variant_is_never_modified_only_reported(repo):
    variant(repo, "mine", "manual", **{"v.txt": "x\n"})
    before = sh(repo, "rev-parse", "variant/mine")
    commit(repo, "m.txt", "main moved\n")
    sh(repo, "push", "-q", "origin", "main")
    assert b.sync(repo) == {"mine": "skipped"}
    assert sh(repo, "rev-parse", "variant/mine") == before
    assert b.status(repo)["variants"][0]["behind_main"] == 1


def test_status_flags_unregistered_variants_and_stale_branches(repo):
    sh(repo, "branch", "variant/ghost")
    sh(repo, "branch", "fix/done-long-ago")                            # merged (no new commits), not deleted
    s = b.status(repo)
    assert s["unregistered"] == ["ghost"] and "fix/done-long-ago" in s["stale"]


def test_the_shipped_registry_and_docs_are_valid():
    from iris import paths
    assert isinstance(b.load_registry(paths.REPO), list)
    for f in ("docs/BRANCHING.md", "AGENTS.md"):
        assert (paths.REPO / f).exists()
    assert "docs/BRANCHING.md" in (paths.REPO / "CLAUDE.md").read_text()


def test_finish_refuses_code_changes_without_a_changelog_line(repo):
    (repo / "CHANGELOG.md").write_text("# Changelog\n\n## [Unreleased]\n")
    commit(repo, "CHANGELOG.md", "# Changelog\n\n## [Unreleased]\n", "changelog")
    sh(repo, "push", "-q", "origin", "main")
    b.start(repo, "fix/code")
    (repo / "iris").mkdir()
    commit(repo, "iris/x.py", "x = 1\n")
    with pytest.raises(b.BranchError, match="CHANGELOG.md"):
        b.finish(repo, approved=True)
    commit(repo, "CHANGELOG.md", "# Changelog\n\n## [Unreleased]\n- x is 1\n", "changelog line")
    assert b.finish(repo, approved=True) == "fix/code"


def test_docs_only_changes_need_no_changelog_line(repo):
    b.start(repo, "docs/typo")
    commit(repo, "notes.md", "fixed a typo\n")
    assert b.finish(repo, approved=True) == "docs/typo"
