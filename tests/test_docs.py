"""The documentation stays complete and correct (docs/README.md, ADR-0024).

These checks are mechanical on purpose: they catch the drift an author (or an unattended agent) forgets,
so a branch opened after months still describes itself. They don't judge prose.
"""

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
ARCH = DOCS / "architecture"
ADR = DOCS / "adr"
PAGES = ["00-overview", "01-runtime-turn", "02-listening", "03-thinking", "04-speaking", "05-commands-and-skills",
         "06-memory", "07-self-store", "08-learning-loops", "09-jobs-and-processes", "10-web-ui",
         "11-observability", "12-data-and-files", "13-configuration-reference", "14-development"]


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def doc_files() -> list[Path]:
    out = [REPO / n for n in ("README.md", "CHANGELOG.md", "AGENTS.md", "CLAUDE.md")]
    return out + sorted(DOCS.rglob("*.md"))


def iris_code() -> str:
    return "\n".join(read(p) for p in (REPO / "iris").rglob("*.py"))


# -- architecture pages ------------------------------------------------------------------------
@pytest.mark.parametrize("page", PAGES)
def test_every_architecture_page_exists_and_has_a_diagram(page):
    text = read(ARCH / f"{page}.md")
    assert "```mermaid" in text, f"{page}.md has no Mermaid diagram"


def test_every_module_is_in_the_module_map():
    overview = read(ARCH / "00-overview.md")
    missing = [str(p.relative_to(REPO)) for p in sorted((REPO / "iris").rglob("*.py"))
               if f"`{p.relative_to(REPO)}`" not in overview]
    assert not missing, f"add these modules to the module map in docs/architecture/00-overview.md: {missing}"


def test_code_paths_named_in_the_docs_exist():
    stale = set()
    for f in doc_files():
        for m in re.finditer(r"(?<![\w/.])((?:iris|tests)/[\w/]+\.py)", read(f)):
            if not (REPO / m.group(1)).exists():
                stale.add(f"{f.relative_to(REPO)}: {m.group(1)}")
    assert not stale, f"docs mention files that don't exist: {sorted(stale)}"


# -- links ---------------------------------------------------------------------------------------
def _slug(heading: str) -> str:
    """GitHub's heading anchor: lowercase, punctuation dropped, spaces to hyphens."""
    h = re.sub(r"`([^`]*)`", r"\1", heading.strip().lower())
    h = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", h)
    h = re.sub(r"[^\w\- ]", "", h)
    return h.replace(" ", "-")


def _anchors(p: Path) -> set[str]:
    seen: dict[str, int] = {}
    out = set()
    in_code = False
    for line in read(p).splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if in_code or not line.startswith("#"):
            continue
        s = _slug(line.lstrip("#"))
        n = seen.get(s, 0)
        out.add(s if n == 0 else f"{s}-{n}")
        seen[s] = n + 1
    return out


def test_relative_links_and_anchors_resolve():
    broken = []
    for f in doc_files():
        text = re.sub(r"```.*?```", "", read(f), flags=re.S)
        text = re.sub(r"`[^`\n]*`", "", text)                # inline code ([word](/phonemes/) is not a link)
        for m in re.finditer(r"\]\(([^)\s]+)\)", text):
            target = m.group(1)
            if re.match(r"[a-z]+:", target) or target.startswith("#") and not target[1:]:
                continue
            path, _, anchor = target.partition("#")
            dest = (f.parent / path).resolve() if path else f
            if not dest.exists():
                broken.append(f"{f.relative_to(REPO)} → {target} (no such file)")
            elif anchor and dest.suffix == ".md" and anchor not in _anchors(dest):
                broken.append(f"{f.relative_to(REPO)} → {target} (no such heading)")
    assert not broken, "broken links:\n" + "\n".join(broken)


# -- references that must list everything ----------------------------------------------------------
def test_every_knob_and_config_section_is_in_the_configuration_reference():
    ref = read(ARCH / "13-configuration-reference.md")
    pod = yaml.safe_load(read(REPO / "self_template/pods/voice/pod.yaml"))
    policy = yaml.safe_load(read(REPO / "self_template/policy.yaml"))
    missing = [f"knob {k}" for k in pod["knobs"] if f"`{k}`" not in ref]
    missing += [f"pod.yaml {k}" for k in pod if f"`{k}`" not in ref]
    missing += [f"policy.yaml {k}" for k in policy if f"`{k}`" not in ref]
    used = set(re.findall(r'pod\.get\("([a-z_]+)"\)', iris_code()))
    missing += [f"pod.yaml {k} (read by the code)" for k in sorted(used) if f"`{k}`" not in ref]
    assert not missing, f"document in docs/architecture/13-configuration-reference.md: {missing}"


def test_every_environment_variable_is_documented():
    ref = read(ARCH / "13-configuration-reference.md")
    names = set(re.findall(r'environ(?:\.get)?\(?\[?"([A-Z][A-Z0-9_]+)"', iris_code()))
    missing = sorted(n for n in names if f"`{n}`" not in ref)
    assert not missing, f"document these environment variables in 13-configuration-reference.md: {missing}"


def test_every_cli_command_is_documented():
    ref = read(ARCH / "14-development.md")
    cmds = re.findall(r'add\("([a-z]+)"', read(REPO / "iris/cli.py"))
    cmds += re.findall(r'\(\("([a-z]+)", cmd_', read(REPO / "iris/cli.py"))      # talk / chat / ui share a loop
    missing = sorted({c for c in cmds if not re.search(rf"^\|.*`{c}[ `]", ref, re.M)})
    assert not missing, f"add these commands to the table in docs/architecture/14-development.md: {missing}"


def test_every_trace_event_is_documented():
    ref = read(ARCH / "11-observability.md")
    events = set(re.findall(r'\.event\("([a-z_]+)"', iris_code()))
    missing = sorted(e for e in events if f"`{e}`" not in ref)
    assert not missing, f"document these trace events in docs/architecture/11-observability.md: {missing}"


# -- ADRs ---------------------------------------------------------------------------------------
SECTIONS = ("## Context", "## Decision", "## Consequences", "## Alternatives considered", "## Evidence")
STATUS = re.compile(r"^- \*\*Status:\*\* (Proposed|Accepted|Rejected|Deprecated|Superseded by ADR-(\d{4}))\s*$", re.M)


def adr_files() -> list[Path]:
    return sorted(p for p in ADR.glob("[0-9][0-9][0-9][0-9]-*.md") if not p.name.startswith("0000"))


def test_adrs_are_numbered_once_and_complete():
    nums = [p.name[:4] for p in adr_files()]
    assert len(nums) == len(set(nums)), "two ADRs share a number"
    for p in adr_files():
        text = read(p)
        assert text.startswith(f"# ADR-{p.name[:4]}: "), f"{p.name}: title must be '# ADR-{p.name[:4]}: …'"
        m = STATUS.search(text)
        assert m, f"{p.name}: needs a '- **Status:** …' line with a valid status"
        if m.group(2):
            assert list(ADR.glob(f"{m.group(2)}-*.md")), f"{p.name}: superseded by a missing ADR-{m.group(2)}"
        for sec in SECTIONS:
            assert sec in text, f"{p.name}: missing section {sec}"
        assert re.search(r"^- \*\*Date:\*\* \d{4}-\d{2}-\d{2}", text, re.M), f"{p.name}: needs a Date"


def test_every_adr_is_in_the_index():
    index = read(ADR / "README.md")
    missing = [p.name for p in adr_files() if f"({p.name})" not in index]
    assert not missing, f"add to docs/adr/README.md: {missing}"


# -- changelog and variants -----------------------------------------------------------------------
def test_changelog_has_an_unreleased_section():
    assert "## [Unreleased]" in read(REPO / "CHANGELOG.md")


def test_a_variant_branch_documents_itself():
    try:
        branch = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip("git not available")
    if not branch.startswith("variant/"):
        pytest.skip("not on a variant branch")
    d = DOCS / "variants" / branch.split("/", 1)[1]
    assert (d / "README.md").exists() and (d / "CHANGELOG.md").exists(), \
        f"a variant documents itself in {d.relative_to(REPO)}/ (README.md, CHANGELOG.md; docs/README.md)"
