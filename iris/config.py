"""Read the self store into one settings object, and edit knobs in place.

Knob edits rewrite only the `value:` inside that knob's one-line entry in pod.yaml, so
your comments and layout survive automatic tuning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import paths


def _load(rel: str, root: Path | None = None) -> dict:
    p = (root or paths.self_dir()) / rel
    try:
        return yaml.safe_load(p.read_text()) or {}
    except FileNotFoundError:
        return {}


@dataclass
class Settings:
    persona: dict
    pod: dict
    policy: dict
    pronunciations: dict[str, str]
    root: Path
    overrides: dict[str, Any] = field(default_factory=dict)
    pron_rules: dict[str, list[str]] = field(default_factory=dict)

    persona_name: str = "iris"

    @classmethod
    def load(cls, root: Path | None = None, overrides: dict[str, Any] | None = None) -> "Settings":
        root = root or paths.self_dir()
        pdoc = _load("pods/voice/pronunciations.yaml", root)
        pron = pdoc.get("entries") or {}
        rules = {str(k): list(v or []) for k, v in (pdoc.get("rules") or {}).items()}
        pname = str(_load("agent.yaml", root).get("persona") or "iris")
        return cls(
            persona_name=pname,
            persona=_load(f"persona/{pname}.yaml", root),
            pod=_load("pods/voice/pod.yaml", root),
            policy=_load("policy.yaml", root),
            pronunciations={str(k): str(v) for k, v in pron.items()},
            pron_rules=rules,
            root=root,
            overrides=dict(overrides or {}),
        )

    # -- knobs and models -------------------------------------------------------------
    def knob(self, name: str, default: Any = None) -> Any:
        if name in self.overrides:
            return self.overrides[name]
        k = (self.pod.get("knobs") or {}).get(name)
        return default if k is None else k.get("value", default)

    def knob_spec(self, name: str) -> dict:
        return dict((self.pod.get("knobs") or {}).get(name) or {})

    def knobs(self) -> dict[str, Any]:
        out = {n: s.get("value") for n, s in (self.pod.get("knobs") or {}).items()}
        out.update({k: v for k, v in self.overrides.items() if k in out})
        return out

    def model(self, name: str, default: Any = None) -> Any:
        key = f"model.{name}"
        if key in self.overrides:
            return self.overrides[key]
        m = (self.pod.get("models") or {}).get(name)
        return default if m is None else m.get("value", default)

    def model_spec(self, name: str) -> dict:
        return dict((self.pod.get("models") or {}).get(name) or {})

    def tts_options(self, engine: str) -> dict:
        return dict((self.pod.get("tts") or {}).get(engine) or {})

    def target(self, name: str, default: Any = None) -> Any:
        return (self.pod.get("targets") or {}).get(name, default)

    # -- persona ----------------------------------------------------------------------
    @property
    def persona_path(self) -> str:
        """Self-store path of the active persona file (what prompt proposals edit)."""
        return f"persona/{self.persona_name}.yaml"

    @property
    def display_name(self) -> str:
        return str(self.persona.get("name") or paths.agent())

    @property
    def system_prompt(self) -> str:
        raw = self.overrides.get("persona.system_prompt") or self.persona.get("system_prompt", "")
        # {name} and {model} placeholders let one persona describe whatever body it runs on
        return (str(raw).replace("{name}", self.display_name)
                .replace("{model}", str(self.model("llm", "a local model"))))

    def cue_vocab(self) -> tuple[list[str], list[str]]:
        cues = self.persona.get("cues") or {}
        return list(cues.get("tones") or []), list(cues.get("events") or [])

    def delivery(self, engine: str) -> dict:
        return dict((self.persona.get("delivery") or {}).get(engine) or {})

    # -- policy -----------------------------------------------------------------------
    def pol(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self.policy
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur


_num = r"-?\d+(?:\.\d+)?"


def set_value(section: str, name: str, value: Any, root: Path | None = None) -> bool:
    """Change `value:` of one entry in pod.yaml (`section` = knobs | models).

    Returns True if the file changed. Values are validated against min/max/choices.
    """
    root = root or paths.self_dir()
    path = root / "pods/voice/pod.yaml"
    text = path.read_text()
    data = yaml.safe_load(text) or {}
    spec = (data.get(section) or {}).get(name)
    if spec is None:
        raise KeyError(f"{section}.{name} not in pod.yaml")
    if "choices" in spec and value not in spec["choices"]:
        raise ValueError(f"{name}={value!r} not in choices {spec['choices']}")
    if "min" in spec and value < spec["min"]:
        raise ValueError(f"{name}={value} below min {spec['min']}")
    if "max" in spec and value > spec["max"]:
        raise ValueError(f"{name}={value} above max {spec['max']}")
    if spec.get("value") == value:
        return False
    rendered = _render(value)
    pattern = re.compile(rf"^(\s*{re.escape(name)}\s*:\s*\{{\s*value:\s*)([^,}}]+)", re.M)
    new, n = pattern.subn(lambda m: m.group(1) + rendered, text, count=1)
    if n != 1:
        raise RuntimeError(f"could not locate one-line entry for {name} in pod.yaml")
    path.write_text(new)
    return True


def _render(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}" if abs(value) >= 1e-4 else repr(value)
    if isinstance(value, int):
        return str(value)
    s = str(value)
    return f'"{s}"' if re.search(r"[:#,{}\[\]]", s) or s.strip() != s else s


def clamp_step(spec: dict, current: Any, direction: int) -> Any:
    """Next value one step up (+1) or down (-1) inside bounds; None if already at a bound."""
    if "choices" in spec:
        ch = list(spec["choices"])
        i = ch.index(current) if current in ch else 0
        j = i + direction
        return ch[j] if 0 <= j < len(ch) else None
    step = spec.get("step", 1)
    new = current + direction * step
    if isinstance(step, float) or isinstance(current, float):
        new = round(new, 4)
    lo, hi = spec.get("min"), spec.get("max")
    if lo is not None and new < lo:
        return None
    if hi is not None and new > hi:
        return None
    return new
