#!/usr/bin/env python3
"""Make a vision-only copy of an Ollama multimodal model: same weights, the audio encoder removed.

Why: gemma4 e4b's projector file holds a vision encoder (350 MiB) and an audio encoder (584 MiB); Ollama
loads both onto the GPU with the model. IRIS hears through its own speech recogniser, so the audio
encoder only costs memory (~0.7 GB with its working buffer on an 8 GB card). The copy keeps vision.

    uv run --no-project --with gguf python scripts/strip_mmproj_audio.py gemma4:e4b-it-qat gemma4:e4b-it-qat-vision

It reads the model's Modelfile from `ollama show --modelfile`, writes the projector without `a.*` /
`mm.a.*` tensors and `clip.audio.*` keys (`clip.has_audio_encoder` = false) to a temporary folder, and
runs `ollama create` with the same model weights, renderer, parser and parameters. The original model is
not touched; remove the copy with `ollama rm NEW`. Ollama de-duplicates the unchanged weights by digest,
so only the new projector (~360 MB) is added to the store.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import gguf


def is_audio_tensor(name: str) -> bool:
    return name.startswith(("a.", "mm.a."))


def is_audio_key(name: str) -> bool:
    return name.startswith("clip.audio.") or name == "clip.has_audio_encoder"


def strip(src: Path, dst: Path) -> tuple[int, int]:
    """Copy the GGUF at `src` to `dst` without the audio encoder. Returns (bytes kept, bytes removed)."""
    reader = gguf.GGUFReader(src)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF.") or is_audio_key(field.name):
            continue
        vtype = field.types[0]
        sub = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vtype, sub_type=sub)
    writer.add_bool("clip.has_audio_encoder", False)
    keep = [t for t in reader.tensors if not is_audio_tensor(t.name)]
    removed = sum(t.n_bytes for t in reader.tensors if is_audio_tensor(t.name))
    for t in keep:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in keep:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()
    return sum(t.n_bytes for t in keep), removed


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src_model, new_model = sys.argv[1], sys.argv[2]
    modelfile = subprocess.run(["ollama", "show", "--modelfile", src_model], capture_output=True, text=True, check=True).stdout
    froms = re.findall(r"^FROM (\S+)$", modelfile, re.M)
    proj = []
    for p in froms:
        if not Path(p).exists() or Path(p).stat().st_size > 3 * 2**30:     # the LLM weights, not a projector
            continue
        f = gguf.GGUFReader(p).fields.get("clip.has_audio_encoder")
        if f is not None and f.contents():
            proj.append(p)
    if not proj:
        print(f"{src_model} has no projector with an audio encoder (FROM lines: {froms})")
        return 1
    with tempfile.TemporaryDirectory(prefix="iris-mmproj-") as tmp:
        out = Path(tmp) / "mmproj-vision.gguf"
        kept, removed = strip(Path(proj[0]), out)
        print(f"projector: kept {kept / 2**20:.0f} MiB (vision), removed {removed / 2**20:.0f} MiB (audio)")
        new = re.sub(rf"^FROM {re.escape(proj[0])}$", f"FROM {out}", modelfile, flags=re.M)
        mf = Path(tmp) / "Modelfile"
        mf.write_text(new)
        subprocess.run(["ollama", "create", new_model, "-f", str(mf)], check=True)
    print(f"created {new_model}; use it with: iris set model llm {new_model}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
