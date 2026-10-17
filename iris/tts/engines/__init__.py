"""TTS engines. Each runs inside its own virtualenv (see install/*.sh).

An engine is a module exposing `Engine` with:
    name: str
    sr: int                                  sample rate after load()
    load(options: dict) -> dict              returns info (device, voices, notes)
    synth(text: str, speed: float, opts: dict) -> numpy float32 mono array
`text` may contain delivery cues ([pause], [dry], ...); `opts["delivery"]` says how
this engine renders them. Adding a new voice model = adding one module here plus an
install script. Nothing else in IRIS needs to change.
"""

ENGINES = ("kokoro", "chatterbox")
