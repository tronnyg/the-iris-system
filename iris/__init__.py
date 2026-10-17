"""IRIS: a local, self-improving voice assistant.

Kept import-light on purpose: the TTS workers run this package inside their own
virtualenvs (torch, model libraries) and must not pull in core-only dependencies.
"""

__version__ = "0.1.0"
