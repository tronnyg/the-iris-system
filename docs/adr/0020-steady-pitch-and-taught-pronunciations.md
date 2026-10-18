# ADR-0020: One steady pitch; pronunciations taught by voice

- **Status:** Accepted
- **Date:** 2026-10-18
- **Supersedes:** none
- **Related:** [architecture/04](../architecture/04-speaking.md)

## Context

For expressiveness, the voice drifted pitch per sentence (up to ±1.5 semitones, questions lifted); the
user heard "sometimes deeper, sometimes higher". Separately "cumin" was mispronounced for nine turns
while the model promised to "adjust its pronunciation", which it could not do.

## Decision

Pitch changes only when the user asks (`tts.pitch_semitones`); per-sentence variety is pace only (±3 % at
most). Contradictory voice remarks change nothing. Pronunciations are taught by voice ("the word cumin is
pronounced koo-min"): a plain-English respelling is converted to misaki phonemes and spliced in as
`[word](/phonemes/)` for Kokoro (spoken as the respelling for other engines), saved in
`pronunciations.yaml: respell`, applied at once and spoken back.

## Consequences

* One recognisable voice; real control over pronunciations; promises about pronunciation are flagged.
* By voice the first syllable is stressed (speech recognition gives no capitals); typed input can mark stress.

## Alternatives considered

* Plain respelling text to Kokoro: hyphenated respellings break its G2P. Kokoro stress markup: ADR-0026.

## Evidence

Median-F0 spread across sentences on real Kokoro audio: 3.1 → 1.4 semitones (the rest is Kokoro's own
intonation). misaki: "Cumin" → `kˈʌmən` by default, `kˈumɪn` with the override. Commit 84f296d.
