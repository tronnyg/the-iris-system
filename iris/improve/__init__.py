"""Improvement loops (design 1: release engineering) and their helpers.

    sleep.py    after each session: memory deltas, episode summary, failure harvesting
    night.py    overnight: grading, bounded tuning with rollback, prompt proposals, scouting
    gate.py     the no-regression gate every candidate must pass
    review.py   approve / reject / rollback (the human side)
"""
