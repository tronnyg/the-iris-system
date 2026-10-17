"""The current date and time (models don't know it)."""

from __future__ import annotations

import datetime as dt

from . import Tool


def run(args: dict) -> str:
    now = dt.datetime.now().astimezone()
    return now.strftime("It is %A %d %B %Y, %H:%M (%Z, UTC%z).")


TOOL = Tool(
    name="current_time",
    description="The current local date, day of the week and time.",
    parameters={"type": "object", "properties": {}},
    run=run, text_form="<<time>>")
