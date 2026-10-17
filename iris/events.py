"""A tiny in-process event bus: the live session publishes, UIs subscribe.

Publishing never blocks (it is called from audio callbacks and the LLM thread): each
subscriber has a bounded queue, and when a slow subscriber's queue is full its oldest
event is dropped. Level/state frames are coalesced the same way.
"""

from __future__ import annotations

import queue
import threading
import time


class Bus:
    def __init__(self, maxsize: int = 512):
        self.maxsize = maxsize
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self.dropped = 0

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(self.maxsize)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, kind: str, **data) -> None:
        ev = {"type": kind, "t": round(time.time(), 3), **data}
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                try:
                    q.get_nowait()
                    self.dropped += 1
                    q.put_nowait(ev)
                except (queue.Empty, queue.Full):
                    pass

    @property
    def subscribers(self) -> int:
        with self._lock:
            return len(self._subs)
