"""A statistical sampling profiler built on SIGPROF.

An interval timer fires on consumed CPU time; the handler walks the current
stack and records it. Nothing is instrumented, so the cost is proportional to
the sampling rate rather than to how many calls the program makes — which is
the whole reason sampling profilers exist.

What is genuinely built here: the timer management, the signal-safe capture
path, stack aggregation into self and cumulative time, the folded/flamegraph
output, and the entire error analysis in `analysis.py`. What is not: walking the
frame chain, which the interpreter exposes directly via `sys._current_frames`.
Any in-process Python profiler gets that for free, and pretending otherwise
would be the interesting part of the claim being false.
"""

from __future__ import annotations

import collections
import signal
import sys
import time
from dataclasses import dataclass, field
from types import FrameType


Stack = tuple[str, ...]


@dataclass
class Profile:
    """Aggregated samples.

    `self_counts` attributes a sample to the frame that was executing.
    `cum_counts` attributes it to every frame on the stack, which is what
    answers "how much time is spent under this call" as opposed to "in it".
    """

    self_counts: collections.Counter = field(default_factory=collections.Counter)
    cum_counts: collections.Counter = field(default_factory=collections.Counter)
    stacks: collections.Counter = field(default_factory=collections.Counter)
    samples: int = 0
    dropped: int = 0
    interval: float = 0.0
    wall: float = 0.0

    def share(self, name: str) -> float:
        """Fraction of samples whose innermost frame was `name`."""
        return self.self_counts[name] / self.samples if self.samples else 0.0

    def cum_share(self, name: str) -> float:
        return self.cum_counts[name] / self.samples if self.samples else 0.0

    def top(self, n: int = 12) -> list[tuple[str, float]]:
        return [(k, v / self.samples) for k, v in self.self_counts.most_common(n)]

    def folded(self) -> str:
        """Brendan Gregg's folded-stack format, the input flamegraphs take."""
        return "\n".join(f"{';'.join(s)} {c}" for s, c in
                         sorted(self.stacks.items(), key=lambda kv: -kv[1]))

    def stderr_of(self, name: str) -> float:
        """Binomial standard error on a measured share.

        A share is a proportion estimated from `samples` Bernoulli draws, so it
        carries sqrt(p(1-p)/n) of noise whether or not anyone reports it. This
        is what makes the detectability floor in `analysis.py` computable rather
        than a matter of taste.
        """
        if not self.samples:
            return 0.0
        p = self.share(name)
        return (p * (1 - p) / self.samples) ** 0.5


def _frame_stack(frame: FrameType | None, limit: int = 128) -> Stack:
    """Innermost-last list of qualified frame names."""
    out: list[str] = []
    depth = 0
    while frame is not None and depth < limit:
        code = frame.f_code
        out.append(f"{code.co_name}")
        frame = frame.f_back
        depth += 1
    out.reverse()
    return tuple(out)


class Sampler:
    """Context manager that profiles the main thread while active.

    Only the main thread is sampled, because CPython only delivers signals
    there. That is a real limitation, is measured in `bench/blindspots.py`, and
    is the reason production Python profilers read the target out of process.
    """

    def __init__(self, hz: float = 1000.0, prefix_drop: int = 0):
        if hz <= 0:
            raise ValueError("hz must be positive")
        self.hz = hz
        self.interval = 1.0 / hz
        self.prefix_drop = prefix_drop
        self.profile = Profile(interval=self.interval)
        self._prev_handler = None
        self._t0 = 0.0

    # -- signal path --------------------------------------------------------

    def _handler(self, signum, frame):
        # Kept deliberately small. Everything here runs between two bytecodes of
        # whatever the program was doing, and anything expensive shows up as
        # observer overhead in the results.
        try:
            stack = _frame_stack(frame)
            if self.prefix_drop:
                stack = stack[self.prefix_drop:]
            if not stack:
                self.profile.dropped += 1
                return
            p = self.profile
            p.samples += 1
            p.stacks[stack] += 1
            p.self_counts[stack[-1]] += 1
            for name in set(stack):
                p.cum_counts[name] += 1
        except Exception:
            self.profile.dropped += 1

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> "Sampler":
        if sys.platform == "win32":
            raise RuntimeError("SIGPROF is not available on Windows")
        self._prev_handler = signal.signal(signal.SIGPROF, self._handler)
        # ITIMER_PROF counts CPU time consumed by the process, not wall clock,
        # so a sleeping program is not sampled and does not dilute the shares.
        signal.setitimer(signal.ITIMER_PROF, self.interval, self.interval)
        self._t0 = time.perf_counter()
        return self

    def stop(self) -> Profile:
        signal.setitimer(signal.ITIMER_PROF, 0.0, 0.0)
        if self._prev_handler is not None:
            signal.signal(signal.SIGPROF, self._prev_handler)
        self.profile.wall = time.perf_counter() - self._t0
        return self.profile

    def __enter__(self) -> "Sampler":
        return self.start()

    def __exit__(self, *_):
        self.stop()
        return False
