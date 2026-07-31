"""Where this profiler is structurally blind, measured rather than asserted.

Two limitations follow directly from how CPython delivers signals, and both are
invisible unless you go looking. A profiler that reports 100% of time in the
wrong place while looking perfectly healthy is worse than no profiler.

    uv run python bench/blindspots.py
"""

from __future__ import annotations

import threading
import time

from prof.sampler import Sampler
from prof.workload import _BODY


def hdr(t):
    print(f"\n=== {t} ===\n")


def pure_python(n: int) -> float:
    acc = 0.0
    for i in range(n):
        acc += (i & 0xFF) * 1.000001
    return acc


def uninterruptible(n: int) -> None:
    """One long C call. CPython cannot run a signal handler inside it, so every
    sample that should have landed here lands somewhere else instead."""
    # sorting a large list is a single bytecode that runs entirely in C
    data = list(range(n, 0, -1))
    data.sort()


def main():
    hdr("blind spot 1: time inside a single C call is unattributable")

    N_PY, N_C = 900_000, 3_000_000
    t0 = time.process_time(); pure_python(N_PY); t_py = time.process_time() - t0
    t0 = time.process_time(); uninterruptible(N_C); t_c = time.process_time() - t0
    true_c = t_c / (t_py + t_c)

    def mixed():
        for _ in range(6):
            pure_python(N_PY)
            uninterruptible(N_C)

    with Sampler(hz=1000) as s:
        mixed()
    p = s.stop()

    print(f"  measured directly:  python {t_py*1000:.0f} ms   C call {t_c*1000:.0f} ms")
    print(f"  true C share:       {true_c:.1%}")
    print(f"\n  {'frame':<22}{'samples':>9}{'share':>9}")
    print("  " + "-" * 40)
    for name, cnt in p.self_counts.most_common(6):
        print(f"  {name:<22}{cnt:>9}{cnt/p.samples:>8.1%}")
    got = p.share("uninterruptible")
    print(f"\n  profiler attributes {got:.1%} to the C-heavy function against a")
    print(f"  true {true_c:.1%}. The handler cannot run during `list.sort`, so the")
    print("  samples land on whichever Python frame resumes afterwards. This is")
    print("  not a tuning problem: it is what in-process signal sampling can see.")
    print("  Production Python profilers read the target from another process")
    print("  precisely to avoid it.")

    hdr("blind spot 2: only the main thread exists")

    done = threading.Event()

    def worker():
        for _ in range(6):
            pure_python(N_PY)
        done.set()

    ns: dict[str, object] = {}
    exec(_BODY.format(name="main_thread_work", n=N_PY * 3), ns)

    t = threading.Thread(target=worker)
    with Sampler(hz=1000) as s:
        t.start()
        ns["main_thread_work"]()
        t.join()
    p2 = s.stop()

    worker_share = p2.cum_share("worker") + p2.cum_share("pure_python")
    print(f"  a worker thread ran the whole time; samples attributed to it: "
          f"{worker_share:.1%}")
    print(f"  {'frame':<22}{'samples':>9}{'share':>9}")
    print("  " + "-" * 40)
    for name, cnt in p2.self_counts.most_common(4):
        print(f"  {name:<22}{cnt:>9}{cnt/p2.samples:>8.1%}")
    print("\n  CPython delivers signals only to the main thread, so a worker doing")
    print("  half the work is invisible. The profile is not wrong about the main")
    print("  thread — it is silent about everything else, which reads the same.")


if __name__ == "__main__":
    main()
