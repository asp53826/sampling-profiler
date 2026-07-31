"""Workloads whose true CPU share is known, because that is the only way to
say whether a profiler is right.

Everything is built from one calibrated primitive: `burn(units)` performs a
fixed amount of arithmetic per unit. If its cost is linear in `units` then a
function calling `burn(n_i)` holds share `n_i / sum(n)`, and the profiler can be
scored against that.

Linearity is not assumed. `calibrate()` measures it and reports the residual,
because a ground truth nobody checked is just another estimate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


def burn(units: int) -> float:
    """Deterministic arithmetic, linear in `units`.

    Pure Python on purpose: it must be interruptible between bytecodes so the
    SIGPROF handler can actually run. A numpy or C call of the same duration is
    a single uninterruptible operation and is exactly the blind spot measured in
    `bench/blindspots.py`.
    """
    acc = 0.0
    for i in range(units * 1000):
        acc += (i & 0xFF) * 1.000001
    return acc


_BODY = """
def {name}():
    acc = 0.0
    for i in range({n}):
        acc += (i & 0xFF) * 1.000001
    return acc
"""


@dataclass
class Calibration:
    slope: float          # seconds per unit
    intercept: float
    r2: float
    max_rel_residual: float

    def ok(self, tol: float = 0.05) -> bool:
        """Whether this machine can support the benchmark's ground truth.

        Deliberately a property of the measurement, not a constant: on hardware
        too noisy to calibrate, the shares below are unreliable and the honest
        answer is to say so rather than report them anyway.
        """
        return self.r2 > 0.999 and self.max_rel_residual < tol


def calibrate(units=(40, 80, 160, 320, 640), repeats: int = 9) -> Calibration:
    """Least-squares fit of runtime against units, with the worst residual.

    If this is not linear, every 'true share' below is wrong and the whole
    benchmark is measuring nothing.

    `min` over many repeats rather than a mean: timing noise on a shared machine
    is one-sided — contention only ever makes a measurement slower — so the
    minimum is the robust estimator and more repeats strictly improve it. An
    earlier version used 3 repeats and the smallest unit, and on a contended CI
    runner that reported R² 0.9955 with a 31% worst residual against 0.9999 and
    1.9% on an idle laptop. The fit was fine; the estimator was not.
    """
    xs, ys = [], []
    for u in units:
        best = min(_time_once(u) for _ in range(repeats))
        xs.append(float(u))
        ys.append(best)

    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx

    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    worst = max(abs(y - (slope * x + intercept)) / y for x, y in zip(xs, ys))
    return Calibration(slope, intercept, r2, worst)


def _time_once(units: int) -> float:
    t0 = time.process_time()
    burn(units)
    return time.process_time() - t0


# ---------------------------------------------------------------------------
# building a workload with named functions at known shares
# ---------------------------------------------------------------------------

def make_workload(shares: dict[str, float], total_units: int = 4000):
    """Return (run, truth) for functions holding the given shares.

    Functions are generated with `exec` so each gets a distinct code object and
    a real name the profiler must recover — wrapping one closure in different
    variables would give every frame the same `co_name` and quietly make the
    attribution problem trivial.
    """
    if abs(sum(shares.values()) - 1.0) > 1e-6:
        raise ValueError(f"shares must sum to 1, got {sum(shares.values())}")

    ns: dict[str, object] = {}
    units = {}
    for name, frac in shares.items():
        u = max(1, round(total_units * frac))
        units[name] = u
        # The loop is inlined into each generated function rather than delegated
        # to burn(). A function that only *calls* burn has no self time — the
        # innermost frame is always burn — so the profiler was correct to give
        # every workload function 0.000 and the workload was the thing that was
        # wrong. Self-time attribution can only be scored against functions that
        # actually execute something.
        exec(_BODY.format(name=name, n=u * 1000), ns)

    total = sum(units.values())
    truth = {name: u / total for name, u in units.items()}

    fns = [ns[name] for name in shares]

    def run(rounds: int = 1):
        for _ in range(rounds):
            for f in fns:
                f()

    return run, truth


def measure_truth(shares: dict[str, float], total_units: int = 4000,
                  repeats: int = 3) -> dict[str, float]:
    """Ground truth by direct timing rather than by construction.

    The constructed shares assume `burn` is perfectly linear. This measures each
    function in isolation and normalises, so the two can be compared and any
    disagreement surfaces instead of being inherited silently.
    """
    ns: dict[str, object] = {}
    times = {}
    for name, frac in shares.items():
        u = max(1, round(total_units * frac))
        exec(_BODY.format(name=name, n=u * 1000), ns)
        f = ns[name]
        best = float("inf")
        for _ in range(repeats):
            t0 = time.process_time()
            f()
            best = min(best, time.process_time() - t0)
        times[name] = best
    tot = sum(times.values())
    return {k: v / tot for k, v in times.items()}


# a workload spanning two decades of share, so the detectability floor is inside
# the range rather than off the end of it
DEFAULT_SHARES = {
    "hot_40": 0.40,
    "warm_25": 0.25,
    "mid_15": 0.15,
    "mid_10": 0.10,
    "cool_05": 0.05,
    "faint_03": 0.03,
    "trace_02": 0.02,
}
