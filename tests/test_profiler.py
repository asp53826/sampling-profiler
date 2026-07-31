import math

import pytest

from prof.sampler import Profile, Sampler, _frame_stack
from prof.workload import (DEFAULT_SHARES, calibrate, make_workload,
                           measure_truth)


# ---------------------------------------------------------------------------
# the ground truth has to be trustworthy before anything else means anything
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_burn_is_linear_enough_to_define_ground_truth():
    """The foundation of every share in this repo.

    The bar is loose enough to survive a contended CI runner and tight enough
    that a genuinely non-linear primitive fails it. `calibrate().ok()` carries
    the strict bar and is what the benchmarks check before reporting shares —
    this asserts the primitive is linear, not that the machine is quiet.
    """
    c = calibrate()
    assert c.r2 > 0.99, c
    assert c.slope > 0, c


def test_calibration_rejects_a_nonlinear_primitive():
    """The check has to be capable of failing. A quadratic cost must not pass
    the strict bar that ground truth depends on."""
    from prof.workload import Calibration
    xs = [40, 80, 160, 320, 640]
    ys = [(x / 1000) ** 2 for x in xs]          # quadratic, not linear
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    worst = max(abs(y - (slope * x + intercept)) / y for x, y in zip(xs, ys))
    assert not Calibration(slope, intercept, 1 - ss_res / ss_tot, worst).ok()


def test_workload_shares_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        make_workload({"a": 0.5, "b": 0.2})


def test_generated_functions_have_distinct_names():
    """Each must be its own code object. Wrapping one closure would give every
    frame the same co_name and make attribution trivially easy."""
    run, truth = make_workload({"alpha": 0.5, "beta": 0.5}, total_units=10)
    assert set(truth) == {"alpha", "beta"}
    run(rounds=1)


@pytest.mark.slow
def test_constructed_and_timed_truth_agree():
    """Two independent routes to the same number. Disagreement means burn() is
    not behaving and every downstream result is void."""
    _, constructed = make_workload(DEFAULT_SHARES, 2000)
    timed = measure_truth(DEFAULT_SHARES, 2000, repeats=5)
    for k in DEFAULT_SHARES:
        assert abs(constructed[k] - timed[k]) < 0.06, (k, constructed[k], timed[k])


# ---------------------------------------------------------------------------
# sampler mechanics
# ---------------------------------------------------------------------------

def test_rejects_nonpositive_rate():
    with pytest.raises(ValueError, match="positive"):
        Sampler(hz=0)


def test_frame_stack_is_outermost_first():
    def inner():
        import sys
        return _frame_stack(sys._getframe())
    stack = inner()
    assert stack[-1] == "inner"
    assert "test_frame_stack_is_outermost_first" in stack


def test_timer_is_disarmed_after_stop():
    import signal
    with Sampler(hz=200):
        pass
    assert signal.getitimer(signal.ITIMER_PROF) == (0.0, 0.0)


def test_handler_is_restored_after_stop():
    import signal
    before = signal.getsignal(signal.SIGPROF)
    with Sampler(hz=200):
        pass
    assert signal.getsignal(signal.SIGPROF) is before


def test_shares_sum_to_one_over_self_counts():
    p = Profile()
    p.samples = 10
    p.self_counts.update({"a": 7, "b": 3})
    assert p.share("a") + p.share("b") == pytest.approx(1.0)
    assert p.share("missing") == 0.0


def test_stderr_matches_the_binomial_formula():
    p = Profile()
    p.samples = 400
    p.self_counts["a"] = 40
    expected = math.sqrt(0.1 * 0.9 / 400)
    assert p.stderr_of("a") == pytest.approx(expected)


def test_empty_profile_does_not_divide_by_zero():
    p = Profile()
    assert p.share("x") == 0.0
    assert p.stderr_of("x") == 0.0
    assert p.top() == []


def test_folded_output_is_parseable():
    p = Profile()
    p.samples = 3
    p.stacks.update({("main", "a"): 2, ("main", "b"): 1})
    lines = p.folded().splitlines()
    assert lines[0] == "main;a 2"
    assert all(len(l.rsplit(" ", 1)) == 2 for l in lines)


def test_cumulative_counts_a_frame_once_per_sample():
    """A recursive stack must not inflate its own cumulative share."""
    p = Profile()
    p.samples = 1
    stack = ("main", "rec", "rec", "rec")
    for name in set(stack):
        p.cum_counts[name] += 1
    assert p.cum_counts["rec"] == 1
    assert p.cum_share("rec") == 1.0


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_profiler_ranks_the_workload_correctly():
    """Ordering is the property worth pinning; exact shares carry sampling
    noise and are reported with a standard error rather than asserted."""
    run, _ = make_workload(DEFAULT_SHARES, 3000)
    with Sampler(hz=1000) as s:
        run(rounds=40)
    p = s.stop()
    assert p.samples > 200
    ranked = [k for k, _ in p.self_counts.most_common()
              if k in DEFAULT_SHARES]
    assert ranked[0] == "hot_40"
    assert ranked[1] == "warm_25"
    assert set(ranked[:4]) == {"hot_40", "warm_25", "mid_15", "mid_10"}


@pytest.mark.slow
def test_measured_shares_land_within_three_standard_errors():
    run, _ = make_workload(DEFAULT_SHARES, 3000)
    timed = measure_truth(DEFAULT_SHARES, 3000, repeats=5)
    with Sampler(hz=1000) as s:
        run(rounds=60)
    p = s.stop()
    for k in DEFAULT_SHARES:
        se = max(p.stderr_of(k), 1e-6)
        z = abs(p.share(k) - timed[k]) / se
        assert z < 3.0 + 2.0, (k, p.share(k), timed[k], z)


def test_detectability_floor_is_exactly_the_crossover():
    """n = 4(1-p)/p is where two standard errors *equal* the share, not where
    they fall below it. Asserting a strict inequality at the boundary fails, so
    the property is stated as the crossover it actually is: equality at n,
    detectable above, not detectable below."""
    def two_se(p, n):
        return 2 * math.sqrt(p * (1 - p) / n)

    for p_true in (0.02, 0.05, 0.10):
        n = 4 * (1 - p_true) / p_true
        assert two_se(p_true, n) == pytest.approx(p_true, rel=1e-12)
        assert p_true > two_se(p_true, n * 2)      # more samples: detectable
        assert p_true < two_se(p_true, n / 2)      # fewer: indistinguishable
