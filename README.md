# sampling-profiler

A SIGPROF sampling profiler, scored against workloads whose true CPU share is
known — including the regimes where it is confidently and completely wrong.

[![CI](https://github.com/asp53826/sampling-profiler/actions/workflows/ci.yml/badge.svg)](https://github.com/asp53826/sampling-profiler/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square)
![runtime deps](https://img.shields.io/badge/runtime_dependencies-0-2ea44f?style=flat-square)

> Ask a profiler where the time went and it answers. The interesting question is
> when to believe it. This one reports a function holding **66.3%** of the
> runtime as **10.8%**, and a thread doing half the work as **0.0%** — both
> while looking entirely healthy.

## Ground truth first

A profiler cannot be scored without knowing the right answer. Everything here is
built from one primitive whose cost is linear in its argument, so a function
doing `n_i` units holds share `n_i / Σn`. Linearity is measured, not assumed:

```
burn() cost per unit    32.38 µs
linearity R²            0.999961
worst rel. residual     0.0193
```

Shares are then established two independent ways — by construction and by
directly timing each function in isolation — and the benchmark reports their
disagreement rather than inheriting it silently.

## Measured, not implied

### The requested sampling rate is a fiction

| requested | achieved | ratio | overhead |
|---:|---:|---:|---:|
| 100 Hz | 91 | **0.91** | 1.2% |
| 250 Hz | 127 | 0.51 | 0.6% |
| 500 Hz | 182 | 0.36 | 2.6% |
| 1000 Hz | 281 | 0.28 | 4.1% |
| 2000 Hz | 257 | 0.13 | 3.2% |
| 5000 Hz | 283 | **0.06** | 1.7% |

The platform tops out near 280 Hz. `ITIMER_PROF` fires on consumed CPU time, but
CPython only runs the handler at a bytecode boundary and the OS timer has its own
granularity. **Asking for 5000 Hz buys overhead, not samples** — the ratio column
is the one that matters, and no profiler that reports only its requested rate
will tell you this.

### Error is noise-limited, then bias-limited

| samples | mean abs error | binomial prediction | ratio |
|---:|---:|---:|---:|
| 507 | 0.0060 | 0.0134 | 0.45× |
| 1,669 | 0.0064 | 0.0074 | 0.87× |
| 6,833 | 0.0017 | 0.0036 | 0.47× |
| 18,214 | 0.0028 | 0.0022 | **1.27×** |

Below roughly 10,000 samples the error sits under the binomial prediction and
falls as samples accumulate — the estimator is behaving. Past that the ratio
crosses 1 and the error stops falling: a systematic floor around **0.003** takes
over and more samples stop buying accuracy.

Not claimed: that the floor is all profiler bias. The ground truth is itself a
timing measurement with its own variance, and at three tenths of a percentage
point the two cannot be separated here.

### The detectability floor is computable

A share `p` clears two standard errors — the usual bar for "not noise" — only
once `n = 4(1−p)/p`. That is an equality at the boundary, not an inequality, and
the test says so.

| true share | samples needed | seconds at 280 Hz |
|---:|---:|---:|
| 10% | 36 | 0.1s |
| 3% | 130 | 0.5s |
| 1% | 396 | 1.5s |
| 0.1% | 3,996 | 14.7s |

Verified empirically: with 1,835 samples all seven functions from 40% down to 2%
clear the bar, and their measured shares land within a standard error or two of
the timed truth.

## Where it loses

Both of these are structural. Neither is a tuning problem.

**Time inside a single C call is unattributable.** A workload split between pure
Python and one long `list.sort`:

| | true | reported |
|---|---:|---:|
| C-heavy function | **66.3%** | **10.8%** |

CPython cannot run a signal handler inside a single C call, so samples that
belong there land on whichever Python frame resumes afterwards. The profile is
not noisy — it is confidently wrong by a factor of six, and nothing in the output
hints at it. Production Python profilers read the target from a separate process
precisely to avoid this.

**Only the main thread exists.** A worker thread running for the entire profile
received **0.0%** of samples. CPython delivers signals only to the main thread,
so a thread doing half the work is invisible, and silence reads identically to
"spent no time here".

Also true, and reported rather than hidden:

- **Stack walking is the interpreter's, not mine.** `sys._current_frames` gives
  the frame chain. Every in-process Python profiler gets that for free; the
  timer management, signal-safe capture, aggregation, folded output and the whole
  error analysis are the parts built here. Claiming to have written an unwinder
  would be the interesting part of the claim being false.
- **CPU time, not wall time.** `ITIMER_PROF` does not tick while blocked, so time
  in I/O is absent by construction. Correct for finding compute hot spots, wrong
  for finding latency.
- **One machine, one platform.** The 280 Hz ceiling is macOS on Apple silicon.
  The shape of the result should transfer; the number should not.

## Verify it

```bash
make test        # 16 tests
make accuracy    # ground truth, rate ceiling, error scaling, detectability floor
make blindspots  # the two regimes where the answer is wrong
```

## Use it

```python
from prof.sampler import Sampler

with Sampler(hz=500) as s:
    do_work()
p = s.stop()

p.top(10)              # [(name, share), ...] by self time
p.share("parse")       # self-time share
p.cum_share("parse")   # including everything called beneath it
p.stderr_of("parse")   # how much of that is noise
print(p.folded())      # folded stacks, ready for a flamegraph
```

`stderr_of` exists because a share without its standard error is an opinion.

## License

MIT
