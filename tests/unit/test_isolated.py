"""tests/isolated.py: peak RSS of one call, in a fresh process.

The timing tests compare gfem's peak memory with Palace's whole-process
number, so the call they gate must not inherit the high-water mark of the
tests that ran before it in the same pytest process.

Each check runs twice: with the kernel's own high-water mark where there is
one (Linux ``VmHWM``), and with the sampled path that kernels without it
(gVisor, Modal's sandbox) take.
"""

import time

import isolated as isolated_module
import numpy as np
import pytest
from isolated import isolated, peak_rss_gib

MIB = 2**20
#: How long `_hold` keeps its block. The sampled path sees a peak only if the
#: caller gets the CPU while it lasts; on a loaded host (ostau, run queue 8-22
#: on 8 vCPUs) the caller went up to 0.32 s without a sample.
HOLD = 0.5


@pytest.fixture(autouse=True, params=["kernel", "sampled"])
def mode(request, monkeypatch):
    monkeypatch.setattr(isolated_module, "SAMPLED_ONLY", request.param == "sampled")


def _touch(mib):
    """Allocate and write ``mib`` MiB, so the pages count in the RSS."""
    block = np.ones(mib * MIB, dtype=np.uint8)
    return int(block[::4096].sum())


def _hold(mib):
    """`_touch`, keeping the block for `HOLD` seconds: a peak a sampler can see."""
    block = np.ones(mib * MIB, dtype=np.uint8)
    time.sleep(HOLD)
    return int(block[::4096].sum())


def _fail():
    raise KeyError("inside the child")


def test_returns_the_value_of_the_call():
    value, _ = isolated(_touch, 1)
    assert value == MIB // 4096


def test_peak_counts_what_the_call_allocates():
    _, rss = isolated(_hold, 300)
    assert rss >= 300 / 1024


def test_peak_ignores_the_callers_high_water_mark():
    _touch(400)
    parent = peak_rss_gib()
    assert parent >= 400 / 1024
    _, rss = isolated(_touch, 1)
    assert rss < parent - 300 / 1024


def test_the_calls_exception_reaches_the_caller():
    with pytest.raises(KeyError, match="inside the child"):
        isolated(_fail)
