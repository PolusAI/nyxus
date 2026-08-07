"""Mechanics tests for the out-of-core (oversized-ROI) path.

Kind: *mechanics* per tests/vetting/SPEC.md 2 -- gating and plumbing, claiming nothing about feature
values. Where a code path has no out-of-core support at all it must fail loudly rather than emit a
silently wrong feature row. The trivial == out-of-core value invariant lives in test_ooc_invariant.py.

ram_limit is a process-global in Nyxus, so each test sets it explicitly to stay order-independent.
"""
import numpy as np
import pytest

import nyxus

# The "large" ram_limit standing for "not oversized". It only has to clear the 300x300 fixture below
# (a couple of MB) -- it must NOT be sized generously, because Nyxus rejects a limit above currently
# available RAM (see _set_ram_limit_mb).
RAM_LIMIT_LARGE_MB = 64


def _set_ram_limit_mb(nyx, mb):
    """Set the process-global ram_limit and verify it was accepted.

    Nyxus rejects a limit above the RAM currently available and keeps the previous value, reporting
    the refusal without raising. ram_limit being a process-global, a silently rejected setting leaves
    an earlier test's 0 or 1 MB limit in place, the "in-RAM" side of a comparison then runs
    out-of-core, and the test fails on an unrelated assertion. Read the value back so a refusal fails
    here, naming the cause."""
    nyx.set_params(ram_limit=mb)
    got = nyx.get_params("ram_limit")["ram_limit"]
    assert got == mb, (
        "ram_limit=%d MB was not accepted (still %d MB) -- Nyxus refuses a limit above available "
        "RAM. Free memory or lower RAM_LIMIT_LARGE_MB." % (mb, got)
    )


def test_ooc_montage_oversized_fails_loudly_mechanics():
    """The in-memory (montage) path has no out-of-core support, so an ROI whose footprint
    reaches ram_limit must fail loudly rather than emit a silent all-zero feature row."""
    Y, X = 300, 300
    xg = (np.arange(X) % 256).astype(np.uint32)
    yg = ((np.arange(Y) % 200) * 256).astype(np.uint32)
    inten = (1 + xg[None, :] + yg[:, None]).astype(np.uint32)
    mask = np.ones((Y, X), np.uint32)

    # sanity: with a large ram_limit the montage path succeeds and is non-zero. Set it
    # explicitly (nyxus ram_limit is process-global) so this does not depend on test order.
    ok = nyxus.Nyxus(["*ALL_INTENSITY*"])
    _set_ram_limit_mb(ok, RAM_LIMIT_LARGE_MB)
    df_ok = ok.featurize(inten, mask, intensity_names=["I"], label_names=["M"])
    assert df_ok["MEAN"].iloc[0] > 0

    # ram_limit=1 makes the single ROI oversized -> must raise, not return zeros
    n = nyxus.Nyxus(["*ALL_INTENSITY*"])
    _set_ram_limit_mb(n, 1)
    with pytest.raises(Exception, match="oversized"):
        n.featurize(inten, mask, intensity_names=["I"], label_names=["M"])
