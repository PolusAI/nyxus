"""Mechanics: a new instance must never inherit a dead instance's backend Environment.

Kind: *mechanics* per tests/vetting/SPEC.md 2 -- this asserts plumbing (which Environment an
instance is bound to), not a feature value.

The backend caches each instance's Environment in pynyxus_cache, keyed on id(self). id() is
the object's address, and CPython hands that address straight to the next object it allocates,
so an entry that outlives its object is not just a leak: findenv() returns the dead instance's
Environment to whoever lands on that address next. A fresh Nyxus would then silently start
with the previous one's ram_limit, feature list, gpu flag or ibsi mode.

This surfaced as a CI failure where an unrelated GABOR test raised "the following ROIS are
oversized and cannot be processed" -- it had inherited ram_limit=0 from an out-of-core test
whose instance had already been collected. Guards the __del__ that releases the entry.
"""
import nyxus
from test_data import intens, seg

# ram_limit=0 makes every ROI oversized (roiFootprint >= 0 is always true), so an inherited
# Environment is unmistakable: featurize raises instead of returning rows.
POISON_RAM_LIMIT = 0

# CPython reuses a freed address for the next same-sized allocation, but that is an
# implementation detail rather than a guarantee, so try several times and require that at
# least one round actually recycled -- otherwise the test would pass without exercising
# anything and quietly stop guarding the bug.
ROUNDS = 8


def test_environment_not_inherited_after_instance_dies_mechanics():
    recycled = False
    for _ in range(ROUNDS):
        doomed = nyxus.Nyxus(["MEAN"])
        doomed.set_params(ram_limit=POISON_RAM_LIMIT)
        poisoned_addr = id(doomed)
        del doomed

        fresh = nyxus.Nyxus(["GABOR"])
        recycled = recycled or id(fresh) == poisoned_addr

        # must compute normally: a fresh instance owns a fresh Environment whatever address
        # it was handed, so the inherited ram_limit=0 cannot reach it
        features = fresh.featurize(intens, seg)
        assert len(features) > 0
        del fresh

    assert recycled, (
        "CPython never reused the freed address across %d rounds, so this run did not "
        "actually exercise environment inheritance" % ROUNDS)
