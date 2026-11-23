"""Which split a plan belongs to. The one definition train and test sets must share.

ResPlan and CubiCasa5K carry their official split in the record. Swiss Dwellings has none, and
apartments of one site (a building complex) are near-duplicates, so the split is by site:
the SHA-1 of the site id, as a hex number, modulo 20 gives a bucket; bucket 0 is val, bucket 1
is test and buckets 2 to 19 are train (90 / 5 / 5). The stage-1 training set was drawn with
this same function, so no test site can appear in it.
"""

from __future__ import annotations

import hashlib

from vlm_floorplan.record import PlanRecord

SWISS_BUCKETS = 20
SWISS_VAL_BUCKET = 0
SWISS_TEST_BUCKET = 1


def swiss_site(record_id: str) -> str:
    """The site id of a `swiss_dwellings/<site>/<apartment>` record id."""
    return record_id.split("/")[1]


def swiss_split(site_id: str) -> str:
    bucket = int(hashlib.sha1(site_id.encode()).hexdigest(), 16) % SWISS_BUCKETS
    if bucket == SWISS_VAL_BUCKET:
        return "val"
    return "test" if bucket == SWISS_TEST_BUCKET else "train"


def split_of(record: PlanRecord) -> str | None:
    """ "train", "val" or "test"; None for a source with no split (MSD)."""
    if record.source == "swiss_dwellings":
        return swiss_split(swiss_site(record.id))
    return record.split
