# SPDX-FileCopyrightText: Copyright (c) 2026 Everserve
# SPDX-License-Identifier: Apache-2.0
"""The money cap: three H100s for 14.6 GPU-hours bill $65 at the booked rate, over a $60 cap
that a watchdog time worked out from an assumed rate cannot hold."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nebius  # noqa: E402

RATE = 65 / 14.6  # the billed rate, everything included


def test_the_rental_s_spend_is_reckoned_from_its_start_at_the_billed_rate():
    hours = 14.6 / 3
    assert round(nebius.spend_usd(0, hours * 3600, 3, RATE), 2) == 65.0
    assert nebius.spend_usd(100, 50, 3, RATE) == 0.0  # a clock read before the start costs nothing


def test_the_cap_holds_back_the_minutes_the_releases_take():
    # three GPUs burn about $0.22 a minute at that rate: ten minutes' reserve is $2.23
    spent_at_cap = 60 - 3 * RATE * 10 / 60
    assert nebius.at_cap(spent_at_cap + 0.01, 3, RATE, 60, 10)
    assert not nebius.at_cap(spent_at_cap - 0.01, 3, RATE, 60, 10)


def test_the_run_1_rental_would_have_been_released_under_its_cap():
    """At the billed rate, with the releases' reserve, the cap stops the rental before $60."""
    gpus, t = 3, 0.0
    while not nebius.at_cap(nebius.spend_usd(0, t, gpus, RATE), gpus, RATE, 60, 10):
        t += 30
    assert nebius.spend_usd(0, t, gpus, RATE) + gpus * RATE * 10 / 60 <= 60 + gpus * RATE * 0.5 / 60
    assert t / 3600 < 14.6 / 3  # before the hours that billed $65


def test_a_vm_boots_from_its_own_region_s_public_images():
    assert nebius.public_images("project-e00aaaaaaaaaaaaaaaaaa") == "project-e00public-images"
    assert nebius.public_images("project-e04bbbbbbbbbbbbbbbbb") == "project-e04public-images"
