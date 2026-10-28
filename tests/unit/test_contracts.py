"""The shared contracts agree with their stored forms."""

from __future__ import annotations

from typing import get_args

from langchain_sync_monitors.contracts import Outcome, OutcomeName


def test_outcome_values_match_the_stored_outcome_names() -> None:
    # Act
    enum_values = {outcome.value for outcome in Outcome}

    # Assert
    assert enum_values == set(get_args(OutcomeName))
