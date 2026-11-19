"""ML-02 T08 (unitaria): intervalos fijos y supresión del histograma de renta."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from thehouse_ml.rent_histogram import (
    BIN_COUNT,
    RentBin,
    bins_from_frame,
    build_bins,
    group_small_bins,
    rows_to_frame,
)
from thehouse_ml.thresholds import MIN_DWELLINGS


def test_bin_labels_are_fixed_and_the_last_one_is_open() -> None:
    bins = build_bins("nacional", [100.0, 700.0, 20_000.0], [10.0, 20.0, 30.0], 1.0)

    assert len(bins) == BIN_COUNT
    assert bins[0].label == "0-499"
    assert bins[1].label == "500-999"
    assert bins[-1].label == "10,000+"
    assert bins[-1].upper is None
    assert bins[0].n_muestra == 1
    assert bins[1].n_muestra == 1
    assert bins[-1].n_muestra == 1


def test_expanded_frequency_is_the_sum_of_the_weights() -> None:
    bins = build_bins("13", [1000.0, 1100.0, 6000.0], [3.0, 7.0, 11.0], 1.0)

    assert sum(bin_.frecuencia_expandida for bin_ in bins) == pytest.approx(21.0)
    assert sum(bin_.n_muestra for bin_ in bins) == 3


def test_deflator_moves_a_rent_to_a_higher_bin() -> None:
    bins = build_bins("nacional", [1000.0], [5.0], 2.0)

    assert bins[4].label == "2,000-2,499"
    assert bins[4].n_muestra == 1
    assert bins[4].frecuencia_expandida == 5.0


def test_group_small_bins_keeps_every_cell_above_the_minimum() -> None:
    bins = build_bins("nacional", [1000.0] * (MIN_DWELLINGS + 1), [2.0] * (MIN_DWELLINGS + 1), 1.0)

    grouped = group_small_bins(bins)

    assert len(grouped) == 1
    assert grouped[0].n_muestra == MIN_DWELLINGS + 1
    assert grouped[0].label == "1,000-1,499"


def test_group_small_bins_preserves_the_expanded_total() -> None:
    rents = [100.0] * 5 + [1000.0] * 40 + [2000.0] * 10 + [20_000.0] * 3
    weights = [1.5] * len(rents)
    bins = build_bins("nacional", rents, weights, 1.0)

    grouped = group_small_bins(bins)

    assert all(bin_.n_muestra >= MIN_DWELLINGS for bin_ in grouped)
    assert sum(bin_.frecuencia_expandida for bin_ in grouped) == pytest.approx(
        sum(bin_.frecuencia_expandida for bin_ in bins)
    )
    assert sum(bin_.n_muestra for bin_ in grouped) == len(rents)


def test_group_small_bins_suppresses_a_tiny_sample() -> None:
    bins = build_bins("13", [1000.0] * (MIN_DWELLINGS - 1), [1.0] * (MIN_DWELLINGS - 1), 1.0)

    assert group_small_bins(bins) == []


def test_frame_round_trip_preserves_the_open_ended_bin() -> None:
    bins = [RentBin("nacional", lower=10_000.0, upper=None, n_muestra=40, frecuencia_expandida=99.0)]

    frame: pd.DataFrame = rows_to_frame(bins)
    restored = bins_from_frame(frame)

    assert math.isnan(float(frame.iloc[0]["upper"]))
    assert restored[0].label == "10,000+"
    assert restored[0].frecuencia_expandida == 99.0


def test_build_bins_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="misma longitud"):
        build_bins("nacional", [1000.0], [1.0, 2.0], 1.0)
