from datetime import datetime

import pandas as pd
import pytest

from pyqualw2.utils import date_to_jday


@pytest.mark.parametrize(
    ("date", "expected"),
    [
        ("1921-01-01", 0),
        (datetime(1921, 1, 1), 0),
        ("2000-01-01", 28854),
        (datetime(2000, 1, 1), 28854),
    ],
)
def test_date_to_jday(date, expected):
    """Test that scalar dates can be converted to julian days."""
    assert date_to_jday(date) == expected


def test_date_to_jday_series():
    """Test that date dtype series can be converted to julian days."""
    series = pd.Series(["1921-01-01", "2000-01-01", "2000-01-01T12:00:00"])
    series2 = pd.to_datetime(series, format="ISO8601")

    expected = pd.Series([0, 28854, 28854.5], dtype=float)

    pd.testing.assert_series_equal(date_to_jday(series), expected)
    pd.testing.assert_series_equal(date_to_jday(series2), expected)
