import datetime
from typing import overload

import numpy as np
import pandas as pd

JULIAN_REFERENCE_START = datetime.datetime(1921, 1, 1)


@overload
def jday_to_date(jday: float) -> datetime.datetime: ...


@overload
def jday_to_date(jday: pd.Series) -> pd.Series: ...


def jday_to_date(jday):
    """Convert a fractional julian day to a datetime using JULIAN_REFERENCE_START.

    Parameters
    ----------
    jday : float | pd.Series
        Julian day to convert

    Returns
    -------
    datetime.datetime | pd.Series
        Datetime  associated with the input julian day
    """
    if isinstance(jday, float):
        return JULIAN_REFERENCE_START + datetime.timedelta(days=jday)

    return pd.to_datetime(JULIAN_REFERENCE_START) + pd.to_timedelta(jday, "days")


def to_fractional_days(series: pd.Series) -> pd.Series:
    """Convert a timedelta64 pd.Series to a floating point fractional day.

    Parameters
    ----------
    series : pd.Series
        Input series of dtype timedelta64

    Returns
    -------
    pd.Series
        The input timedelta, but in fractions of a day (floating point)
    """
    if series.dtype.type == np.timedelta64:
        return series / pd.to_timedelta(1, "days")
    raise ValueError(
        f"Cannot convert series of dtype {series.dtype} to fractional days"
    )
