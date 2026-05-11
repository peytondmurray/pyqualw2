from datetime import datetime, timedelta
from os import PathLike
from pathlib import Path
from typing import overload

import pandas as pd

JULIAN_REFERENCE_START = datetime(1921, 1, 1)


def get_path_relative_to_home(path: PathLike | str) -> str:
    """Return a path relative to home as a string.

    Parameters
    ----------
    path : PathLike | str
        Path to convert to a relative path

    Returns
    -------
    str
        New path relative to the home directory, if possible; otherwise,
        return the original path
    """
    try:
        return f"~/{Path(path).relative_to(Path.home())}"
    except ValueError:
        return str(path)


@overload
def jday_to_date(jday: float) -> datetime: ...


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
        Datetime associated with the input julian day
    """
    if isinstance(jday, float):
        return JULIAN_REFERENCE_START + timedelta(days=jday)

    return pd.to_datetime(JULIAN_REFERENCE_START) + pd.to_timedelta(jday, "days")


@overload
def date_to_jday(day: str) -> float: ...


@overload
def date_to_jday(day: datetime) -> float: ...


@overload
def date_to_jday(day: pd.Series) -> pd.Series: ...


def date_to_jday(day):
    """Convert a timedelta64 pd.Series to a floating point fractional day.

    Parameters
    ----------
    series : pd.Series | datetime.datetime | str
        Input series of dtype timedelta64

    Returns
    -------
    float | pd.Series
        Time since JULIAN_REFERENCE_START in days (floating point)
    """
    if isinstance(day, pd.Series):
        if pd.api.types.is_datetime64_any_dtype(day.dtype):
            return (day - JULIAN_REFERENCE_START) / pd.to_timedelta(1, "days")
        if pd.api.types.is_string_dtype(day.dtype):
            return (
                pd.to_datetime(day, format="ISO8601") - JULIAN_REFERENCE_START
            ) / pd.to_timedelta(1, "days")
        raise ValueError(
            f"Cannot convert series of dtype {day.dtype} to fractional days"
        )

    if isinstance(day, str):
        return (datetime.fromisoformat(day) - JULIAN_REFERENCE_START) / timedelta(
            days=1
        )

    if isinstance(day, datetime):
        return (day - JULIAN_REFERENCE_START) / timedelta(days=1)

    raise ValueError(f"Cannot convert object of type {type(day)} to fractional days")
