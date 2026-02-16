from os import PathLike
from pathlib import Path


def relative_to_home_label(path: PathLike | str) -> str:
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
