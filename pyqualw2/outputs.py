import functools
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import StringIO
from os import PathLike
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate as si
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pyqualw2.utils import jday_to_date

log = logging.getLogger(__name__)


class OutputBase(ABC):
    """Base class for all output file types."""

    @classmethod
    @abstractmethod
    def from_file(cls, filename: PathLike | str) -> Self:
        """Parse an output data file to a python object.

        Parameters
        ----------
        filename : PathLike | str
            Path to the data file

        Returns
        -------
        Self
            An object containing the data from the file
        """


@dataclass
class QWO(OutputBase):
    """A container for the QWO flow data."""

    header: str
    data: pd.DataFrame
    filename: PathLike | str | None = None

    @classmethod
    def from_file(cls, filename: PathLike | str) -> Self:
        """Load a QWO file containing total flow data.

        Parameters
        ----------
        filename : PathLike | str
            File to load

        Returns
        -------
        Self
            An object containing the flow data for the whole system
        """
        with open(filename) as f:
            lines = f.readlines()

        df = pd.read_csv(
            StringIO("\n".join(lines[3:])),
            header=None,
        ).dropna(axis="columns", how="all")
        df.columns = ["JDAY", "QWD [m^3/s]"] + [
            f"flow_branch_{i} [m^3/s]" for i, _ in enumerate(df.columns[2:], start=1)
        ]

        return cls(
            filename=Path(filename),
            header="\n".join(lines[:2]),
            data=df,
        )


@dataclass
class QWOLayers(OutputBase):
    """A container for QWO layer-dependent flow data."""

    header: str
    data: pd.DataFrame
    filename: PathLike | str | None = None

    def __post_init__(self):
        # JDAY, QWD, and ELWS aren't layer flows
        self.n_layers = self.data.shape[1] - 3

    @classmethod
    def from_file(cls, filename: PathLike | str) -> Self:
        """Load a QWO file containing layer-dependent flow data.

        Parameters
        ----------
        filename : PathLike | str
            File to load

        Returns
        -------
        Self
            An object containing the layer flow data
        """
        with open(filename) as f:
            lines = f.readlines()

        df = pd.read_csv(
            StringIO("\n".join(lines[3:])),
            header=None,
        )
        df.columns = ["JDAY", "QWD [m^3/s]", "ELWS [m]"] + [
            f"flow_layer_{i} [m^3/s]" for i, _ in enumerate(df.columns[3:], start=2)
        ]

        return cls(
            filename=Path(filename),
            header="\n".join(lines[:2]),
            data=df,
        )

    def get_layer(self, layer: int) -> pd.DataFrame:
        """Get the flow data for an individual layer.

        Parameters
        ----------
        layer : int
            Layer to get the flow for; must be in the range [2, n_layers)

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the timestamp and flow for the requested layer
        """
        if layer < 2 or layer > self.n_layers + 1:
            raise ValueError(
                f"Cannot get layer flow for layer {layer}. Valid layer "
                f"numbers are in the range [2, {self.n_layers})"
            )

        column = f"flow_layer_{layer} [m^3/s]"
        return self.data[["JDAY", column]].rename(columns={column: "Flow [m^3/s]"})

    def plot_colormap(self, ax: Axes | None = None) -> Figure:
        """Generate a colormap showing flow as a function of layer number and time.

        Parameters
        ----------
        ax : Axes | None
            Axes on which the colormap should be rendered

        Returns
        -------
        Figure
            The figure containing the colormap
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        # Reshape the data into 3 grids of shape (nlayers, njday) so that the
        # x-coordinate is time and the y-coordinate is layer number
        jday, layers = np.meshgrid(
            self.data["JDAY"],
            np.arange(2, 2 + self.n_layers),
        )
        flows = self.data.iloc[:, 3:].transpose()

        # Interpolate a regularly spaced grid of flows from the original data
        jday_min, jday_max = self.data["JDAY"].min(), self.data["JDAY"].max()
        layer_min, layer_max = 2, 2 + self.n_layers
        jday_interp, layers_interp = np.meshgrid(
            np.linspace(jday_min, jday_max, len(self.data)),
            np.arange(layer_min, layer_max),
        )
        flows_interp = si.griddata(
            (np.reshape(jday, (-1,)), np.reshape(layers, (-1,))),
            np.reshape(flows, (-1,)),
            (jday_interp, layers_interp),
            method="nearest",
        )

        im = ax.imshow(
            flows_interp,
            extent=(
                jday_to_date(jday_min),
                jday_to_date(jday_max),
                layer_min,
                layer_max,
            ),
            cmap="viridis",
            aspect="auto",
            origin="lower",
        )

        # Add a colorbar, labels, and style the plot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", pad=0.05, size="5%")
        fig.colorbar(im, cax=cax, label="Layer flows $[m^3/s]$")  # ty:ignore[unresolved-attribute]

        ax.set_xlabel("Date")
        ax.set_ylabel("Layer")
        ax.yaxis.set_inverted(True)

        return fig  # ty:ignore[invalid-return-type]


@dataclass
class TWO(OutputBase):
    """A container for TWO temperature data for each structure."""

    header: str
    data: pd.DataFrame
    filename: PathLike | str | None = None

    def __post_init__(self):
        # JDAY and T [C] aren't structure-specific temperature columns
        self.n_structures = self.data.shape[1] - 2

    @classmethod
    def from_file(cls, filename: PathLike | str) -> Self:
        """Load a TWO file containing temperature data for each structure.

        Parameters
        ----------
        filename : PathLike | str
            File to load

        Returns
        -------
        Self
            An object containing the temperature of the water in each structure
        """
        with open(filename) as f:
            lines = f.readlines()

        df = pd.read_csv(
            StringIO("\n".join(lines[3:])),
            header=None,
        ).dropna(axis="columns", how="all")

        cols = ["JDAY", "T [C]"]
        for i, _ in enumerate(df.columns[2:], start=1):
            cols.append(f"temperature_structure_{i} [C]")

        df.columns = cols
        return cls(
            filename=Path(filename),
            header="\n".join(lines[:2]),
            data=df,
        )

    def get_structure(self, structure: int) -> pd.DataFrame:
        """Get the timeseries temperature data for the given structure.

        Parameters
        ----------
        structure : int
            Structure number to get the temperature of

        Returns
        -------
        pd.DataFrame
            A DataFrame containing two columns:

                JDAY
                Temperature [C]
        """
        if structure < 1 or structure > self.n_structures + 1:
            raise ValueError(
                "Cannot get temperature of flow in structure {structure}. Valid "
                f"structure numbers are in the range [1, {self.n_structures})"
            )

        col = f"temperature_structure_{structure} [C]"
        return self.data[["JDAY", col]].rename(columns={col: "Temperature [C]"})

    def plot_structure(self, structure: int, ax: Axes | None = None, **kw) -> Figure:
        """Plot the temperature of the given structure.

        Parameters
        ----------
        structure : int
            Structure to plot temperature of
        ax : Axes | None
            Axes on which to plot the temperature. If None, a new figure is generated
        **kw
            Other keyword arguments to pass to `ax.plot`

        Returns
        -------
        Figure
            Figure on which the plot has been made

        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        df = self.get_structure(structure)

        ax.plot(jday_to_date(df["JDAY"]), df["Temperature [C]"], **kw)
        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature [C]")

        return fig  # ty:ignore[invalid-return-type]


@functools.total_ordering
class RunResult:
    """A container which holds the results of a single cequalw2 run."""

    @staticmethod
    def _find_two_file(path: PathLike | str) -> Path:
        path = Path(path)
        for file in (path / "outputs").iterdir():
            if re.match(r"two_\d+.csv", file.name):
                return file
        raise ValueError(f"Cannot find a TWO file in {str(path)}.")

    def __init__(self, path: PathLike | str):
        self.path = Path(path)
        self.two = TWO.from_file(self._find_two_file(path))
        self.name = self.path.name

    def plot_two_structure(
        self,
        structure: int,
        ax: Axes | None = None,
        **kw,
    ) -> Figure:
        """Plot the temperature of the output at the given structure.

        Parameters
        ----------
        structure : int
            Structure for which the temperature should be plotted
        ax : Axes | None
            Axes to plot on. If None, a new figure is generated
        **kw
            Additional kwargs are passed to TWO.plot_structure

        Returns
        -------
        Figure
            The figure containing the axes where the data was plotted
        """
        return self.two.plot_structure(structure, ax, **kw)

    def __ge__(self, other):  # noqa: ANN001
        return self.name >= other.name

    def __eq__(self, other):  # noqa: ANN001
        return self.name == other.name


class MultiRunResult:
    """Class which manipulates outputs for multiple simulation runs."""

    def __init__(self, path: PathLike | str):
        self.path = Path(path)
        self.runs: list[RunResult] = []

        # Sort the run directories so that different platforms generate the same
        # MultiRunResult when run on the same data (iterdir has no guarantee of
        # ordering)
        for item in sorted(self.path.iterdir()):
            if item.is_dir():
                self.runs.append(RunResult(item))

    def plot_two_structure(
        self,
        structure: int,
        ax: Axes | None = None,
        **kw,
    ) -> Figure:
        """Plot the temperature of the output at the given structure for all runs.

        Parameters
        ----------
        structure : int
            Structure for which the temperature should be plotted
        ax : Axes | None
            Axes to plot on. If None, a new figure is generated
        **kw
            Additional kwargs are passed to TWO.plot_structure

        Returns
        -------
        Figure
            The figure containing the axes where the data was plotted
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        for run in self.runs:
            run.plot_two_structure(structure, ax, **kw)

        return fig  # ty:ignore[invalid-return-type]

    def get_two_structure(self, structure: int) -> pd.DataFrame:
        """Get the temperature data for the given structure for all runs.

        Normally, cequalw2 does not produce the same timestamps for temperature time
        series for different runs. This function linearly interpolates the temperature
        of each run across a new JDAY linspace, with start, stop, and step using the
        first run's JDAY as reference. This makes the temperatures at each timestep
        directly comparable.

        Parameters
        ----------
        structure : int
            Structure for which the data should be retrieved

        Returns
        -------
        pd.DataFrame
            Interpolated time series temperature data for the given structure
        """
        already_warned = False
        data = {}

        for run in self.runs:
            df = run.two.get_structure(structure)

            if "JDAY" not in data:
                jday = df["JDAY"].to_numpy()
                data["JDAY"] = np.linspace(jday.min(), jday.max(), len(jday))
            else:
                if (
                    (df["JDAY"].max() > data["JDAY"].max())
                    or (df["JDAY"].min() < data["JDAY"].min())
                    and not already_warned
                ):
                    range_data = [float(data["JDAY"].min()), float(data["JDAY"].max())]
                    range_df = [float(df["JDAY"].min()), float(df["JDAY"].max())]
                    log.warning(
                        f"Comparing temperature data for structure {structure}; one "
                        "dataset has bounds outside another, meaning that the "
                        "interpolated date will have artefacts at the edge of the "
                        f"combined dataset. Interpolated JDAY: ({range_data}), this "
                        f"dataset: ({range_df})"
                    )
                    already_warned = True

            data[f"Temperature ({run.name}) [C]"] = np.interp(
                data["JDAY"], df["JDAY"], df["Temperature [C]"]
            )

        return pd.DataFrame(data=data)

    def plot_two_structure_composite(
        self,
        structure: int,
        ax: Axes | None = None,
        fill_kw: dict | None = None,
        tavg_kw: dict | None = None,
    ) -> Figure:
        """Plot the average temperature of a structure vs time over multiple runs.

        The min and max temperature at any given timestep is represented with a shaded
        region.

        Parameters
        ----------
        structure : int
            Structure for which the temperature should be plotted
        ax : Axes | None
            Axes on which to plot. If this is None, a new figure is generated
        fill_kw : dict | None
            Additional keyword arguments to pass to the fill_between function
        tavg_kw : dict | None
            Additional keyword arguments to pass to the average temperature plot call

        Returns
        -------
        Figure
            The figure on which the data was plotted
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            fig = ax.get_figure()

        df = self.get_two_structure(structure)

        temperature_cols = df.columns[1:]

        df["Tmax [C]"] = df[temperature_cols].max(axis="columns")
        df["Tmin [C]"] = df[temperature_cols].min(axis="columns")
        df["Tavg [C]"] = df[temperature_cols].mean(axis="columns")

        date = jday_to_date(df["JDAY"])

        if fill_kw is None:
            fill_kw = {}
        fill_kw = {"alpha": 0.6} | fill_kw

        if tavg_kw is None:
            tavg_kw = {}
        tavg_kw = {"color": "k"} | tavg_kw

        ax.fill_between(
            date,
            df["Tmin [C]"],
            df["Tmax [C]"],
            label="Temperature Range [C]",
            **fill_kw,
        )
        ax.plot(
            date,
            df["Tavg [C]"],
            label="Tavg",
            **tavg_kw,
        )

        ax.set_xlabel("Date")
        ax.set_ylabel("Temperature [C]")
        return fig  # ty:ignore[invalid-return-type]
