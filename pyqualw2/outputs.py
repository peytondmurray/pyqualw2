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
            raise TypeError(
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
