import itertools
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from matplotlib.testing.decorators import image_comparison
from numpy.testing import assert_equal

from pyqualw2.outputs import QWO, TWO, MultiRunResult, QWOLayers


def make_integer_combinations(it: range) -> Iterator[tuple[int, ...]]:
    """Make all possible combinations of the input iterator.

    Yields
    ------
    Iterator[tuple[int, ...]]
        Combinations of the input iterator of all possible lengths > 0

    Examples
    --------
    >>> list(make_integer_combinations(range(1, 5)))
    [

        (1,),
        (2,),
        (3,),
        (4,),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
        (1, 2, 3),
        (1, 2, 4),
        (1, 3, 4),
        (2, 3, 4)
    ]
    """
    for i in range(1, len(it)):
        yield from itertools.combinations(it, i)


@pytest.fixture
def sample_outputs(sample_data1) -> Path:
    """Get the outputs directory for the sample_data1 data."""
    return sample_data1 / "outputs"


@pytest.fixture
def sample_qwo_layers(sample_outputs) -> Path:
    """Get the path to the layer-dependent output flow data."""
    return sample_outputs / "qwo_layers_31_.csv"


@pytest.fixture
def sample_qwo(sample_outputs) -> Path:
    """Get the path to the total output flow data."""
    return sample_outputs / "qwo_31.csv"


@pytest.fixture
def sample_two(sample_outputs) -> Path:
    """Get the path to the temperature data."""
    return sample_outputs / "two_31.csv"


def test_QWO(sample_qwo):
    """Test that the QWO class correctly loads."""
    qwo = QWO.from_file(sample_qwo)
    assert list(qwo.data.columns[:2]) == ["JDAY", "QWD [m^3/s]"]

    # 4 branches, plus JDAY and total QWD. Last column gets dropped (there's an
    # extra comma at the end of each line)
    assert len(qwo.data.columns) == 6


@image_comparison(
    baseline_images=[
        "qwo_colormap",
    ],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_QWOLayers_colormap(sample_qwo_layers):
    """Test that the QWOLayers class produces a valid flow colormap."""
    layers = QWOLayers.from_file(sample_qwo_layers)
    layers.plot_colormap()


def test_QWOLayers(sample_qwo_layers):
    """Test that the QWOLayers class produces a valid flow colormap."""
    layers = QWOLayers.from_file(sample_qwo_layers)

    # Get the lowest layer; the flow should never be NA, since it's at the deepest part
    # of the water body.
    flow_layer_208 = layers.get_layer(208)
    assert flow_layer_208.columns.to_list() == ["JDAY", "Flow [m^3/s]"]
    assert flow_layer_208["Flow [m^3/s]"].notna().all()
    assert_equal(
        flow_layer_208.to_numpy(), layers.data[["JDAY", "flow_layer_208 [m^3/s]"]]
    )


@image_comparison(
    baseline_images=[
        "two_plot1",
        "two_plot2",
        "two_plot3",
        "two_plot4",
    ],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_TWO_plot(sample_two):
    """Test that the TWO class produces a valid plot."""
    two = TWO.from_file(sample_two)
    for i in range(1, 5):
        two.plot_structure(i)


@image_comparison(
    baseline_images=[
        "multirun_composite",
    ],
    remove_text=True,
    extensions=["png"],
    style="mpl20",
)
def test_multirun_composite_structure(multi_run_result):
    """Test that plotting multi-run TWO structure works."""
    res = MultiRunResult(multi_run_result)
    res.plot_two_structure_composite(4)


def test_multirun_result_days_above_threshold(multi_run_result):
    """Test that a MultiRunResult can produce a dict of the days above a threshold."""
    res = MultiRunResult(multi_run_result)
    ndays = res.get_days_above_threshold(4)
    assert "Tmax [C]" in ndays
    assert "Tmin [C]" in ndays
    assert "Tavg [C]" in ndays


def test_multirun_get_two_structure_composite_data(multi_run_result):
    """Test that a MultiRunResult can produce a dict of the days above a threshold."""
    res = MultiRunResult(multi_run_result)
    data, stats = res.get_two_structure_composite_data(4)

    assert "JDAY" in data.columns
    assert len(data.columns) == len(res.runs) + 1
    assert {"Tmax [C]", "Tavg [C]", "Tmin [C]"} < set(stats.columns)


@pytest.mark.parametrize(
    "branches",
    make_integer_combinations(range(1, 5)),
)
def test_mixing_temperature(multi_run_result, branches):
    """Test that the mixing temperature of multiple outputs is correctly calculated."""
    res = MultiRunResult(multi_run_result)
    df = res.runs[0].mixing_temperature(*branches)
    assert {"JDAY", "mixed_flow_temperature [C]"} == set(df.columns)

    qwo = res.runs[0].qwo
    two = res.runs[0].two

    # The interpolated JDAY should cover the range defined by the QWO and TWO datasets
    assert df["JDAY"].min() <= two.data["JDAY"].min()
    assert df["JDAY"].max() >= two.data["JDAY"].max()
    assert df["JDAY"].min() <= qwo.data["JDAY"].min()
    assert df["JDAY"].max() >= qwo.data["JDAY"].max()

    # Each mixing temperature should lie between the temperature of the individual
    # temperatures of the branches. Interpolate each branch on the timestamps of the
    # output, then check that the mixed flow temperature is between the hot and cold
    # temperatures.
    temps = []
    for br in branches:
        temps.append(
            np.interp(
                df["JDAY"],
                two.data["JDAY"],
                two.data[f"temperature_structure_{br} [C]"],
            )
        )

    total_flow = np.sum(
        [qwo.data[f"flow_structure_{br} [m^3/s]"] for br in branches],
        axis=0,
    )

    stacked = np.vstack(temps)
    hot = np.max(stacked, axis=0)
    cold = np.min(stacked, axis=0)

    col = "mixed_flow_temperature [C]"

    # The mixing temperature must be between the cold and hot temperatures, or the
    # flow must be zero (and therefore the mixed temperature equal to NaN)
    assert (
        (df[col] > cold)
        | np.isclose(df[col], cold)
        | ((total_flow == 0) & df[col].isna())
    ).all()
    assert (
        (df[col] < hot)
        | np.isclose(df[col], hot)
        | ((total_flow == 0) & df[col].isna())
    ).all()
