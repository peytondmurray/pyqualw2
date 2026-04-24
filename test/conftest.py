from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_data1() -> Path:
    """Get the path to the sample data."""
    return Path(__file__).parent / "sample_data1"


@pytest.fixture
def sample_w2_con(sample_data1) -> Path:
    """Get the path to an example w2_con.csv file."""
    return sample_data1 / "w2_con.csv"


@pytest.fixture
def sample_inputs(sample_data1) -> Path:
    """Get the inputs/ directory for sample_data1.

    This contains the actual input files required by cequalw2.
    """
    return sample_data1 / "inputs"


@pytest.fixture
def sample_bathymetry(sample_inputs) -> Path:
    """Get the path to an example bathymetry file."""
    return sample_inputs / "mbth_wb1.csv"


@pytest.fixture
def sample_profile(sample_inputs) -> Path:
    """Get the path to an example profile file."""
    return sample_inputs / "mvpr1.npt"


@pytest.fixture
def sample_met_2018(sample_data1) -> Path:
    """Get the path to an example met data input file."""
    return sample_data1 / "historic_data" / "met_data" / "2018_CEQUAL_met_inputs.csv"


@pytest.fixture
def sample_temp_2018(sample_data1) -> Path:
    """Get the path to an example temp inflow data input file."""
    return sample_data1 / "historic_data" / "temp_data" / "SJA_2018_temp.csv"


@pytest.fixture
def sample_flow_2018(sample_data1) -> Path:
    """Get the path to an example inflow data input file."""
    return sample_data1 / "historic_data" / "flow_data" / "2018_Observed_Flow.csv"


@pytest.fixture
def sample_shade(sample_inputs) -> Path:
    """Get the path to an example shade data input file."""
    return sample_inputs / "mshade.npt"


@pytest.fixture
def sample_wind_sheltering(sample_inputs) -> Path:
    """Get the path to an example wind sheltering data input file."""
    return sample_inputs / "mwsc.npt"


@pytest.fixture
def sample_temperature_tributary(sample_inputs) -> Path:
    """Get the path to an example tributary temperature data input file."""
    return sample_inputs / "mtdt_br1.npt"


@pytest.fixture
def cequalw2_binary() -> Path:
    """Get the path to the cequalw2 binary."""
    return Path(__file__).parent.parent / "cequalw2" / "w2_v45_64.exe"


@pytest.fixture
def sample_mmet3(sample_inputs) -> pd.DataFrame:
    """Get some example metrology data ready for cequalw2 input."""
    return pd.read_csv(sample_inputs / "mmet3.csv", skiprows=2)
