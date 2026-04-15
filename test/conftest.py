from pathlib import Path

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
def sample_bathymetry(sample_data1) -> Path:
    """Get the path to an example bathymetry file."""
    return sample_data1 / "inputs" / "mbth_wb1.csv"


@pytest.fixture
def sample_profile(sample_data1) -> Path:
    """Get the path to an example profile file."""
    return sample_data1 / "inputs" / "mvpr1.npt"


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
