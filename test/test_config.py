from pathlib import Path

import pandas as pd
import pytest

from pyqualw2.config.config import Config
from pyqualw2.config.inputs import JULIAN_REFERENCE_START, FlowInput


@pytest.fixture
def met_data_files(sample_data1) -> list[str]:
    """Get a list of met data files."""
    return [str(p) for p in Path(sample_data1 / "historic_data" / "met_data").iterdir()]


def test_parametrize(
    met_data_files,
    sample_w2_con,
    sample_bathymetry,
    sample_profile,
    sample_shade,
    sample_wind_sheltering,
    sample_temperature_tributary,
    sample_temp_2018,
    sample_met_2018,
    sample_flow_2018,
    cequalw2_binary,
):
    """Test that Config.parametrize works as intended."""
    config = Config.from_files(
        name="test_name",
        con=sample_w2_con,
        bathymetry=sample_bathymetry,
        profile=sample_profile,
        met_data=sample_met_2018,
        shade=sample_shade,
        temperature_tributaries=[sample_temperature_tributary],
        # Due to misconfiguration in w2_con.csv for this test data, we need to fiddle
        # the branch temperature data to get cequalw2 to run. See
        # https://github.com/steelhead-dev/pyqualw2/issues/69 and
        # https://github.com/steelhead-dev/pyqualw2/issues/64 for details.
        temp_data=[
            sample_temp_2018,
            sample_temp_2018,
            sample_temp_2018,
            sample_temp_2018,
        ],
        wind_sheltering=sample_wind_sheltering,
        flow_data=sample_flow_2018,
        flow_data_date_col="Date",
        flow_data_inflow_cols=[["Date", "M_IN"]],
        flow_data_outflow_cols=[["Date", "SPL_OUT", "FKC_OUT", "MC_OUT", "SJR_OUT"]],
        flow_data_evaporation_cols=[["Date", "MIL_EVAP"]],
        cequalw2_path=cequalw2_binary,
    )

    # Due to misconfiguration in w2_con.csv for this test data, we need to fiddle
    # the branch data to get cequalw2 to run. See
    # https://github.com/steelhead-dev/pyqualw2/issues/69 for more information.
    # Set all the flow to 0 for branches 2, 3, and 4. Then add three copies of this
    # dummy data to serve as those dummy branches
    dummy_flow_data = config.branch_inflow[0].data.copy()
    dummy_flow_data.iloc[:, 1] = 0
    config.branch_inflow.extend(
        [
            FlowInput(data=dummy_flow_data.copy()),
            FlowInput(data=dummy_flow_data.copy()),
            FlowInput(data=dummy_flow_data.copy()),
        ]
    )

    configs = config.parameterize(parameters={"met_data": met_data_files})
    start, end, year = config.con.timedata

    names = []
    for conf in configs:
        # The w2_con.csv simulation start/stop/year should be the same...
        assert conf.con.timedata == (start, end, year)

        # ...but the metrology data should be adjusted
        sim_start_date = JULIAN_REFERENCE_START + pd.to_timedelta(start, "days")
        assert (
            JULIAN_REFERENCE_START
            + pd.to_timedelta(conf.met_data.data["JDAY"].iloc[0], "days")
        ).year == sim_start_date.year

        names.append(conf.name)

    assert sorted(set(names)) == sorted(names)
