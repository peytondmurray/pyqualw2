from pathlib import Path

from pyqualw2 import model_runner
from pyqualw2.config.config import Config
from pyqualw2.config.inputs import FlowInput


def test_model_runner(
    tmp_path,
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
    """Test the ModelRunner class."""
    config = Config.from_files(
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
        flow_data=sample_flow_2018,
        wind_sheltering=sample_wind_sheltering,
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

    test_dir = Path(__file__).parent.parent / "test" / "sample_data1"
    Runner = model_runner.ModelRunner(config, tmp_path, test_dir, "test_run")
    Runner.run()

    # Check that the input files are all there
    expected = []
    for file in (test_dir / "inputs").iterdir():
        expected.append(file.name)

    found = []
    for file in (tmp_path / "test_run" / "inputs").iterdir():
        found.append(file.name)

    assert set(expected) == set(found)

    # Check that there is an outputs directory, and that there are files there
    assert (tmp_path / "test_run" / "outputs").exists()
    assert len(set((tmp_path / "test_run" / "outputs").iterdir())) > 0
