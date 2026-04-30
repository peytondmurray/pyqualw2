from pathlib import Path

import pytest

from pyqualw2 import model_runner
from pyqualw2.config.config import Config
from pyqualw2.config.inputs import FlowInput


@pytest.fixture
def sample_config(
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
) -> Config:
    """Generate a sample Config class using existing data."""
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

    return config


@pytest.mark.e2e
def test_model_runner(tmp_path, sample_config):
    """Test the ModelRunner class."""
    test_dir = Path(__file__).parent.parent / "test" / "sample_data1"
    result_dir = tmp_path / "test_name"
    Runner = model_runner.ModelRunner(sample_config, tmp_path)
    Runner.run()

    # Check that the input files are all there
    expected = []
    for file in (test_dir / "inputs").iterdir():
        expected.append(file.name)

    found = []
    file_modified_times = {}
    for file in (result_dir / "inputs").iterdir():
        file_modified_times[file.name] = file.stat().st_mtime
        found.append(file.name)

    assert set(expected) == set(found)

    # Check that there is an outputs directory, and that there are files there
    assert (result_dir / "outputs").exists()
    assert len(set((result_dir / "outputs").iterdir())) > 0

    # Test that this fails when the output files already exist
    with pytest.raises(FileExistsError):
        Runner.run(overwrite=False)

    # Check that overwriting the result directory works, i.e. the files are actually
    # being replaced (verified by checking the file modified time of everything in
    # inputs/).
    Runner.run(overwrite=True)
    found = []
    for file in (result_dir / "inputs").iterdir():
        assert file.stat().st_mtime > file_modified_times[file.name]
        found.append(file.name)

    assert set(expected) == set(found)

    # Check that there is an outputs directory, and that there are files there
    assert (result_dir / "outputs").exists()
    assert len(set((result_dir / "outputs").iterdir())) > 0
