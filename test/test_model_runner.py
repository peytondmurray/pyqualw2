from pathlib import Path

from pyqualw2 import model_runner
from pyqualw2.config.config import Config


def test_model_runner(
    tmp_path,
    sample_w2_con,
    sample_bathymetry,
    sample_profile,
    sample_temp,
    sample_met,
    sample_flow,
):
    """Test the ModelRunner class."""
    config = Config.from_files(
        sample_w2_con,
        sample_bathymetry,
        sample_profile,
        sample_temp,
        sample_met,
        sample_flow,
    )

    expected_file_list = set()
    test_dir = Path(__file__).parent.parent / "test/sample_data/test_model"

    Runner = model_runner.ModelRunner(config, tmp_path, test_dir, "test_run")
    Runner.run()
    for file in test_dir.iterdir():
        expected_file_list.add(file.name)

    conf_dir = tmp_path / "test_run"
    conf_output_files = set()
    for file in conf_dir.iterdir():
        conf_output_files.add(file.name)
    assert expected_file_list == conf_output_files
