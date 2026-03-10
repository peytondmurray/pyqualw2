from pathlib import Path

from pyqualw2 import model_runner


def test_model_runner(tmp_path):
    """Test the ModelRunner class."""
    config = {
        "run name": "test",
        "model dir": Path(__file__).parent.parent / "test/sample_data/test_model",
        "wait time": 60,
    }
    Runner = model_runner.ModelRunner(config, tmp_path)
    Runner.run()

    expected_file_list = set()
    test_dir = config["model dir"]

    for file in test_dir.iterdir():
        expected_file_list.add(file.name)

    conf_dir = tmp_path / str(config["run name"])
    conf_output_files = set()
    for file in conf_dir.iterdir():
        conf_output_files.add(file.name)
    assert expected_file_list == conf_output_files
