from pathlib import Path

from pyqualw2 import model_runner


def test_model_runner(tmp_path):
    """Test the ModelRunner class."""
    config = [1, 2]
    Runner = model_runner.ModelRunner(config, tmp_path)
    Runner.run()

    test_dir = Path(__file__).parent.parent / "test/sample_data/test_model"
    expected_file_list = set()

    for file in test_dir.iterdir():
        expected_file_list.add(file.name)
    for conf in config:
        conf_dir = tmp_path / str(conf)
        conf_output_files = set()
        for file in conf_dir.iterdir():
            conf_output_files.add(
                file.name
            )  # Right before the assert - temporary debug
        expected_file_list - conf_output_files
        conf_output_files - expected_file_list
        assert expected_file_list == conf_output_files
