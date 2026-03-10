from pathlib import Path

import pytest

from pyqualw2.config import Config


@pytest.mark.skip
def test_from_csv():
    """Test that a config can be generated from data files."""
    Para_Dict = {
        "run_name": "test run",
        "model_name": "millerton",
        "start": "1/1/24",
        "end": "1/2/24",
        "years": [2020, 2023],
        "i_WSE": 273.5,
        "i_profile": "1/1/24",
        "model_dir": Path(__file__).parent.parent / "test/sample_data/test_model",
        "desc": """
        this is a test run for testing purposes
        """,
    }

    Config.from_dict(Para_Dict)
