"""Test functions for the Backend"""
from pathlib import Path
from Project_Paperwork.Legacy_Code.classic_ml_scripts.data_split import DataSplit


data_path = Path(__file__).parent.parent / "Data"


def test_datasplit_testsize():
    """Test functions for the datasplit_testsize"""
    data_split = DataSplit(split_type="random", test_size=150, data_path=str(data_path))
    _, test_set = data_split.split  # pylint: disable=unbalanced-tuple-unpacking
    assert len(test_set) == 150
