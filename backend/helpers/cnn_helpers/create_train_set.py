"""
This module creates 4 folders that are named like the labels and
stores the images selected for retraining the model from the frontend
in the correct folder according to their labels.
This folder structure is needed for creating the tensorflow train sets.
"""
import os
import shutil
from pathlib import Path
from backend.helpers.parse_label_name_to_folder import parse_label_name


def create_train_folders(data_dict):
    """Function that creates the folders that are used to create the re-train set"""

    folder_path = Path(__file__).parent.parent.parent.parent / "Data/Correction_Train"
    frontend_images_path = Path(__file__).parent.parent.parent.parent / "frontend/static/temp"

    # create folders
    exist = folder_path.exists()
    if exist:
        shutil.rmtree(folder_path )

    os.mkdir(folder_path)
    os.mkdir(Path(folder_path / "a_scratch"))
    os.mkdir(Path(folder_path / "b_dent"))
    os.mkdir(Path(folder_path / "c_rim"))
    os.mkdir(Path(folder_path / "d_other"))

    for item in data_dict["annotations"]:
        label = parse_label_name(item["label"])

        img_path = Path(frontend_images_path / item["file_name"])
        dst_path = Path(folder_path / label)

        shutil.copy(img_path, dst_path)
