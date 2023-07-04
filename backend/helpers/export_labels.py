"""
This module returns a json representation of image names and labels
"""

import json


def export_labels(json_file_path):
    """
    takes the json file with annotations
    returns a json file with labels and file_names as requested in project description
    """
    with open(json_file_path, "r", encoding="utf8") as json_file:
        data = json.load(json_file)
    image_id_list = [data["images"][idx]["id"] for idx in range(len(data["images"]))]
    image_file_name_list = [
        data["images"][idx]["file_name"] for idx in range(len(data["images"]))
    ]
    image_dict = dict(zip(image_id_list, image_file_name_list))

    export = []

    for ele in data["annotations"]:
        export.append({"file_name": image_dict[ele["image_id"]], "label": ele["label"]})

    # convert into JSON:
    export_dict = {}
    export_dict["annotations"] = export
    return json.dumps(export_dict)
