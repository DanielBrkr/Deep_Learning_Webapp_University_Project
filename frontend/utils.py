import json


def sort_certainty(json_path: str = 'frontend/static/predictions_to_display.json') -> None:
    """
    This function sorts a json file with the following structure based on 'certainty' - from low to high
    Structure of the json file:
        {"annotations": [{"file_name": "name1.jpeg", "label": "x", "certainty": 0.12858912348747253},
                         {"file_name": "name2.jpeg", "label": "y", "certainty": 0.2035144567489624}]}
    Parameters:
        (string) json_path: file path of the json file
    Return:
        (None)
    """

    # get existing json file
    with open(json_path, 'r') as file:
        dct = json.load(file)

    # sort dict
    sorted_dict = {'annotations': sorted(dct['annotations'], key=lambda d: d['certainty'])}

    # write sorted json file
    with open(json_path, 'w') as file:
        json.dump(sorted_dict, file, indent=4)
