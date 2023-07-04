"""
This module extends each image dictionary in the annotations list by the
actual image converted to a tensor and the certainty of the prediction which will
be determined during the prediction process. The structure of the data dictionary
therefore extends to the following as it is needed for the correction module in
the frontend:
{
    "annotations":
    [
        {
            "image": ...,
            "file_name" ...,
            "label" ...,
            "certainty" ...,
        }
        ...
    ]
}
"""
from pathlib import Path
import tensorflow as tf


def make_tensor(data, path, img_tuple):
    """Function that creates the image tensors and adds the certainty instance in each image dict"""
    test_images = []

    for item in data["annotations"]:
        test_img = {"image": None, "file_name": None, "label": None, "certainty": None}
        img_name = item["file_name"]
        img = Path(path / img_name)
        img = tf.keras.utils.load_img(
            img, target_size=img_tuple
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        test_img["image"] = img_array
        test_img["file_name"] = img_name
        test_img["label"] = item["label"]
        test_img["certainty"] = item["certainty"]
        test_images.append(test_img)

    data["annotations"] = test_images

    return data
