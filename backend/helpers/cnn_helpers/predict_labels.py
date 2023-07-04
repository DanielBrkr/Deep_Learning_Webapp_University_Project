"""
In this module the labels for the images, provided from the frontend
are predicted by the model.
"""
import os
import numpy as np
from backend.helpers.reverse_label_img import reverse_label_img
from backend.helpers.cnn_helpers.al_strategies import margin_based

# Prevents crashes while running in non-cuda environments by hiding the cuda device from tensorflow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def predict_labels(data, model):
    """
    This function predicts the label and stores the prediction in the data dictionary
    with the certainty of each prediction in addition.
    """

    for img in data["annotations"]:
        pred = model.predict(img["image"])
        img["label"] = reverse_label_img(np.argmax(pred, axis=-1)[0])
        img["certainty"] = margin_based(pred)
        img.pop("image")

    return data
