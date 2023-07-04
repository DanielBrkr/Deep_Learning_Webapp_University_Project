"""
In this module different strategies for calculating the certainty of the model for the
active learning process are implemented
"""
import numpy as np


def margin_based(outputs):
    """
    Returns margin based certainty
    """
    values = outputs[0]
    conf_1 = np.max(values, axis=-1)
    values = values[values != conf_1]
    conf_2 = np.max(values, axis=-1)
    margin = conf_1 - conf_2

    return margin


def confidence_based(outputs):
    """
    value of output node that represents the predicted class
    """
    return np.max(outputs, axis=-1)[0]
