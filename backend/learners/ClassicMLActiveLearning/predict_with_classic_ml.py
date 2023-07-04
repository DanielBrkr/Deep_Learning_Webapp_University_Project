"""This module predicts labels of new images using an existing active learning model"""
import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from backend.pre_processing.feature_extraction.radiomics_features import RadiomicsFeatures
from backend.helpers.reverse_label_img import reverse_label_img


def predict_with_classic_ml(image_list, model: Pipeline):
    """
    Returns the labels of the given test images with the given ML pipeline & model.

        Parameters:
            image_list (list):
                List of image paths.
            model (Pipeline):
                Pipeline instance .

        Returns:
            annotations (dict):
                Dictionary with the image paths and corresponding labels
                (according to specification).
            data (pd.Dataframe):
                Extracted features of image_list.
    """

    # Feature extraction
    data = extract_f(image_list)

    # Label prediction
    labels = model.predict(data)
    class_proba = model.predict_proba(data)

    # Calculate the uncertainties of each sample using the smallest margin strategy
    class_poba_sort = class_proba.copy()
    class_poba_sort.sort(axis=1)  # sort the rows
    conf_margin = np.diff(class_poba_sort)[:, 2]  # Difference btw. top 2 conf. scores

    # Save image paths and corresponding labels in dictionary according to specification
    annotations = {'annotations': []}
    labels_list = list(labels)
    # Convert integer labels into string labels
    for idx, label in enumerate(labels_list):
        labels_list[idx] = reverse_label_img(label)

    for idx, image in enumerate(image_list):
        annotations['annotations'].append({'file_name': os.path.split(image)[1],
                                           'label': labels_list[idx],
                                           'certainty': conf_margin[idx]})
    return annotations, data


def extract_f(image_list: list):
    """
    Extract radiomics features from given list of image paths

        Parameters:
            image_list (list):
                list of image paths

        Returns:
            data (pd.Dataframe):
                dataframe of extracted features
    """
    # Feature extraction
    rad_features = RadiomicsFeatures()
    data = rad_features.extract_features(image_list)  # extracted features
    return data


def transform_features(data: pd.DataFrame, model: Pipeline):
    """
    Transforms the given features using the StandardScaler, removed_features and PCA from training.

        Parameters:
            data (dataframe):
                extracted features
            model (Pipeline):
                Pipeline instance.

        Returns:
            data_pca (dataframe):
                Features with applied StandardScaler, removed_features & PCA
    """
    # Feature selection
    # Use FeatureSelection object from pipeline to transform the data
    data_selected = model.named_steps.featselec.transform(data)

    # PCA transform
    data_pca = pd.DataFrame(model.named_steps.dimred.transform(data_selected))
    return data_pca
