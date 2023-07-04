"""This module retrains an existing active learning model"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from backend.helpers.parse_label_name_to_folder import parse_label_name
from backend.helpers.label_img import label_img


def retrain_model_classic_ml(annotations: dict, model: Pipeline,
                             new_train_data_transformed: pd.DataFrame):
    """
    Returns an Active Learning model that is retrained on new image data that was labeled by the
    model in a previous prediction step.

        Parameters:
            annotations (dict):
                Dictionary of the image paths and corresponding labels (according to specification).
            model (Pipeline):
                Pipeline instance.
            new_train_data_transformed (dataframe):
                Feature correspondence of new data transformed via model.

        Returns:
            model (Pipeline):
                Retrained Pipeline instance.
    """

    # Concatenating the new train data and old train data + new train labels and old train labels
    new_train_labels = [val['label'] for idx, val in enumerate(annotations['annotations'])]
    # Convert string labels into integer labels
    for idx, val in enumerate(new_train_labels):
        new_train_labels[idx] = label_img(parse_label_name(val))

    new_train_labels = pd.Series(new_train_labels)

    train_data_transformed = model.named_steps.classifier.init_train_batch.drop(['Label'], axis=1)
    train_labels = model.named_steps.classifier.init_train_batch['Label']

    total_train_data_transformed = pd.concat([train_data_transformed, new_train_data_transformed],
                                             axis=0, ignore_index=True)
    total_train_labels = pd.concat([train_labels, new_train_labels], axis=0, ignore_index=True)

    # Fitting the ML model(s) inside the FinalTrainingPipeline instance "model"
    model.named_steps.classifier.base_estimator.fit(np.array(total_train_data_transformed),
                                                    np.array(total_train_labels))
    if model.named_steps.classifier.ensemble:  # if the AL ensemble was used
        model.named_steps.classifier.ensemble_estimator_1.fit(
            np.array(total_train_data_transformed),
            np.array(total_train_labels))
        model.named_steps.classifier.ensemble_estimator_2.fit(
            np.array(total_train_data_transformed),
            np.array(total_train_labels))

    # Concatenate total train data and total train labels into one dataframe
    # and save in init_train_batch of the AL classifier
    model.named_steps.classifier.init_train_batch = pd.concat(
        [total_train_data_transformed, total_train_labels], axis=1)
    model.named_steps.classifier.init_train_batch.set_axis(
        [*model.named_steps.classifier.init_train_batch.columns[:-1], 'Label'], axis=1,
        inplace=True)  # rename the last column to "Label"

    # Return new model
    return model
