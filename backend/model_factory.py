import pickle
import hashlib
import shutil
from abc import ABC, abstractmethod
from PIL import Image
import numpy as np
import pandas as pd
from tensorflow import keras
from pathlib import Path
from backend.helpers.cnn_helpers.create_tensor_predict import make_tensor
from backend.helpers.cnn_helpers.predict_labels import predict_labels
from backend.helpers.cnn_helpers.finetune_train import finetune_train
from backend.learners.ClassicMLActiveLearning.predict_with_classic_ml import (
    predict_with_classic_ml,
)
from backend.learners.ClassicMLActiveLearning.retrain_model_classic_ml import (
    retrain_model_classic_ml,
)
from backend.learners.ClassicMLActiveLearning.predict_with_classic_ml import (
    transform_features,
)
from backend.learners.ClassicMLActiveLearning.predict_with_classic_ml import extract_f
from backend.helpers.cnn_helpers import (
    cnn_net_arch_visualizer,
    cnn_net_weight_matrix_vis,
    cnn_saliency_maps,
)
from PIL import Image
import os

PATH = Path(__file__).parent.parent / "frontend/static/temp"
DATA = {
    "annotations": [
        {"file_name": "neu_1.jpeg", "label": None, "certainty": None},
        {"file_name": "neu_2.jpeg", "label": None, "certainty": None},
        {"file_name": "neu_3.jpeg", "label": None, "certainty": None},
        {"file_name": "neu_4.jpeg", "label": None, "certainty": None},
        {"file_name": "neu_5.jpeg", "label": None, "certainty": None},
        {"file_name": "neu_6.jpeg", "label": None, "certainty": None},
        {"file_name": "neu_7.jpeg", "label": None, "certainty": None},
    ]
}
DATA_TRAIN = {
    "annotations": [
        {"file_name": "neu_3.jpeg", "label": "scratch", "certainty": None},
        {"file_name": "neu_5.jpeg", "label": "dent", "certainty": None},
        {"file_name": "neu_7.jpeg", "label": "scratch", "certainty": None},
    ]
}


class IModel(ABC):
    """Abstract class that sets the required functions for ML-Models"""

    @property
    @abstractmethod
    def model(self):
        """Each classifier should have a model attribute that contains the
        weights of the model"""
        ...

    @model.setter
    @abstractmethod
    def model(self, value):
        """model setter is mandatory to implement"""
        ...

    @abstractmethod
    def json_to_data_format(self, data: dict):
        """A method that converts the input dict to the
        required format"""
        ...

    @abstractmethod
    def predict(self, data: dict) -> dict:
        """A method that takes the dict provided from the frontend and
        adds the labels and uncertainties"""
        ...

    @abstractmethod
    def train_model(self, train_data: dict) -> None:
        """A method that implements the active learning routine"""
        ...

    @abstractmethod
    def save_model(self) -> None:
        """A method that saves the model to a format that
        can be loaded with the load model function"""
        ...

    def show_edge_heatmap(self, path_save):
        pass


class ResNet50V2(IModel):
    """A simple but efficient CNN model with only one hidden dense layer"""

    def __init__(
        self,
        model_path=Path(__file__).parent / "Trained_Models/ResNet50V2/res_net_50_v2.h5",
    ):
        self.data = None  # self.json_to_data_format()
        self.train_data = None
        self.model = keras.models.load_model(model_path)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def json_to_data_format(self, data: dict):
        return make_tensor(data, PATH, (224, 224, 3))

    def predict(self, data: dict) -> dict:
        self.data = self.json_to_data_format(data)
        return predict_labels(self.data, self.model)

    def show_attention(self, img_path, path_save):
        """Predicts the label for the provided image and plots the saliency/attention
        of the model with regard for frontend display.
        to the predicted class, resnet (bool): needs to be set due to the different
        image preprocessing"""
        cnn_saliency_maps.cnn_visualize_saliency(self.model, img_path,
                                                 resnet=True, path_save=path_save)


    def train_model(self, data_train) -> None:
        self.train_data = self.json_to_data_format(data_train)
        self.model = finetune_train(self.model, self.train_data, (224, 224))
        self.save_model()

    def show_edge_heatmap(self, path_save):
        """Plots the edge heatmap of the neutral network via the current weight values
        For this specific model: the number of nodes before the output layer is 516,
        we want to reduce this number down to 8 neurons and the index of the layer before
        the outpout layer is 7.
        """
        cnn_net_weight_matrix_vis.cnn_network_edge_heatmap(
            self.model,
            number_nodes=128,
            number_condensed_nodes=8,
            window_size=16,
            pre_output_layer_idx=7,
            model_suffix="ResNet",
            path_save=path_save
        )

    def save_model(self) -> None:
        save_path = Path(__file__).parent / "Trained_Models/ResNet50V2/res_net_50_v2.h5"
        if save_path.exists():
            os.remove(save_path)
        self.model.save(save_path)




class XceptionTransfer(IModel):
    """Uses the best Xception model architecture"""

    def __init__(
        self,
        model_path=Path(__file__).parent / "Trained_Models/xception_transfer_model/"
                                           "xception_transfer.h5",
    ):
        self.data = None  # self.json_to_data_format()
        self.train_data = None
        self.model = keras.models.load_model(model_path)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def json_to_data_format(self, data: dict):
        return make_tensor(data, PATH, (150, 150, 3))

    def load_default_data(self) -> dict:
        pass

    def show_architecture(self):
        """Plots the Xception model architecture to the system viewer, can be modified to save it,
        removing the need to plot it dynamically."""

        cnn_net_arch_visualizer.cnn_visualize_net_minimalistic(
            self.model, scale_xy=1, scale_z=7, max_z=100
        )

    def predict(self, data: dict) -> dict:
        self.data = self.json_to_data_format(data)
        return predict_labels(self.data, self.model)

    def show_attention(self, img_path, path_save):
        """Predicts the label for the provided image and plots the saliency/attention of the model
        with regard for frontend display.
        to the predicted class, resnet (bool): needs to be set due to the different
        image preprocessing"""
        cnn_saliency_maps.cnn_visualize_saliency(self.model, img_path,
                                                 resnet=False, path_save=path_save)

    def train_model(self, data_train) -> None:
        """Used for the retraining process"""

        self.train_data = self.json_to_data_format(data_train)
        self.model = finetune_train(self.model, self.train_data, (150, 150))
        self.save_model()

    def show_edge_heatmap(self, path_save):
        """Plots the edge heatmap of the neutral network via the current weight values
        For this specific model: the number of weights before the output layer is 516,
        we want to reduce this number down to 8 neurons and the index of the layer before
        the outpout layer is 7.
        """
        cnn_net_weight_matrix_vis.cnn_network_edge_heatmap(
            self.model,
            number_nodes=128,
            number_condensed_nodes=8,
            window_size=16,
            pre_output_layer_idx=7,
            path_save=path_save
        )

    def save_model(self) -> None:
        save_path = (
            Path(__file__).parent / "Trained_Models/xception_transfer_model_refined"
        )
        if save_path.exists():
            shutil.rmtree(save_path)
        self.model.save(save_path)


class ClassicalML(IModel):
    def __init__(self,
                 model_path=Path(__file__).parent / "Trained_Models/ClassicML/"
                                                    "classic_ensemble_best.pkl"):
        self.image_list = None
        self.image_hash_list = None
        self.image_features = None
        self.model = self.load_model(model_path)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def json_to_data_format(self, data: dict) -> list:
        image_list = [
            str(PATH / image_dict["file_name"]) for image_dict in data["annotations"]
        ]
        return image_list

    def load_default_data(self) -> dict:
        pass

    def predict(self, data: dict) -> dict:  # 'data' is dictionary of image paths (+/-) labels
        """
        Prediction of given data

            Parameters:
                data (dict): dictionary of image paths
            Returns:
                annotations (dict) : dictionary of image paths including labels
        """
        self.image_list = self.json_to_data_format(data)
        self.image_hash_list = self.get_hash(self.image_list)
        annotations, self.image_features = predict_with_classic_ml(
            self.image_list, self.model
        )
        return annotations  # dict with image paths + labels

    def train_model(self, data: dict):  # 'data' is dictionary of image paths + labels
        """
        Retrain classic ML model and save retrained model

            Parameters:
                data (dict): dictionary of image paths
            Returns:
                self
        """
        self.image_list = self.json_to_data_format(data)
        image_hash_list = self.get_hash(self.image_list)
        # Check if data was labeled by model or manually
        # how to distinguish? e.g. predict -> upload new images + manually label
        # -> retrain (self.image_features would be already set by previous run)
        if image_hash_list == self.image_hash_list:
            data_pca = transform_features(self.image_features, self.model)
        else:
            self.image_hash_list = image_hash_list
            data_pca = transform_features(extract_f(self.image_list), self.model)
        self.model = retrain_model_classic_ml(data, self.model, data_pca)
        self.save_model()
        return self

    def big_data_active_learning(self) -> None:
        pass

    def save_model(self) -> None:
        """
        Save retrained model
        """
        save_path = Path(__file__).parent / "Trained_Models/ClassicML/classic_ensemble_best.pkl"
        # Overwrite latest model
        with open(save_path, 'wb') as outp:
            pickle.dump(self.model, outp, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_model(model_path):
        with open(model_path, "rb") as inp:
            loaded_model = pickle.load(inp)
        return loaded_model

    @staticmethod
    def get_hash(img_path_list):
        """
        Compare hash of image lists to provide faster prediction if image are retrained after
        prediction.
        """
        hash_img_list = []
        for img_path in img_path_list:
            # This function will return the `md5` checksum for any input image.
            with open(img_path, "rb") as f:
                img_hash = hashlib.md5()
                while chunk := f.read(8192):
                    img_hash.update(chunk)
            hash_img_list.append(img_hash.hexdigest())
        return hash_img_list


def Factory(model_name) -> IModel:
    """Constructs a Model based on the user's preference."""
    factories = {
        "Xception_Transfer": XceptionTransfer,
        "ResNet50V2": ResNet50V2,
        "Classical_ML": ClassicalML,
    }
    return factories[model_name]()


if __name__ == "__main__":

    model.show_edge_heatmap(path_save="./")
