"""This module creates the Radiomics features"""
from typing import List
from typing import Tuple
from typing import Union
import logging
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.utils.image_utils import img_to_array
from keras.utils.image_utils import load_img
import SimpleITK as sitk
from radiomics import featureextractor
# import cv2

warnings.filterwarnings("ignore")


class RadiomicsFeatures:
    """
    This class is for extracting radiomics features

        Attributes:
            extractor: radiomics extractor object

        Methods:
            create_radiomics_extractor():
                Create a radiomics feature extractor object
            extract_features(data):
                Extract features from the given data
            get_train_test_features(train_set, test_set):
                Extract features from train set and test set data
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the RadiomicsFeatures object.
        """
        self.extractor = self.create_radiomics_extractor()

    @staticmethod
    def create_radiomics_extractor():
        """
        Static method to initialize a radiomics feature extractor

            Returns:
                extractor: RadiomicsFeatureExtractor object
        """

        # Create feature extractor object
        #  Use of parameter file possible!
        extractor = featureextractor.RadiomicsFeatureExtractor()
        #  Image types used to extract features from
        extractor.enableImageTypeByName('LoG', customArgs={'sigma': [0.1, 1.0, 3.0, 5.0, 7.0, 10]})
        extractor.enableImageTypeByName('Square')
        extractor.enableImageTypeByName('Exponential')
        extractor.enableImageTypeByName('Wavelet')
        extractor.enableImageTypeByName('Gradient')
        #  Enabled features. 7 feature classes in total
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName('firstorder')
        extractor.enableFeatureClassByName('glcm')
        extractor.enableFeatureClassByName('gldm')
        extractor.enableFeatureClassByName('glrlm')
        extractor.enableFeatureClassByName('glszm')
        extractor.enableFeatureClassByName('ngtdm')
        extractor.enableFeatureClassByName('shape2D')

        # --> Possibility to change settings (define HOW the features are extracted)

        #  Print Settings, Image type and Features
        print('Extraction parameters:\n\t', extractor.settings)
        print('Enabled filters:\n\t', extractor.enabledImagetypes)
        print('Enabled features:\n\t', extractor.enabledFeatures)

        return extractor

    def get_train_test_features(self, train_set: List[Tuple[str, int]],
                                test_set: List[Tuple[str, int]]) -> Tuple[
                                pd.DataFrame, pd.DataFrame]:
        """
        Extract features from train set and test set data

            Parameters:
                train_set (list(tuple(str, int))):
                    list of tuples with image path and label
                test_set (list(tuple(str, int))):
                    list of tuples with image path and label

            Returns:
                train_features (pd.Dataframe):
                    Extracted features from train data
                test_features (pd.Dataframe):
                    Extracted features from test data
        """

        train_features = self.extract_features(train_set)
        test_features = self.extract_features(test_set)
        return train_features, test_features

    # Data is list of tuple (train/test set from DataSplit)
    def extract_features(self, data: Union[List[str], List[Tuple[str, int]]]) -> pd.DataFrame:
        """
        Extract features from the given data

            Parameters:
                data (list(str) or list(tuple(str, int))):
                    list of image paths to extract features from (with or without labels)
            Returns:
                feature_df (pd.Dataframe):
                    Extracted features in form of a dataframe
        """

        # pylint: disable-msg=too-many-locals
        label_list = []
        feature_df = pd.DataFrame()
        for idx, val in enumerate(data):
            # Extract features from images of the given data set (train set or test set)
            if all(isinstance(item, tuple) for item in data):
                image_path = val[0]
                image_label = val[1]
                label_list.append(image_label)
            else:
                image_path = val
            # image_number = image_path.split('/')[1].split('.')[0]

            # image = cv2.imread(image_path)
            # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image_load = load_img(image_path)
            image_tf = img_to_array(image_load).astype('uint8')
            image_tf = tf.convert_to_tensor(image_tf)

            # Uncomment to use with resizing images
            # Resize image to have the largest side max. 185 pixels
            # (chosen according to available images)
            # if tf.shape(image_tf).numpy()[0] > 185 or tf.shape(image_tf).numpy()[1] > 185:
            #    # Size needs to be a 1-D int32 Tensor of 2 elements: new_height, new_width
            #    size = tf.constant([185, 185], dtype=tf.int32)  # Default value of max. 185 pixels
            #    image_tf = tf.image.resize(image_tf, size, preserve_aspect_ratio=True)

            image_gray = tf.image.rgb_to_grayscale(image_tf)
            image_gray = image_gray.numpy().astype(np.uint8).reshape(image_gray.shape[:2])
            im_sitk = sitk.GetImageFromArray(image_gray)

            # PyRadiomics requires the image and mask to be a SimpleITK.Image object
            # or to be a string pointing to a single file containing the image/mask.
            # Create mask: Outer pixel frame is set to zero
            ma_arr = np.ones(im_sitk.GetSize()[::-1])
            ma_arr[:, 0] = 0
            ma_arr[:, ma_arr.shape[1] - 1] = 0
            ma_arr[0, :] = 0
            ma_arr[ma_arr.shape[0] - 1, :] = 0
            ma_sitk = sitk.GetImageFromArray(ma_arr)
            ma_sitk.CopyInformation(im_sitk)

            logger = logging.getLogger("radiomics.glcm")  # supress warnings of feature <glcm>
            logger.setLevel(logging.ERROR)

            # Extract features
            result_all = self.extractor.execute(im_sitk, ma_sitk)
            result = result_all.copy()
            for key in result_all.keys():
                if key.find('diagnostics') != -1:
                    del result[key]
                else:
                    break

            # Convert features into dataframe
            feature_df = pd.concat([feature_df, pd.DataFrame(result, index=[0])], axis=0,
                                   ignore_index=True)
            print(f'Finished! Image {idx} / {len(data)-1}')  # len(data_set)

        if all(isinstance(item, tuple) for item in data):
            # Add label vector at far right of the dataframe
            feature_df['Label'] = label_list
        return feature_df
