"""This module scales and selects features from data"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin

# pylint: disable=C0103, W0613, R0914
# C0103 (invalid-name): has to be in this style since sklearn specification for transformers.
# W0613 (unused-argument): has to be given as argument since sklearn specification for transformers.
# R0914 (too-many-locals): not a problem here. Can be improved.


class FeatureSelection(BaseEstimator, TransformerMixin):
    """
    Class of Feature Selection to perform scaling and feature selection on given data.

        Attributes:
            remove_corrfeat (boolean):
                Additionally remove correlated features
            s_c (StandardScaler):
                StandardScaler object
            removed_features (list):
                list of features that were removed during feature selection
        Methods:
            fit(data):
                Perform scaling and feature selection on given data
            transform(data):
                Apply fitted StandardScaler and removed features on the data to transform it
    """

    def __init__(self, remove_corrfeat=False):  # remove_corrfeat default False
        """
        Constructs all the necessary attributes for the FeatureSelection object.

            Parameters:
                remove_corrfeat (boolean): Additionally remove correlated features
        """
        self.remove_corrfeat = remove_corrfeat
        self.s_c = StandardScaler()
        self.removed_features = None

    def fit(self, data: pd.DataFrame, y=None):  # data = X
        """
        Method to perform scaling and feature selection on given data

            Parameters
                data (pd.Dataframe): dataframe with features and labels in the last column
                remove_corrfeat (boolean): whether to remove correlated features or not

            Returns
                s_c (StandardScaler) : fitted StandardScaler object
                removed_features (list): removed features
                data_unique (pd.Dataframe): remaining features
        """
        data_sc = self.s_c.fit_transform(data)
        data_sc = pd.DataFrame(data_sc, columns=data.columns, index=data.index)

        # Preprocessing of the features
        removed_features = []

        # Remove constant features
        constant_filter = VarianceThreshold(threshold=0)
        constant_filter.fit(data_sc)
        constant_columns = [column for column in data_sc.columns if
                            column not in data_sc.columns[constant_filter.get_support()]]
        print(f'Constant features: {constant_columns}')
        print(f'Overall features: {data_sc.shape[1]} \n '
              f'Non-constant features: {len(data_sc.columns[constant_filter.get_support()])}')
        data_cf = data_sc.drop(labels=constant_columns, axis=1)  # inplace=False !
        removed_features.extend(constant_columns)

        # Remove quasi-constant features
        qconstant_filter = VarianceThreshold(threshold=0.01)
        qconstant_filter.fit(data_cf)
        qconstant_columns = [column for column in data_cf.columns if
                             column not in data_cf.columns[qconstant_filter.get_support()]]
        print(f'Quasi constant features: {qconstant_columns}')
        print(f'Overall features: {data_cf.shape[1]} \n '
              f'Non-quasi constant features: '
              f'{len(data_cf.columns[qconstant_filter.get_support()])}')
        data_qcf = data_cf.drop(labels=qconstant_columns, axis=1)  # inplace=False !
        removed_features.extend(qconstant_columns)

        # Remove duplicate features
        data_qcf_transpose = data_qcf.T
        data_unique = data_qcf_transpose.drop_duplicates(keep='first').T
        print(f'Shape of dataframe with duplicates: {data_qcf.shape}')
        print(f'Shape of dataframe without duplicates: {data_unique.shape}')
        duplicated_features = [dup_col for dup_col in data_qcf.columns if
                               dup_col not in data_unique.columns]
        print(f'Duplicated features: {duplicated_features}')
        removed_features.extend(duplicated_features)

        # Remove correlated features
        if self.remove_corrfeat:
            correlated_features = set()  # empty set to contain all correlated features
            correlation_matrix = data_unique.corr()
            for i, val in enumerate(correlation_matrix.columns):
                for j in range(i):
                    # collect samples where absolute correlation is > 0.9
                    if abs(correlation_matrix.iloc[i, j]) > 0.9:
                        correlated_features.add(val)
            print(f'Correlated Features: {correlated_features}')
            data_unique.drop(labels=correlated_features, axis=1, inplace=True)
            removed_features.extend(correlated_features)
        self.removed_features = removed_features
        return self

    def transform(self, data: pd.DataFrame, y=None):  # X=data
        """
        Transform given data

            Parameters:
                data (pd.Dataframe): data to transform
            Returns:
                data_transformed (pd.Dataframe): transformed data
        """
        data_sc = self.s_c.transform(data)
        data_sc = pd.DataFrame(data_sc, columns=data.columns, index=data.index)
        data_transformed = data_sc.drop(labels=self.removed_features, axis=1, inplace=False)
        return data_transformed
