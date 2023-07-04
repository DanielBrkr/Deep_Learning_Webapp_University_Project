"""This module creates an active learning pipeline"""
import random
from typing import Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import entropy


# pylint: disable=C0103, R0902, R0912, R0913, R0914, R0915
# C0103 (invalid-name):  Disabled since sklearn naming convention is the same
# R0902 (too-many-instance-attributes): Disabled since sklearn estimators also have >7 attributes
# R0912 (too-many-branches): Disabled since long method is no problem here. Can be improved.
# R0913 (too-many-arguments): Disabled since sklearn estimators also have >5 arguments (parameters)
# R0914 (too-many-locals): Disabled since long method is no problem here. Can be improved.
# R0915 (too-many-statements): Disabled since long method is no problem here. Can be improved.
class ActiveLearningClassifier(BaseEstimator):
    """
    A class to represent the complete active learning training pipeline.

        Attributes:
            train_size (int):
                size of train set to train the AL model with
            base_estimator:
                base model object
            ensemble (boolean):
                Whether to use ensemble as model or not
            ensemble_estimator_1:
                ensemble model 1
            ensemble_estimator_2:
                ensemble model 2
            strategy (str):
                Active Learning strategy
            in_split (str):
                Initial picking strategy
            budget (float):
                Active Learning budget
            itrain_perc (float):
                Percentage of training set to start Active Learning with
            val_loss (list):
                track validation loss if fitting is given a validation set
            val_accuracy (list):
                track validation accuracy if fitting is given a validation set
            pool_loss (list):
                track pool loss
            train_loss (list):
                track train loss
            itrain_samplesize (int):
                track the size of the initial train set after the Active Learning has stopped

        Methods:
            ensemble_predict_proba(X):
                Predict the probability scores for class labels of X for ensemble model
            fit(X,y,validation):
                Fit the model (single or ensemble) using Active Learning
            predict(X):
                Predict class labels of X
            predict_proba(X):
                Predict the probability scores for class labels of X
            score(X,y):
                Predict mean accuracy on X using y
    """
    def __init__(self, train_size: int, base_estimator, ensemble=False,
                 ensemble_estimator_1=SVC(probability=True),
                 ensemble_estimator_2=RandomForestClassifier(n_jobs=-1),
                 strategy='margin',
                 in_split='random',
                 budget=0.4,
                 itrain_perc=0.02):
        """
        Constructs all the necessary attributes for the ActiveLearningClassifier object.

            Parameters:
                train_size (int):
                    size of train set to train the AL model with
                base_estimator:
                    base model object
                ensemble (boolean):
                    Whether to use ensemble as model or not
                ensemble_estimator_1:
                    ensemble model 1
                ensemble_estimator_2:
                    ensemble model 2
                strategy (str):
                    Active Learning strategy
                in_split (str):
                    Initial picking strategy
                budget (float):
                    Active Learning budget
                itrain_perc (float):
                    Percentage of training set to start Active Learning with
        """

        self.train_size = train_size
        self.base_estimator = base_estimator
        self.base_estimator.random_state = 1
        self.ensemble = ensemble
        self.ensemble_estimator_1 = ensemble_estimator_1
        self.ensemble_estimator_1.random_state = 1
        self.ensemble_estimator_2 = ensemble_estimator_2
        self.ensemble_estimator_2.random_state = 1
        self.strategy = strategy          # default margin
        self.budget = budget              # default 0.25
        self.itrain_perc = itrain_perc    # default 0.025
        self.in_split = in_split  # default 'random'

        self.init_train_batch = None

        self.val_loss = []
        self.val_accuracy = []
        self.pool_loss = []
        self.train_loss = []
        self.itrain_samplesize = 0

    @property
    def strategy(self):
        """Getter method for <strategy> parameter"""
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        """Setter method for <strategy> parameter"""
        valid = {'margin', 'random', 'entropy', 'least_confidence'}  # limit the possible strategies
        if value not in valid:
            raise ValueError(f'<strategy> must be one of {valid}.')
        self._strategy = value

    @property
    def in_split(self):
        """Setter method for <in_split> parameter"""
        return self._in_split

    @in_split.setter
    def in_split(self, value):
        """Setter method for <in_split> parameter"""
        valid = {'random', 'uniform', 'dist'}  # limit the possible in_split strategy
        if value not in valid:
            raise ValueError(f'<in_split> must be one of {valid}.')
        self._in_split = value

    @property
    def itrain_perc(self):
        """Getter method for <itrain_perc> parameter"""
        return self._itrain_perc

    @itrain_perc.setter
    def itrain_perc(self, value):
        """Setter method for <itrain_perc> parameter"""
        if value <= 0 or value >= 1 or value >= self.budget:
            raise ValueError(
                "<itrain_perc> must be in the interval (0,1) and can NOT be >= <budget>")
        self._itrain_perc = value

    @property
    def budget(self):
        """Getter method for <budget> parameter"""
        return self._budget

    @budget.setter
    def budget(self, value):
        """Setter method for <budget> parameter"""
        if value <= 0 or value > 1:
            raise ValueError("<budget> must be in the interval (0,1]")
        self._budget = value

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y: pd.Series,
            validation: pd.DataFrame = None):
        """
        Fit given base_estimator/ensemble using Active Learning

            Parameters:
                X (np.ndarray / pd.Dataframe):
                    Train data to fit
                y (pd.Series):
                    Train labels
                validation (pd.Dataframe):
                    Optional validation set with labels

            Returns:
                self
        """

        print('AL fitting started')
        self.val_loss.clear()  # clear to be able to refit model
        self.pool_loss.clear()  # clear to be able to refit model
        self.train_loss.clear()  # clear to be able to refit model

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            y.reset_index(drop=True, inplace=True)  # Reset the index of y.
            # Only needed for CV with PCA pipeline since pca convert df to ndarray
            # and resets indices.
        train_data = pd.concat([X, y], axis=1)
        if validation is not None:
            X_val = validation.drop(columns=['Label'])
            y_val = validation['Label']

        # Choose subset of train data to start the active learning iteration
        # Randomly select/pick the initial train set from train pool
        if self.in_split == 'random':
            init_train_batch = train_data.sample(n=int(np.ceil(self.train_size * self.itrain_perc)),
                                                 random_state=1)  # , random_state=1
            samples_todrop = list(init_train_batch.index)
            train_pool = train_data.drop(samples_todrop, axis=0)  # drop chosen initial train batch

        # Assume uniform distribution of labels,
        # select/pick equal amount of labels per class for initial train set from train pool
        elif self.in_split == 'uniform':
            nlabel_perclass = int(np.ceil(self.train_size/4 * self.itrain_perc))
            labels = train_data.iloc[:, -1:]
            class_counts = labels.value_counts(sort=False)
            class_idxs = {
                str(class_label): train_data.index[train_data['Label'] == class_label].tolist() for
                class_label in range(4)}
            batch_idxs = []
            for class_label in range(4):
                if class_counts[class_label] < nlabel_perclass:
                    batch_idxs.append(random.Random(1).sample(class_idxs[str(class_label)],
                                                              class_counts[class_label]))
                    # random.Random(1)
                else:
                    batch_idxs.append(random.Random(1).sample(class_idxs[str(class_label)],
                                                              nlabel_perclass))  # random.Random(1)
            batch_idxs_flat = [x for xs in batch_idxs for x in xs]
            random.Random(1).shuffle(batch_idxs_flat)  # random.Random(1);
            # ^ shuffle the indexes of the small train batch with seed=1
            init_train_batch = train_data.iloc[batch_idxs_flat, :]
            train_pool = train_data.drop(batch_idxs_flat, axis=0)  # drop chosen initial train batch

        # random_split == 'dist'
        # Select/pick initial train set based on distribution of the labels in the whole train data
        else:
            labels = train_data.iloc[:, -1:]
            class_counts = labels.value_counts(sort=False)
            class_perc = np.ceil(class_counts * self.itrain_perc).astype(int).to_dict()
            class_idxs = {
                str(class_label): train_data.index[train_data['Label'] == class_label].tolist() for
                class_label in range(4)}
            batch_idxs = []
            for class_label in range(4):
                batch_idxs.append(random.Random(1).sample(class_idxs[str(class_label)], class_perc[
                    (class_label,)]))  # random.Random(1)
            batch_idxs_flat = [x for xs in batch_idxs for x in xs]
            random.Random(1).shuffle(batch_idxs_flat)  # random.Random(1);
            # ^ shuffle the indexes of the small train batch with seed=1

            init_train_batch = train_data.iloc[batch_idxs_flat, :]
            train_pool = train_data.drop(batch_idxs_flat, axis=0)  # drop chosen initial train batch

        # Check if there are 4 classes in the initial train set.
        # If not, add one sample of each missing class to the train set.
        cc_init_train = init_train_batch['Label'].value_counts().sort_index()
        if len(cc_init_train) != 4:
            all_classes = [0, 1, 2, 3]
            classes = list(cc_init_train.index.values)
            missing_classes = list(sorted(set(all_classes) - set(classes)))
            for sample in missing_classes:
                class_samples = train_pool.loc[train_pool['Label'] == sample]
                random_sample = class_samples.sample(random_state=1)  # random_state=1
                # ^ WITH random seed to not make starting samples different each time
                init_train_batch = pd.concat([init_train_batch, random_sample])
                train_pool.drop(random_sample.index, inplace=True)

        self.itrain_samplesize = init_train_batch.shape[0]

        # Active Learning iteration
        sample_num = self.train_size
        b = init_train_batch.shape[0]
        while b <= np.ceil(self.budget * sample_num - 1):
            print(f'budget = {b} ')
            # Create the train and test set for active learning
            X_train = np.array(init_train_batch.drop(['Label'], axis=1))
            y_train = np.array(init_train_batch['Label'])
            X_pool = np.array(train_pool.drop(['Label'], axis=1))
            y_pool = np.array(train_pool['Label'])

            self.base_estimator.fit(X_train, y_train)
            y_pool_score_b = self.base_estimator.predict_proba(X_pool)  # pred class probabilities
            y_train_score_b = self.base_estimator.predict_proba(X_train)
            if validation is not None:
                y_val_score_b = self.base_estimator.predict_proba(X_val)

            if self.ensemble:
                self.ensemble_estimator_1.fit(X_train, y_train)
                self.ensemble_estimator_2.fit(X_train, y_train)
                y_pool_score_e1 = self.ensemble_estimator_1.predict_proba(X_pool)
                y_train_score_e1 = self.ensemble_estimator_1.predict_proba(X_train)
                y_pool_score_e2 = self.ensemble_estimator_2.predict_proba(X_pool)
                y_train_score_e2 = self.ensemble_estimator_2.predict_proba(X_train)
                y_pool_score = np.mean(np.array([y_pool_score_b, y_pool_score_e1, y_pool_score_e2]),
                                       axis=0)
                y_train_score = np.mean(
                    np.array([y_train_score_b, y_train_score_e1, y_train_score_e2]), axis=0)
                if validation is not None:
                    y_val_score_e1 = self.ensemble_estimator_1.predict_proba(X_val)
                    y_val_score_e2 = self.ensemble_estimator_2.predict_proba(X_val)
                    y_val_score = np.mean(np.array([y_val_score_b, y_val_score_e1, y_val_score_e2]),
                                          axis=0)
            else:
                y_pool_score = y_pool_score_b
                y_train_score = y_train_score_b
                if validation is not None:
                    y_val_score = y_val_score_b

            if validation is not None:
                y_val_pred = y_val_score.argmax(axis=1)
                y_val_acc = accuracy_score(y_val, y_val_pred)
                self.val_accuracy.append(y_val_acc)
                print(f'Validation accuracy: {y_val_acc}')

                self.val_loss.append(log_loss(y_val, y_val_score, labels=[0, 1, 2, 3]))

            self.pool_loss.append(log_loss(y_pool, y_pool_score, labels=[0, 1, 2, 3]))
            self.train_loss.append(log_loss(y_train, y_train_score, labels=[0, 1, 2, 3]))
            # pool_score = self.base_estimator.score(X_pool, y_pool)
            # print(f'Accuracy on train pool: {pool_score}')

            # Smallest margin sampling strategy
            if self.strategy == 'margin':
                y_pool_score_sort = y_pool_score.copy()
                y_pool_score_sort.sort(axis=1)  # sort the rows
                conf_margin = np.diff(y_pool_score_sort)[:, 2]  # Difference btw. top 2 conf. scores
                smallest_margin_idx = np.argmin(conf_margin)
                most_uncertain_sample = train_pool.iloc[[smallest_margin_idx]]

            # Random sampling strategy
            elif self.strategy == 'random':
                most_uncertain_sample = train_pool.sample(random_state=1)
                smallest_margin_idx = most_uncertain_sample.index

            # Entropy sampling strategy
            elif self.strategy == 'entropy':
                # Sampling according to an entropy threshold,
                # selecting the sample that produces the highest entropy score.
                entropy_samp = entropy(y_pool_score, axis=1)
                smallest_margin_idx = np.argmax(entropy_samp)
                most_uncertain_sample = train_pool.iloc[[smallest_margin_idx]]

            # Least confidence sampling strategy
            else:  # 'least_confidence'
                least_conf = 1 - y_pool_score.max(axis=1)
                smallest_margin_idx = np.argmax(least_conf)
                most_uncertain_sample = train_pool.iloc[[smallest_margin_idx]]

            init_train_batch = pd.concat([init_train_batch, most_uncertain_sample],
                                         axis=0)  # Add most uncertain sample to train batch
            if self.strategy == 'random':
                train_pool.drop(smallest_margin_idx, inplace=True)
            else:
                train_pool.drop(train_pool.index[smallest_margin_idx],
                                inplace=True)  # Drop most uncertain sample from train pool

            b += 1
        self.init_train_batch = init_train_batch  # save the training samples from active learning
        return self

    def ensemble_predict_proba(self, X: pd.DataFrame):
        """
        Predict the probability scores for class labels of X for ensemble model

            Parameters:
                X (pd.Dataframe): Data to predict

            Returns:
                y_score (ndarray): probability scores of class labels of X
        """

        if self.ensemble:
            y_score_b = self.base_estimator.predict_proba(np.array(X))
            y_score_e1 = self.ensemble_estimator_1.predict_proba(np.array(X))
            y_score_e2 = self.ensemble_estimator_2.predict_proba(np.array(X))
            y_score = np.mean(np.array([y_score_b, y_score_e1, y_score_e2]), axis=0)
            return y_score

        raise ValueError(
            "Active Learning with Ensemble not activated. "
            "Instantiate object with 'ensemble=True' to activate Ensemble")

    def predict(self, X: pd.DataFrame):
        """
        Predict class labels of X

            Parameters:
                X (pd.Dataframe): Data to predict
            Returns:
                y_pred (ndarray): predicted class labels of X
        """

        if self.ensemble:
            y_score = self.ensemble_predict_proba(X)
            y_pred = y_score.argmax(axis=1)
        else:
            y_pred = self.base_estimator.predict(np.array(X))
        return y_pred

    def predict_proba(self, X: pd.DataFrame):
        """
        Predict the probability scores for class labels of X

            Parameters:
                X (pd.Dataframe): Data to predict

            Returns:
                y_score (ndarray): probability scores of class labels of X
        """

        if self.ensemble:
            y_score = self.ensemble_predict_proba(X)
        else:
            y_score = self.base_estimator.predict_proba(np.array(X))
        return y_score

    def score(self, X: pd.DataFrame, y: pd.Series):
        """
        Predict mean accuracy on X using y

            Parameters:
                X (pd.Dataframe): Data to predict
            Returns:
                y_acc (float): mean accuracy score of prediction
        """

        if self.ensemble:
            y_score = self.ensemble_predict_proba(X)
            y_pred = y_score.argmax(axis=1)
            y_acc = accuracy_score(y, y_pred)
        else:
            y_acc = self.base_estimator.score(np.array(X), np.array(y))
        return y_acc
