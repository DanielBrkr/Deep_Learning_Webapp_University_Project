import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import palettable
from sklearn.metrics import (
    precision_score,
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
)
from pathlib import Path
import time


def cm2inch(value):
    """converts cm to inch for plotting
    :param float value:
    """
    return value / 2.54


def run_evaluation_on_test(model, test_ds):
    """Calculates some statistical performance metrics with the provided model on the test set and returns them
    in the indicated sequence, for e.g. plotting
    :param object model: Model being test
    :param test_ds: The dataset on which the evaluation is performed with model

    :return: A list with the statistical metrics
    :rtype: array
    """

    def concat_tensor_and_labels():
        """Ist hier für Testzwecke um mit dem tf.Data Object die Test Peformance in um das test_ds in einem rutsch zu
        evaluieren"""

        predictions = np.array([])
        labels = np.array([])
        for x, y in test_ds:
            predictions = np.concatenate(
                [predictions, np.argmax(model.predict(x), axis=-1)]
            )
            labels = np.concatenate([labels, y.numpy()])

        return predictions, labels

    y_pred, label_batch = concat_tensor_and_labels()

    print(f"Size of the test set: {len(label_batch)}")

    accuracy = accuracy_score(label_batch, y_pred)
    balance_accuracy = balanced_accuracy_score(
        label_batch, y_pred
    )  # Balance accuracy for the unbalanced test dataset
    kappa = cohen_kappa_score(
        label_batch, y_pred
    )  # Overall accuracy of the model given the distributions of the target and predicted classes:
    recall = recall_score(label_batch, y_pred, average=None)  # True positive rate
    precision = precision_score(
        label_batch, y_pred, average=None
    )  # Ability of the classifier not to label as positive a sample that is negative.
    f1 = f1_score(
        label_batch, y_pred, average=None
    )  # Harmonic mean of precision & recall for each class individually

    array_met = np.concatenate((precision, recall))
    array_met = np.concatenate((array_met, f1))
    array_acc = np.array([accuracy, balance_accuracy, kappa])

    statistical_metrics = np.concatenate((array_acc, array_met))

    return statistical_metrics


def plot_test_statistical_metrics(statistical_metrics):
    """Plots the statistical metrics from the function run_evaluation_on_test()"""

    plt.rcParams["axes.titlepad"] = 20
    plt.rc("font", family="serif", size=16)
    fig3, axes3 = plt.subplots(
        nrows=1, ncols=1, figsize=(cm2inch(40), cm2inch(10))
    )  # title = 'Training Analysis')

    axes3.grid(which="major")
    width = 0.5

    axes3.set_prop_cycle("color", palettable.matplotlib.Plasma_12.mpl_colors)
    for i in range(15):
        axes3.bar(
            i,
            height=statistical_metrics[i],
            width=width,
            ecolor="black",
            lw=3,
            capsize=5,
            alpha=1,
            label="UR 10e",
            zorder=3,
        )

    plt.title("Statistical Metrics on the Test Set \n Custom Fine-Tuned Model", size=16)
    plt.xlabel(r"p = precision, r = recall, f1 = f1-score", size=16)
    plt.ylabel(r"%", size=16)
    #   plt.suptitle("p = precision", y=1.05, fontsize=18)

    plt.xticks(
        np.arange(0, 15, step=1),
        (
            r"accuracy",
            "balanced \n accuracy",
            "cohens kappa",
            "precision scratch",
            "precision dent",
            "precision rim",
            "precision other",
            "recall scratch",
            "recall dent",
            "recall rim",
            "recall other",
            "f1 scratch",
            "f1 dent",
            "f1 rim",
            "f1 other",
        ),
    )

    # Achtung je nach batch kann sich die Reihenfolge von den labels ändern?
    plt.setp(axes3.get_xticklabels(), rotation=45, horizontalalignment="right")

    axes3.set_ylim(0, 1)


def run_test_eval_10(model, dataset):
    n = 10
    metrics_row = []
    for i in range(n):
        metrics = run_evaluation_on_test(model, dataset)

        metrics_row.append(metrics)

    eval_df = pd.DataFrame(metrics_row)
    eval_df.columns = [
        r"accuracy",
        "balanced \n accuracy",
        "cohens kappa",
        "precision scratch",
        "precision dent",
        "precision rim",
        "precision other",
        "recall scratch",
        "recall dent",
        "recall rim",
        "recall other",
        "f1 scratch",
        "f1 dent",
        "f1 rim",
        "f1 other",
    ]

    print(eval_df.describe())

    return eval_df


def plot_test_validation_eval(eval_dataframe):
    """Plots the statistical metrics from the function run_evaluation_on_test()"""

    plt.rcParams["axes.titlepad"] = 20
    plt.rc("font", family="serif", size=16)
    fig3, axes3 = plt.subplots(
        nrows=1, ncols=1, figsize=(cm2inch(40), cm2inch(10))
    )  # title = 'Training Analysis')

    axes3.grid(which="major")
    width = 0.5

    eval_dataframe_stats = eval_dataframe.describe()

    get_time = time.strftime("%Y%m%d%%M%S")
    eval_dataframe_stats.describe().to_csv(get_time + "_eval_stats.csv")

    # reindexing alphabetical class name order (1. dent, 2. other, 3. rim, 4. scratch) to scratch dent rim other

    xception_column_swap = False

    if xception_column_swap == True:
        columns_to_swap = [
            "accuracy",
            "balanced accuracy",
            "cohens kappa",
            "p scratch",
            "p dent",
            "p rim",
            "p other",
            "r4",
            "r1",
            "r3",
            "r2",
            "f1_4",
            "f1_1",
            "f1_3",
            "f1_2",
        ]

        eval_dataframe_stats = eval_dataframe_stats.reindex(columns=columns_to_swap)

    else:
        pass

    eval_dataframe_stats.columns = [
        "accuracy",
        "balanced accuracy",
        "cohens kappa",
        "precision scratch",
        "precision dent",
        "precision rim",
        "precision other",
        "recall scratch",
        "recall dent",
        "recall rim",
        "recall other",
        "f1 scratch",
        "f1 dent",
        "f1 rim",
        "f1 other",
    ]

    axes3.set_prop_cycle("color", palettable.matplotlib.Plasma_12.mpl_colors)
    for i in range(15):
        axes3.bar(
            i,
            yerr=eval_dataframe_stats.iloc[2, i],
            height=eval_dataframe_stats.iloc[1, i],
            width=width,
            ecolor="black",
            lw=3,
            capsize=5,
            alpha=1,
            label="UR 10e",
            zorder=3,
        )

    plt.title("Statistical Metrics on the Test Set", size=16)
    # plt.xlabel(r"p = precision, r = recall, f1 = f1-score", size=16)
    plt.ylabel(r"%", size=16)
    #   plt.suptitle("p = precision", y=1.05, fontsize=18)

    plt.xticks(np.arange(0, 15, step=1), (list(eval_dataframe_stats.columns.values))),

    # Achtung je nach batch kann sich die Reihenfolge von den labels ändern?

    plt.setp(axes3.get_xticklabels(), rotation=45, horizontalalignment="right")

    axes3.set_ylim(0)

    plt.savefig(
        os.path.join("./eval_results" + get_time + "_performance_bars.pdf"),
        bbox_inches="tight",
    )

    return eval_dataframe_stats
