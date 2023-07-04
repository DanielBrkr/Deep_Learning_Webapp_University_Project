"""
This function uses the train data set folders created by the DataSplit class
and creates the tensorflow train and validation set out of it
"""
from keras.utils import image_dataset_from_directory


def create_train_val_ds(
    path, batch_size=20, image_size=(224, 224), shuffle=False, val_split=0.2
):
    """Function returns train and data set for tensorflow models"""
    labels = "inferred"
    subset = "training"
    seed = 42

    data_sets = {"train": [], "validation": []}
    val_ds = False

    if not val_split:
        subset = None

    train_ds = image_dataset_from_directory(
        path,
        labels=labels,
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        shuffle=shuffle,
        validation_split=val_split,
        subset=subset,
    )

    if not val_split:
        data_sets["train"] = train_ds
        data_sets["validation"] = val_ds
        return data_sets

    val_ds = image_dataset_from_directory(
        path,
        labels=labels,
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
        shuffle=shuffle,
        validation_split=val_split,
        subset="validation",
    )

    data_sets["train"] = train_ds
    data_sets["validation"] = val_ds
    return data_sets
