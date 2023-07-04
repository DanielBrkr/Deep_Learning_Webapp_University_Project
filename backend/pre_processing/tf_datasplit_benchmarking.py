import os
import numpy as np
import tensorflow as tf
from pathlib import Path


def load_validation_test_ds(image_size:tuple, active_learning_test : bool, val_test : bool, data_subfolder_string):
    """Generates a test set from the  designated test set folder, by setting the active_learning_test variable
    to True one can split the set further up to validate the performance for active learning or retraining in general """


    #image_size = (150, 150)
    batch_size = 32

    if active_learning_test == False and val_test == False:
        test_set_withold = None
        subset_state_1 = None
        subset_state_2 = None
    elif active_learning_test == True and val_test == False:
        test_set_withold = 0.5
        subset_state_1 = "training"
        subset_state_2 = "validation"
        print("Active learning validation test split on test set active")
    elif active_learning_test == False and val_test == True:
        test_set_withold = 0.2
        subset_state_1 = "training"
        subset_state_2 = "validation"
        print("Validation set split on train set active")


    working_dir = Path.cwd().resolve()
    print(f"Working directory: {working_dir}")
    root_path = working_dir.parent  # .resolve()#os.path.join(working_dir.parents)
    print(f"Project root path: {root_path}")
    data_path = os.path.join(root_path.parent, "Data")
    print(f"Data path: {data_path}")

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_path, data_subfolder_string),
        validation_split=test_set_withold,
        subset=subset_state_1,
        seed=24,  # spongebob squarepants season 3 episode 11
        image_size=image_size,
        batch_size=batch_size)

    active_learning_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_path, data_subfolder_string),
        validation_split=test_set_withold,
        subset=subset_state_2,
        seed=24,
        image_size=image_size,
        batch_size=batch_size,
    )

    print('Batches for testing -->', test_ds.cardinality().numpy())
    print(f'Batches withold for active learning validation --> {active_learning_ds.cardinality().numpy()},if #testing == #al_validation is the active_learning ds empty ')

    return test_ds, active_learning_ds


#load_validation_test_ds((150,150), False, 'sorted_data')

