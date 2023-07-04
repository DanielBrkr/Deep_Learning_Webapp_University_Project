"""
This module retrains the model using the re-train images from the frontend
"""
from pathlib import Path
from backend.helpers.cnn_helpers.create_train_set import create_train_folders
from backend.helpers.cnn_helpers.create_train_val_ds import create_train_val_ds
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import RMSprop


def finetune_train(model, data, input_tuple):
    """This function retrains the model"""

    path = Path(__file__).parent.parent.parent.parent / "Data/Correction_Train"

    fine_tune_epochs = 15
    metric = "accuracy"
    base_learning_rate = model.optimizer.learning_rate.numpy()
    unfreeze = 5
    lr_factor = 10
    functional_layer_nr = 3
    batch_size = 20

    create_train_folders(data)
    data_set = create_train_val_ds(
        path, batch_size=batch_size, image_size=input_tuple, val_split=None
    )
    data_set = data_set["train"]

    base_model = model.layers[functional_layer_nr]
    base_model.trainable = True
    num = len(base_model.layers)
    for layer in base_model.layers[: num - unfreeze]:
        layer.trainable = False
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),
        optimizer=RMSprop(learning_rate=base_learning_rate / lr_factor),
        metrics=[metric],
    )
    model.fit(data_set, epochs=fine_tune_epochs)

    return model
