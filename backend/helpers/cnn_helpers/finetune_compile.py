"""
This module configures and recompiles the model for retraining/refinement prior to retraining
"""
from tensorflow import keras


def finetune_compile(model, learning_rate):
    """This function takes the model and recompiles it, keep the learning rate well below 0.001"""

    model.compile(
        optimizer=keras.optimizers.Nadam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model
