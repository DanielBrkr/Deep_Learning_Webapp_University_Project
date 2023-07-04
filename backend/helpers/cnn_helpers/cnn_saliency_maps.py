import os
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def cnn_visualize_saliency(model, image_path, resnet: bool, path_save):
    """Provides a routine to plot the "attention" of the model or in other words the relevant areas for the model
    with regard to the predicted class via the gradient-based approach c.f. "A saliency map is a way to measure the
     spatial support of a particular class in each image"
    """

    def input_img(image_path):
        image = tf.image.decode_jpeg(tf.io.read_file(image_path))
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, tf.float32)

        if resnet == True:
            image = tf.image.resize(
                image, [224, 224]
            )  #  e.g. [150, 150] muss nach model angepasst werden
        else:
            image = tf.image.resize(image, [150, 150])
        return image

    def normalize_image(img):
        grads_norm = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
        grads_norm = (grads_norm - tf.reduce_min(grads_norm)) / (
            tf.reduce_max(grads_norm) - tf.reduce_min(grads_norm)
        )
        return grads_norm

    def plot_attention_maps(img1, img2, vmin=0.3, vmax=0.7, mix_val=2):
        """
        Plots the "attention" of the model with regard to the predicted class via the
        gradient-based approach
        """

        plt.rc("font", family="serif", size=30)
        plt.rcParams["axes.titlepad"] = 15

        fig, axes = plt.subplots(
            nrows=1, ncols=3, figsize=(30, 10), gridspec_kw={"width_ratios": [2, 2, 2]}
        )
        axes[0].imshow(img2, cmap="gray", alpha=1)
        axes[1].imshow(img1, vmin=vmin, vmax=vmax, cmap="plasma")
        axes[2].imshow(img2, cmap="gray", alpha=0.5)
        axes[2].imshow(
            img1 * mix_val + img2 / mix_val, cmap="plasma", zorder=1, alpha=0.9
        )

        for ax in axes.flat:
            ax.axis("off")

        # die titel könnte man im frontend entfernen damits cleaner aussieht

        axes[0].title.set_text("Grayscaled Original Image")
        axes[1].title.set_text("Attention Map")
        axes[2].title.set_text("Superimposed Attention Map")

        fig.suptitle(
            "Saliency map with respect to the predicted class", size=46, y=0.99
        )

    image = input_img(image_path)

    result = model(image)
    max_idx = tf.argmax(result, axis=1)

    with tf.GradientTape() as tape:
        tape.watch(image)
        result = model(image)
        max_score = result[0, max_idx[0]]
    grads = tape.gradient(max_score, image)
    plot_attention_maps(normalize_image(grads[0]), normalize_image(image[0]))

    plt.savefig(path_save, bbox_inches="tight")
