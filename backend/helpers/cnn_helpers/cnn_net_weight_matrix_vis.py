import os
from math import cos, sin, atan
import pandas as pd
from tensorflow import keras
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import palettable
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def cm2inch(value):
    """Quick maths"""
    return value / 2.54


def condense_weights_matrix(weights_dataframe, window_size, number_condensed_nodes):
    """Takes a datasframe of weights and aggregates them according to the formula below,
    the nodes of the weight matrix a downsampled according to the 'number_target_nodes' expression.
    The window on which the mean is calculated corresponds to the 'number_target_nodes'nodes' integer.

    example:    target_number_nodes = 8
                number_nodes = 516
    target_number_nodes = int(number_nodes / window_size)
    It's nice if the resulting shrinkage factor results in a whole number, otherwise round to the smallest one
    """

    # number_condensed_nodes = int(number_nodes / window_size)

    weights_df_reduced = pd.DataFrame()
    for i in range(1, number_condensed_nodes + 1):
        weights_df_window = weights_dataframe.iloc[i - 1 : window_size * i, :].mean(
            axis=0
        )
        weights_df_reduced = pd.concat([weights_df_reduced, weights_df_window], axis=1)

    weights_df_reduced = weights_df_reduced.to_numpy()

    return weights_df_reduced


def cnn_network_edge_heatmap(
    model,
    number_nodes,
    number_condensed_nodes,
    window_size,
    pre_output_layer_idx,
    path_save,
    model_suffix: str = "",
):
    """Takes the weight matrix from (previous layer to the output layer) and visualizes the weights by
    calculating the mean over the weight vector for each output node according to a window function.
     The "downsampled" weight matrix is subsequently plotted via a quasi 2d-line heatmap
     for frontend visualization

     number_condensed_nodes: e.g. from 128 -> (consdensed_weight_matrix) = 8"""

    global vertical_distance_between_layers, horizontal_distance_between_neurons, neuron_radius, number_of_neurons_in_widest_layer

    class Neuron:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def draw(self):
            circle = plt.Circle(
                (self.x, self.y),
                radius=neuron_radius,
                fill=False,
                zorder=5,
                linewidth=2.5,
            )
            plt.gca().add_patch(circle)

    class Layer:
        def __init__(self, network, number_of_neurons, weights):
            self.previous_layer = self.__get_previous_layer(network)
            self.y = self.__calculate_layer_y_position()
            self.neurons = self.__intialise_neurons(number_of_neurons)
            self.weights = weights
            # self.norm = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))

        def __intialise_neurons(self, number_of_neurons):
            neurons = []
            x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
            for iteration in range(number_of_neurons):
                neuron = Neuron(x, self.y)
                neurons.append(neuron)
                x += horizontal_distance_between_neurons
            return neurons

        def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
            return (
                horizontal_distance_between_neurons
                * (number_of_neurons_in_widest_layer - number_of_neurons)
                / 2
            )

        def __calculate_layer_y_position(self):
            if self.previous_layer:
                return self.previous_layer.y + vertical_distance_between_layers
            else:
                return 0

        def __get_previous_layer(self, network):
            if len(network.layers) > 0:
                return network.layers[-1]
            else:
                return None

        def __line_between_two_neurons(self, neuron1, neuron2, linewidth, weights):
            angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
            x_adjustment = neuron_radius * sin(angle)
            y_adjustment = neuron_radius * cos(angle)
            line_x_data = (neuron1.x - x_adjustment, neuron2.x + x_adjustment)
            line_y_data = (neuron1.y - y_adjustment, neuron2.y + y_adjustment)

            if linewidth != linewidth:
                linewidth_norm = 0
            else:
                cmap = palettable.matplotlib.Inferno_9.mpl_colormap
                vmin = np.min(np.abs(weights))
                vmax = np.max(np.abs(weights))
                norm = Normalize(vmin=vmin, vmax=vmax)
                linewidth_norm = (linewidth - np.min(np.abs(weights))) / (
                    np.max(np.abs(weights)) - np.min(np.abs(weights))
                )

            line = plt.Line2D(
                line_x_data,
                line_y_data,
                linewidth=linewidth_norm * 20,
                color=cmap(norm(linewidth_norm)),
                alpha=0.7,
                solid_capstyle="round",
                solid_joinstyle="round",
            )
            plt.gca().add_line(line)

        def draw(self, weights):
            for this_layer_neuron_index in range(len(self.neurons)):
                neuron = self.neurons[this_layer_neuron_index]
                neuron.draw()
                if self.previous_layer:
                    for previous_layer_neuron_index in range(
                        len(self.previous_layer.neurons)
                    ):
                        previous_layer_neuron = self.previous_layer.neurons[
                            previous_layer_neuron_index
                        ]
                        weight = self.previous_layer.weights[
                            this_layer_neuron_index, previous_layer_neuron_index
                        ]
                        self.__line_between_two_neurons(
                            neuron, previous_layer_neuron, weight, weights
                        )

    class NeuralNetwork:
        def __init__(self):
            self.layers = []

        def add_layer(self, number_of_neurons, weights=None):
            layer = Layer(self, number_of_neurons, weights)
            self.layers.append(layer)

        def draw(self):
            plt.rcParams["figure.subplot.left"] = 0
            plt.rcParams["figure.subplot.bottom"] = 0
            plt.rcParams["figure.subplot.right"] = 1
            plt.rcParams["figure.subplot.top"] = 1
            fig, axes = plt.subplots(
                nrows=1, ncols=1, figsize=(cm2inch(20), cm2inch(10))
            )

            for layer in self.layers:
                layer.draw(weights)

            divider = make_axes_locatable(axes)
            # cax = divider.append_axes("right", size="100%")

            norm = Normalize(vmin=0, vmax=1)
            cmap = palettable.matplotlib.Inferno_9.mpl_colormap
            cbar = plt.colorbar(
                matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                fraction=0.1,
                shrink=0.5,
                use_gridspec=True,
            )
            cbar.set_ticks([])

            plt.axis("scaled")
            axes.spines.top.set(visible=False)
            axes.spines.right.set(visible=False)
            axes.spines.left.set(visible=False)
            axes.spines.bottom.set(visible=False)
            plt.axis("off")

            # plt.show()

    # Setup of plotting parameters

    vertical_distance_between_layers = 7
    horizontal_distance_between_neurons = 4
    neuron_radius = 1
    number_of_neurons_in_widest_layer = 8

    weights = model.layers[pre_output_layer_idx].get_weights()[
        0
    ]  # it's 7 for xception, deduce the layer via model.summary() if needed

    weights_df = pd.DataFrame(weights, columns=["dmg1", "dmg2", "dmg3", "dmg4"])
    if model_suffix == "ResNet":
        pass

    else:
        # Damage Label correspond to different nodes on models trained with tensorflow data-split

        column_swap = ["dmg4", "dmg1", "dmg3", "dmg2"]
        weights_df.reindex(columns=column_swap)

    condensed_weights = condense_weights_matrix(weights_df, number_nodes, window_size)

    network = NeuralNetwork()
    network.add_layer(number_condensed_nodes, condensed_weights)
    network.add_layer(4)
    network.draw()
    # plt.show()

    plt.savefig(path_save, bbox_inches="tight")
