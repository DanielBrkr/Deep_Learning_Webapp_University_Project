import visualkeras
from ann_visualizer.visualize import ann_viz


def cnn_visualize_net_minimalistic(model, scale_xy, scale_z, max_z):
    """Plots the model in an minimalistic fashion, take a look at the the examples below for usage
    visualkeras.layered_view(model.layers[3], scale_xy=100, scale_z=1, max_z=100, legend = True).
    show()

    Assuming the provided model is xception, model.layers[3] unfolds the hidden functional model c.f.
    model.layers[3].summary()

    visualkeras.layered_view(model_derp, scale_xy=1, scale_z=1, max_z=100, legend = True).show()
    The setup can be scaled as appropriate, as this is dependent on the actual architecture
    """

    visualkeras.layered_view(
        model, scale_xy=scale_xy, scale_z=scale_z, max_z=max_z, legend=False
    ).show()


def cnn_visalize_net_boxes(model):
    """Plots the model architecture with more precise parameters"""

    ann_viz(model, title="arg", view=True)


"""
cnn_visualize_net_minimalistic(model_derp)
cnn_visualize_net_minimalistic(model)
"""
