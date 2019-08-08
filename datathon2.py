"""
The datathon package is a collection of helper functions used when running datathons.
"""

__version__ = "0.1.11"

import pandas as pd
import numpy as np
from sklearn import tree
import pydotplus
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import Image


def make_colormap(seq):
    """Return a LinearSegmentedColormap

    Args:
        seq (list): a sequence of floats and RGB-tuples. The floats should be
            increasing and in the interval (0,1).

    Returns:
        colormap (obj): matplotlib colormap.
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return matplotlib.colors.LinearSegmentedColormap('CustomMap', cdict)


def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the input map
    By @jakevdp: https://gist.github.com/jakevdp/91077b0cae40f8f8244a
    """
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def plot_model_pred_2d(mdl, X, y, cm=None, cbar=True, xlabel=None, ylabel=None,
    title=None, tight=True):
    """
    For a 2D dataset, plot the decision surface of a tree model.
    Based on scikit-learn tutorial plot_iris.html

    Args:
        mdl (obj): Model used for prediction.
        X (np.ndarray): 2D array of n predictor variables, shaped (n, 2).
        y (np.ndarray): 1D array of n outcomes, shaped (n,).
        cm (obj): Colormap.
        cbar (bool): Display the colorbar.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        title (str): A title for the plot.
        tight (bool): Apply plt.tight_layout to avoid overlapping plots.
    """
    # handle a dataframe as input
    if isinstance(X, pd.DataFrame):
        if xlabel is None:
            xlabel = X.columns[0]
        if ylabel is None:
            ylabel = X.columns[1]
        X = X.values

    if isinstance(y, pd.Series):
        y = y.values

    # create grid to match the size of the predictor values
    x0_min = X[:, 0].min()
    x0_max = X[:, 0].max()
    x1_min = X[:, 1].min()
    x1_max = X[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x0_min, x0_max, 100),
                         np.linspace(x1_min, x1_max, 100))

    # create vector of predictions using the xx, yy grid
    Z = mdl.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    if not cm:
        # custom colormap
        s = list()
        # e58139f9 - orange
        lo = np.array(matplotlib.colors.to_rgb('#e5813900'))
        # 399de5e0 - to blue
        hi = np.array(matplotlib.colors.to_rgb('#399de5e0'))
        for i in range(255):
            s.append(list((hi-lo)*(float(i)/255)+lo))
        cm = make_colormap(s)

    # plot the contour.
    # colour regions of the decision boundary
    N = len(set(y))
    plt.contourf(xx, yy, Z, cmap=discrete_cmap(N, base_cmap=cm))

    # plot the individual data points.
    # colour by the *true* outcome
    color = y.ravel()
    plt.scatter(X[:, 0], X[:, 1], c=color, edgecolor='k', linewidth=2,
                marker='o', s=60, cmap=discrete_cmap(N, base_cmap=cm))

    # add labels
    if xlabel is None:
        xlabel = "var1"
    if ylabel is None:
        ylabel = "var2"

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis("tight")

    plt.clim(-0.5, N - 0.5)
    if cbar:
        plt.colorbar(ticks=range(N))

    # title
    if title:
        plt.title(title)

    # avoid overlapping on subplots
    if tight:
        plt.tight_layout()


def create_graph(mdl, feature_names=None, cmap=None):
    """
    Display a graph of the decision tree.

    Args:
        mdl (obj): Model used for prediction.
        feature_names (list): Names of the features.
        cmap (obj): Colormap.

    Returns:
        graph (obj): Graphviz graph.

    Example usage:
      cmap = np.linspace(0.0, 1.0, 256, dtype=float)
      cmap = matplotlib.cm.coolwarm(cmap)
    """
    tree_graph = tree.export_graphviz(mdl, out_file=None,
                                      feature_names=feature_names,
                                      filled=True, rounded=True)
    graph = pydotplus.graphviz.graph_from_dot_data(tree_graph)

    # get colormap
    if cmap:
        # remove transparency
        if cmap.shape[1] == 4:
            cmap = cmap[:, 0:2]

        nodes = graph.get_node_list()
        for node in nodes:
            if node.get_label():
                # get number of samples in group 1 and group 2
                num_samples = [int(ii) for ii in node.get_label().split(
                    'value = [')[1].split(']')[0].split(',')]

                # proportion that is class 2
                cm_value = float(num_samples[1]) / float(sum(num_samples))
                # convert to (R, G, B, alpha) tuple
                cm_value = matplotlib.cm.coolwarm(cm_value)
                cm_value = [int(np.ceil(255*x)) for x in cm_value]
                color = '#{:02x}{:02x}{:02x}'.format(
                    cm_value[0], cm_value[1], cm_value[2])
                node.set_fillcolor(color)

    Image(graph.create_png())
    return graph


def prune(dt, min_samples_leaf=1):
    """
    Prune a tree model by setting node children to -1. Note: displaying the
    graph will still show all nodes.

    Args:
        dt (obj): The decision tree to be pruned.
        min_samples_leaf (int): Minimum number of samples.
    """
    # Pruning is done by the "min_samples_leaf" property of decision trees
    if dt.min_samples_leaf >= min_samples_leaf:
        print('The tree is already pruned at an equal or higher level.')
    else:
        # update prune parameter
        dt.min_samples_leaf = min_samples_leaf

        # loop through each node of the tree
        tree = dt.tree_
        for i in range(tree.node_count):
            n_samples = tree.n_node_samples[i]
            if n_samples <= min_samples_leaf:
                # we can't delete splits because they are fixed in the model
                # instead, we remove the split by setting the child values to -1
                tree.children_left[i] = -1
                tree.children_right[i] = -1

    return dt


def run_query(query, project_id):
    """
    Read data from BigQuery into a DataFrame.

    Args:
        query (str): The query string in standard BigQuery dialect.
        project_id (str): The Google Project ID used for billing purposes.

    Returns:
        df (pd.DataFrame): Results to the query.
    """
    return pd.io.gbq.read_gbq(query, project_id=project_id, dialect="standard")
