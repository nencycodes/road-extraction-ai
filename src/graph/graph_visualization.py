"""
Road graph visualization utilities for RoadVision AI.

Creates a clean PNG visualization of the extracted road graph.
The visualization can be served by FastAPI and displayed by the frontend.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def _ensure_directory(path):
    """Create parent directory if it does not exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _get_xy(node):
    """
    Convert a graph node into x/y coordinates.

    RoadVision graph nodes are normally pixel coordinates:
        (row, col)

    We return:
        x = col
        y = row
    """
    if isinstance(node, (tuple, list)) and len(node) >= 2:
        return float(node[1]), float(node[0])

    if isinstance(node, np.ndarray) and node.size >= 2:
        return float(node[1]), float(node[0])

    return None, None


def create_road_graph_visualization(
    graph_result=None,
    output_path="data/processed/road_graph.png",
    graph=None,
    skeleton=None,
    critical_node=None,
):
    """
    Create a PNG visualization of the extracted road graph.

    Parameters
    ----------
    graph_result : dict, optional
        Result returned by build_road_graph().
        Expected keys:
            graph
            skeleton
            critical_node (optional)

    output_path : str
        Path where visualization PNG will be saved.

    graph : networkx.Graph, optional
        Graph can be passed directly.

    skeleton : numpy.ndarray, optional
        Skeleton image can be passed directly.

    critical_node : tuple, optional
        Node to highlight.

    Returns
    -------
    str
        Output image path.
    """

    if graph_result is not None:
        if graph is None:
            graph = graph_result.get("graph")

        if skeleton is None:
            skeleton = graph_result.get("skeleton")

        if critical_node is None:
            critical_node = graph_result.get("critical_node")

    if graph is None:
        graph = nx.Graph()

    _ensure_directory(output_path)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=160)

    # ---------------------------------------------------------
    # Background skeleton
    # ---------------------------------------------------------
    if skeleton is not None:
        skeleton_array = np.asarray(skeleton)

        if skeleton_array.ndim == 2:
            ax.imshow(
                skeleton_array,
                cmap="gray",
                origin="upper",
                interpolation="nearest",
            )

    # ---------------------------------------------------------
    # Draw graph edges
    # ---------------------------------------------------------
    for u, v in graph.edges():
        x1, y1 = _get_xy(u)
        x2, y2 = _get_xy(v)

        if x1 is None or x2 is None:
            continue

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=1.2,
            alpha=0.85,
        )

    # ---------------------------------------------------------
    # Draw graph nodes
    # ---------------------------------------------------------
    node_x = []
    node_y = []

    for node in graph.nodes():
        x, y = _get_xy(node)

        if x is not None:
            node_x.append(x)
            node_y.append(y)

    if node_x:
        ax.scatter(
            node_x,
            node_y,
            s=7,
            alpha=0.8,
        )

    # ---------------------------------------------------------
    # Highlight critical node
    # ---------------------------------------------------------
    if critical_node is not None and critical_node in graph:
        x, y = _get_xy(critical_node)

        if x is not None:
            ax.scatter(
                [x],
                [y],
                s=90,
                marker="o",
                facecolors="none",
                edgecolors="red",
                linewidths=2,
                zorder=10,
            )

    # ---------------------------------------------------------
    # Technical presentation
    # ---------------------------------------------------------
    ax.set_title(
        "ROAD NETWORK GRAPH",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    ax.set_xlabel("X / Column (pixels)")
    ax.set_ylabel("Y / Row (pixels)")

    ax.set_aspect("equal", adjustable="box")

    ax.grid(
        True,
        alpha=0.15,
        linewidth=0.5,
    )

    # Preserve image coordinates when skeleton exists.
    if skeleton is not None:
        height, width = np.asarray(skeleton).shape[:2]
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)

    fig.tight_layout()

    fig.savefig(
        output_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    return output_path


def visualize_graph(
    graph_result,
    output_path="data/processed/road_graph.png",
):
    """
    Backwards-compatible convenience wrapper.
    """
    return create_road_graph_visualization(
        graph_result=graph_result,
        output_path=output_path,
    )


if __name__ == "__main__":
    print(
        "graph_visualization.py provides visualization functions "
        "for the RoadVision graph pipeline."
    )