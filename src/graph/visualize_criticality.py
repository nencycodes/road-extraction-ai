"""
RoadVision AI
Criticality / betweenness-centrality visualization.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image

from src.graph.graph_engine import (
    build_road_graph,
    calculate_criticality,
)


def load_mask(mask_path):
    """Load a binary prediction mask."""

    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Mask not found: {mask_path}"
        )

    image = Image.open(mask_path).convert("L")

    mask = np.array(image)

    return (mask > 127).astype(np.uint8)


def node_xy(node):
    """Convert (row, col) pixel coordinate to (x, y)."""

    if isinstance(node, (tuple, list)) and len(node) >= 2:
        return float(node[1]), float(node[0])

    return None, None


def create_criticality_visualization(
    graph,
    centrality,
    output_path,
    skeleton=None,
):
    """Create criticality map using betweenness centrality."""

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(12, 8),
        dpi=160,
    )

    # --------------------------------------------------------
    # Skeleton background
    # --------------------------------------------------------

    if skeleton is not None:
        ax.imshow(
            skeleton,
            cmap="gray",
            origin="upper",
            interpolation="nearest",
        )

    # --------------------------------------------------------
    # Graph edges
    # --------------------------------------------------------

    for u, v in graph.edges():

        x1, y1 = node_xy(u)
        x2, y2 = node_xy(v)

        if x1 is None or x2 is None:
            continue

        ax.plot(
            [x1, x2],
            [y1, y2],
            linewidth=0.7,
            alpha=0.35,
        )

    # --------------------------------------------------------
    # Centrality values
    # --------------------------------------------------------

    nodes = []
    xs = []
    ys = []
    values = []

    for node, score in centrality.items():

        x, y = node_xy(node)

        if x is None:
            continue

        nodes.append(node)
        xs.append(x)
        ys.append(y)
        values.append(float(score))

    # --------------------------------------------------------
    # Criticality points
    # --------------------------------------------------------

    if values:

        scatter = ax.scatter(
            xs,
            ys,
            c=values,
            cmap="hot",
            s=12,
            alpha=0.9,
        )

        fig.colorbar(
            scatter,
            ax=ax,
            label="Betweenness Centrality",
        )

        critical_node = max(
            centrality,
            key=centrality.get,
        )

        cx, cy = node_xy(critical_node)

        if cx is not None:

            ax.scatter(
                [cx],
                [cy],
                s=120,
                facecolors="none",
                edgecolors="red",
                linewidths=2.5,
                zorder=10,
            )

            ax.annotate(
                "CRITICAL NODE",
                (cx, cy),
                xytext=(10, -10),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
            )

    # --------------------------------------------------------
    # Technical presentation
    # --------------------------------------------------------

    ax.set_title(
        "ROAD NETWORK CRITICALITY MAP",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )

    ax.set_xlabel("X / Column (pixels)")
    ax.set_ylabel("Y / Row (pixels)")

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(
        True,
        alpha=0.15,
        linewidth=0.5,
    )

    if skeleton is not None:

        height, width = skeleton.shape[:2]

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


def main():

    mask_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/processed/prediction.png"
    )

    output_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "data/processed/criticality_map.png"
    )

    print("=" * 60)
    print("ROADVISION AI - CRITICALITY VISUALIZATION")
    print("=" * 60)

    mask = load_mask(mask_path)

    print("\nBuilding graph...")

    result = build_road_graph(
        mask,
        max_gap_distance=20,
        max_gap_angle=60,
    )

    graph = result["graph"]

    print(
        f"Graph: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

    print("\nCalculating betweenness centrality...")

    centrality = calculate_criticality(graph)

    print(
        "Critical node:",
        max(
            centrality,
            key=centrality.get,
        )
        if centrality
        else None,
    )

    create_criticality_visualization(
        graph=graph,
        centrality=centrality,
        skeleton=result.get("skeleton"),
        output_path=output_path,
    )

    print(
        f"\nCriticality map saved to: {output_path}"
    )


if __name__ == "__main__":
    main()