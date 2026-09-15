"""
RoadVision AI
Standalone road graph generation script.
"""

import os
import sys

import numpy as np
from PIL import Image

from src.graph.graph_engine import build_road_graph
from src.graph.graph_visualization import create_road_graph_visualization


def load_mask(mask_path):
    """Load a binary road mask."""

    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Mask not found: {mask_path}"
        )

    image = Image.open(mask_path).convert("L")

    mask = np.array(image)

    # Convert grayscale mask into binary mask.
    mask = (mask > 127).astype(np.uint8)

    return mask


def main():
    mask_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/processed/prediction.png"
    )

    output_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "data/processed/road_graph.png"
    )

    print("=" * 60)
    print("ROADVISION AI - ROAD GRAPH GENERATION")
    print("=" * 60)

    print(f"\nInput mask : {mask_path}")
    print(f"Output     : {output_path}")

    mask = load_mask(mask_path)

    print("\nBuilding road graph...")

    result = build_road_graph(
        mask,
        max_gap_distance=20,
        max_gap_angle=60,
    )

    graph = result["graph"]

    print("\nGRAPH RESULTS")
    print("-" * 40)
    print("Skeleton pixels :", int(result["skeleton"].sum()))
    print("Nodes            :", graph.number_of_nodes())
    print("Edges            :", graph.number_of_edges())
    print(
        "Healing edges    :",
        len(result.get("healing_edges", [])),
    )

    print("\nCreating visualization...")

    visualization = create_road_graph_visualization(
        graph_result=result,
        output_path=output_path,
    )

    print(f"Visualization saved to: {visualization}")

    print("\nDone.")


if __name__ == "__main__":
    main()