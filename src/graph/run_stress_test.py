"""
RoadVision AI
Standalone network resilience stress-test runner.
"""

import os
import sys

import numpy as np
from PIL import Image

from src.graph.graph_engine import build_road_graph
from src.graph.stress_test import run_stress_test


def load_mask(mask_path):
    """Load prediction mask."""

    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Mask not found: {mask_path}"
        )

    image = Image.open(mask_path).convert("L")

    mask = np.array(image)

    return (mask > 127).astype(np.uint8)


def main():

    mask_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "data/processed/prediction.png"
    )

    print("=" * 60)
    print("ROADVISION AI - NETWORK STRESS TEST")
    print("=" * 60)

    print(f"\nInput mask: {mask_path}")

    mask = load_mask(mask_path)

    print("\nBuilding road graph...")

    graph_result = build_road_graph(
        mask,
        max_gap_distance=20,
        max_gap_angle=60,
    )

    graph = graph_result["graph"]

    print(
        f"Graph: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )

    print("\nRunning network stress test...")

    result = run_stress_test(graph)

    print("\nSTRESS TEST RESULTS")
    print("-" * 40)

    print(
        "Critical node:",
        result.get("critical_node"),
    )

    print(
        "Betweenness:",
        result.get("betweenness"),
    )

    print(
        "Baseline connectivity:",
        result.get("baseline_connectivity"),
    )

    print(
        "Post-failure connectivity:",
        result.get("post_failure_connectivity"),
    )

    print(
        "Connectivity loss (%):",
        result.get("connectivity_loss"),
    )

    print(
        "Resilience index:",
        result.get("resilience_index"),
    )

    print(
        "Time before:",
        result.get("time_before"),
    )

    print(
        "Time after:",
        result.get("time_after"),
    )

    print(
        "Travel-time increase (%):",
        result.get("percentage_increase"),
    )

    if result.get("failed_route") is None:
        print("\nAlternate route: NOT AVAILABLE")
    else:
        print("\nAlternate route: AVAILABLE")

    print("\nStress test complete.")


if __name__ == "__main__":
    main()