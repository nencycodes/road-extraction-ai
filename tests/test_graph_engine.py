import numpy as np

from src.graph.graph_engine import (
    build_road_graph,
    calculate_criticality,
    connectivity_ratio,
    find_emergency_route,
)


# Create a simple road network:
#
#        │
#        │
# ───────┼───────
#        │
#        │
#
# The center is an intersection.
def create_test_road():

    mask = np.zeros((100, 100), dtype=np.uint8)

    # Horizontal road
    mask[48:53, 20:81] = 1

    # Vertical road
    mask[20:81, 48:53] = 1

    return mask


def main():

    mask = create_test_road()

    result = build_road_graph(
        mask,
        max_gap_distance=12,
        max_gap_angle=60
    )

    skeleton = result["skeleton"]
    graph = result["graph"]
    endpoints = result["endpoints"]
    junctions = result["junctions"]
    healing_edges = result["healing_edges"]

    print("=" * 50)
    print("ROADVISION GRAPH ENGINE TEST")
    print("=" * 50)

    print("Skeleton pixels:", int(skeleton.sum()))
    print("Endpoints:", len(endpoints))
    print("Junction pixels:", len(junctions))

    print("Graph nodes:", graph.number_of_nodes())
    print("Graph edges:", graph.number_of_edges())

    print("Healing connections:", len(healing_edges))

    print(
        "Connected:",
        __import__("networkx").is_connected(graph)
        if len(graph) > 0
        else False
    )

    print(
        "Connectivity ratio:",
        round(connectivity_ratio(graph), 4)
    )

    criticality = calculate_criticality(graph)

    print(
        "Criticality nodes:",
        len(criticality)
    )

    print("\nTop critical nodes:")

    for node, score in sorted(
        criticality.items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]:

        print(
            " ",
            node,
            "→",
            round(score, 4)
        )

    # Test routing between two graph nodes
    nodes = list(graph.nodes())

    if len(nodes) >= 2:

        source = nodes[0]
        target = nodes[-1]

        route = find_emergency_route(
            graph,
            source,
            target
        )

        print("\nRoute test:")

        if route is not None:

            print(
                "Source:",
                source
            )

            print(
                "Target:",
                target
            )

            print(
                "Route nodes:",
                len(route["route"])
            )

            print(
                "Distance:",
                round(route["distance"], 2)
            )

        else:

            print("No route found.")

    print("\n" + "=" * 50)
    print("GRAPH ENGINE TEST COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()