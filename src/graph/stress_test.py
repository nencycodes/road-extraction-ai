"""
RoadVision AI - Network Stress Testing

Simulates failure of the most critical road-network node
and measures the resulting impact on connectivity and travel time.
"""

import math
import networkx as nx

from src.graph.graph_engine import (
    calculate_criticality,
    connectivity_ratio,
    simulate_node_failure,
    find_emergency_route,
    route_travel_time,
    resilience_index,
)


# ============================================================
# CRITICAL NODE
# ============================================================

def find_critical_node(G):
    """
    Find the node with the highest betweenness centrality.

    Criticality is calculated on the largest connected
    component when the graph is fragmented.
    """

    centrality = calculate_criticality(G)

    if not centrality:
        return None, 0.0, {}

    critical_node = max(
        centrality,
        key=centrality.get
    )

    critical_score = centrality[critical_node]

    return (
        critical_node,
        float(critical_score),
        centrality
    )


# ============================================================
# ROUTE ENDPOINTS
# ============================================================

def select_route_endpoints(G):
    """
    Select two nodes from the largest connected component.

    The selected nodes are approximately far apart in
    shortest-path distance so that the failure test has
    meaningful route impact.
    """

    if len(G) < 2:
        return None, None

    components = list(
        nx.connected_components(G)
    )

    if not components:
        return None, None

    largest_component = max(
        components,
        key=len
    )

    if len(largest_component) < 2:
        return None, None

    subgraph = G.subgraph(
        largest_component
    ).copy()

    nodes = list(subgraph.nodes())

    source = nodes[0]

    distances = nx.single_source_dijkstra_path_length(
        subgraph,
        source,
        weight="weight"
    )

    if not distances:
        return None, None

    target = max(
        distances,
        key=distances.get
    )

    return source, target


# ============================================================
# BASELINE ROUTE
# ============================================================

def calculate_baseline_route(
    G,
    source,
    target
):
    """
    Calculate the baseline shortest route and travel time.
    """

    route_result = find_emergency_route(
        G,
        source,
        target
    )

    if route_result is None:
        return {
            "route": None,
            "distance": None,
            "time": None
        }

    route = route_result["route"]
    distance = route_result["distance"]

    time = route_travel_time(
        G,
        route
    )

    return {
        "route": route,
        "distance": float(distance),
        "time": float(time)
    }


# ============================================================
# FAILED ROUTE
# ============================================================

def calculate_failed_route(
    failed_graph,
    source,
    target
):
    """
    Calculate the route after the critical node failure.
    """

    route_result = find_emergency_route(
        failed_graph,
        source,
        target
    )

    if route_result is None:
        return {
            "route": None,
            "distance": None,
            "time": None
        }

    route = route_result["route"]
    distance = route_result["distance"]

    time = route_travel_time(
        failed_graph,
        route
    )

    return {
        "route": route,
        "distance": float(distance),
        "time": float(time)
    }


# ============================================================
# STRESS TEST
# ============================================================

def run_stress_test(G):
    """
    Run a complete network resilience stress test.

    Steps:
        1. Find most critical node.
        2. Calculate baseline connectivity.
        3. Select route endpoints.
        4. Calculate baseline travel time.
        5. Remove critical node.
        6. Calculate post-failure connectivity.
        7. Calculate resilience index.
        8. Calculate alternate route / travel-time impact.
    """

    if G is None or len(G) == 0:
        return {
            "critical_node": None,
            "betweenness": 0.0,
            "baseline_connectivity": 0.0,
            "post_failure_connectivity": 0.0,
            "connectivity_loss": 0.0,
            "resilience_index": 0.0,
            "time_before": None,
            "time_after": None,
            "percentage_increase": None,
            "time_increase": None,
            "alternate_route": None,
            "failed_node": None,
        }

    # --------------------------------------------------------
    # 1. Critical node
    # --------------------------------------------------------

    critical_node, critical_score, centrality = (
        find_critical_node(G)
    )

    # --------------------------------------------------------
    # 2. Baseline connectivity
    # --------------------------------------------------------

    baseline_connectivity = connectivity_ratio(G)

    # --------------------------------------------------------
    # 3. Route endpoints
    # --------------------------------------------------------

    source, target = select_route_endpoints(G)

    # --------------------------------------------------------
    # 4. Baseline route
    # --------------------------------------------------------

    baseline_route = None

    if (
        source is not None
        and target is not None
    ):
        baseline_route = calculate_baseline_route(
            G,
            source,
            target
        )

    # --------------------------------------------------------
    # 5. Simulate critical-node failure
    # --------------------------------------------------------

    failed_graph = simulate_node_failure(
        G,
        critical_node
    )

    # --------------------------------------------------------
    # 6. Post-failure connectivity
    # --------------------------------------------------------

    post_failure_connectivity = connectivity_ratio(
        failed_graph
    )

    connectivity_loss = (
        baseline_connectivity
        - post_failure_connectivity
    )

    connectivity_loss_percent = (
        connectivity_loss * 100.0
    )

    # --------------------------------------------------------
    # 7. Path resilience
    # --------------------------------------------------------

    path_resilience = resilience_index(
        G,
        failed_graph
    )

    # --------------------------------------------------------
    # 8. Alternate route
    # --------------------------------------------------------

    failed_route = None

    if (
        source is not None
        and target is not None
    ):
        failed_route = calculate_failed_route(
            failed_graph,
            source,
            target
        )

    # --------------------------------------------------------
    # 9. Travel-time comparison
    # --------------------------------------------------------

    time_before = None
    time_after = None
    time_increase = None
    percentage_increase = None
    alternate_route = None

    if baseline_route is not None:
        time_before = baseline_route["time"]

    if (
        failed_route is not None
        and failed_route["route"] is not None
    ):
        time_after = failed_route["time"]
        alternate_route = failed_route["route"]

        if (
            time_before is not None
            and time_before > 0
        ):
            time_increase = (
                time_after - time_before
            )

            percentage_increase = (
                time_increase
                / time_before
                * 100.0
            )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return {
        "critical_node": critical_node,
        "betweenness": float(critical_score),

        "baseline_connectivity": float(
            baseline_connectivity
        ),

        "post_failure_connectivity": float(
            post_failure_connectivity
        ),

        "connectivity_loss": float(
            connectivity_loss_percent
        ),

        "resilience_index": float(
            path_resilience
        ),

        "time_before": time_before,
        "time_after": time_after,

        "time_increase": time_increase,
        "percentage_increase": percentage_increase,

        "alternate_route": alternate_route,

        "failed_node": critical_node,

        "source": source,
        "target": target,

        "baseline_route": (
            baseline_route["route"]
            if baseline_route is not None
            else None
        ),

        "failed_route": (
            failed_route["route"]
            if failed_route is not None
            else None
        ),

        "centrality": centrality,
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("ROADVISION NETWORK STRESS TEST")
    print("=" * 60)

    print()
    print("This module is intended to be called")
    print("through run_stress_test.py or the backend pipeline.")