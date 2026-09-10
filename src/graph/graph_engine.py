import math
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize


# ============================================================
# DISJOINT SET / UNION-FIND
# ============================================================

class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

        return True


# ============================================================
# 8-CONNECTED NEIGHBOURS
# ============================================================

def get_neighbors(y, x, skeleton):
    h, w = skeleton.shape
    neighbors = []

    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):

            if dy == 0 and dx == 0:
                continue

            ny = y + dy
            nx_ = x + dx

            if 0 <= ny < h and 0 <= nx_ < w:
                if skeleton[ny, nx_]:
                    neighbors.append((ny, nx_))

    return neighbors


# ============================================================
# SKELETONIZATION
# ============================================================

def create_skeleton(binary_mask):
    binary_mask = np.asarray(binary_mask) > 0
    return skeletonize(binary_mask)


# ============================================================
# ENDPOINT + JUNCTION DETECTION
# ============================================================

def detect_key_pixels(skeleton):

    endpoints = []
    junctions = []

    ys, xs = np.where(skeleton)

    for y, x in zip(ys, xs):

        degree = len(
            get_neighbors(y, x, skeleton)
        )

        if degree == 1:
            endpoints.append((y, x))

        elif degree >= 3:
            junctions.append((y, x))

    return endpoints, junctions


# ============================================================
# CLUSTER NEARBY PIXELS
# ============================================================

def cluster_pixels(pixels, radius=3):

    if not pixels:
        return []

    pixels = list(pixels)
    visited = set()
    clusters = []

    for i in range(len(pixels)):

        if i in visited:
            continue

        stack = [i]
        visited.add(i)
        cluster = []

        while stack:

            current = stack.pop()
            cluster.append(pixels[current])

            cy, cx = pixels[current]

            for j in range(len(pixels)):

                if j in visited:
                    continue

                py, px = pixels[j]

                distance = math.sqrt(
                    (cy - py) ** 2 +
                    (cx - px) ** 2
                )

                if distance <= radius:
                    visited.add(j)
                    stack.append(j)

        clusters.append(cluster)

    return clusters


def cluster_centers(pixels, radius=3):

    clusters = cluster_pixels(
        pixels,
        radius
    )

    centers = []

    for cluster in clusters:

        ys = [p[0] for p in cluster]
        xs = [p[1] for p in cluster]

        centers.append(
            (
                int(round(np.mean(ys))),
                int(round(np.mean(xs)))
            )
        )

    return centers


# ============================================================
# PIXEL GRAPH
# ============================================================

def build_pixel_graph(skeleton):

    G = nx.Graph()

    ys, xs = np.where(skeleton)

    for y, x in zip(ys, xs):

        G.add_node(
            (y, x),
            pos=(x, y)
        )

    for y, x in zip(ys, xs):

        for ny, nx_ in get_neighbors(
            y,
            x,
            skeleton
        ):

            if G.has_edge(
                (y, x),
                (ny, nx_)
            ):
                continue

            distance = math.dist(
                (y, x),
                (ny, nx_)
            )

            G.add_edge(
                (y, x),
                (ny, nx_),
                weight=distance
            )

    return G


# ============================================================
# ENDPOINT DIRECTION
# ============================================================

def endpoint_direction(
    endpoint,
    skeleton,
    steps=6
):

    current = endpoint
    previous = None

    start_y, start_x = endpoint

    last_point = None

    for _ in range(steps):

        neighbors = get_neighbors(
            current[0],
            current[1],
            skeleton
        )

        if previous is not None:

            neighbors = [
                p for p in neighbors
                if p != previous
            ]

        if not neighbors:
            break

        # Choose the first continuation.
        # At an endpoint this gives the local road direction.
        next_point = neighbors[0]

        previous = current
        current = next_point

        last_point = current

    if last_point is None:
        return None

    dy = last_point[0] - start_y
    dx = last_point[1] - start_x

    return np.array(
        [dx, dy],
        dtype=float
    )


# ============================================================
# ANGLE CALCULATION
# ============================================================

def angle_between(v1, v2):

    if v1 is None or v2 is None:
        return 180.0

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 180.0

    cosine = np.dot(v1, v2) / (
        norm1 * norm2
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return math.degrees(
        math.acos(cosine)
    )


# ============================================================
# GAP CANDIDATES
# ============================================================

def generate_gap_candidates(
    endpoints,
    skeleton,
    max_distance=12,
    max_angle=60
):

    candidates = []

    directions = {}

    for endpoint in endpoints:

        directions[endpoint] = endpoint_direction(
            endpoint,
            skeleton
        )

    for i in range(len(endpoints)):

        a = endpoints[i]

        for j in range(i + 1, len(endpoints)):

            b = endpoints[j]

            distance = math.dist(a, b)

            if distance > max_distance:
                continue

            direction_a = directions[a]
            direction_b = directions[b]

            if (
                direction_a is None
                or direction_b is None
            ):
                continue

            # A -> B
            connecting_ab = np.array(
                [
                    b[1] - a[1],
                    b[0] - a[0]
                ],
                dtype=float
            )

            # B -> A
            connecting_ba = -connecting_ab

            angle_a = angle_between(
                direction_a,
                connecting_ab
            )

            angle_b = angle_between(
                direction_b,
                connecting_ba
            )

            if angle_a > max_angle:
                continue

            if angle_b > max_angle:
                continue

            angular_penalty = (
                angle_a + angle_b
            )

            cost = (
                distance
                + 0.15 * angular_penalty
            )

            candidates.append(
                {
                    "a": a,
                    "b": b,
                    "distance": distance,
                    "angle": angular_penalty,
                    "cost": cost
                }
            )

    candidates.sort(
        key=lambda item: item["cost"]
    )

    return candidates


# ============================================================
# MST + DISJOINT SET GAP HEALING
# ============================================================

def heal_gaps(
    G,
    skeleton,
    endpoints,
    max_distance=12,
    max_angle=60
):

    healed_graph = G.copy()

    candidates = generate_gap_candidates(
        endpoints,
        skeleton,
        max_distance=max_distance,
        max_angle=max_angle
    )

    if not candidates:
        return healed_graph, []

    # Each connected component gets an ID.
    components = list(
        nx.connected_components(
            healed_graph
        )
    )

    component_id = {}

    for idx, component in enumerate(components):

        for node in component:
            component_id[node] = idx

    dsu = DisjointSet(
        len(components)
    )

    healing_edges = []

    for candidate in candidates:

        a = candidate["a"]
        b = candidate["b"]

        if a not in component_id:
            continue

        if b not in component_id:
            continue

        component_a = component_id[a]
        component_b = component_id[b]

        # Only connect different components.
        if component_a == component_b:
            continue

        # Kruskal / MST logic.
        if dsu.union(
            component_a,
            component_b
        ):

            healed_graph.add_edge(
                a,
                b,
                weight=candidate["distance"],
                healed=True,
                angle=candidate["angle"]
            )

            healing_edges.append(
                candidate
            )

    return healed_graph, healing_edges


# ============================================================
# COMPRESSED ROAD GRAPH
# ============================================================

def build_road_graph(
    binary_mask,
    max_gap_distance=12,
    max_gap_angle=60
):

    skeleton = create_skeleton(
        binary_mask
    )

    endpoints, junctions = detect_key_pixels(
        skeleton
    )

    pixel_graph = build_pixel_graph(
        skeleton
    )

    healed_graph, healing_edges = heal_gaps(
        pixel_graph,
        skeleton,
        endpoints,
        max_distance=max_gap_distance,
        max_angle=max_gap_angle
    )

    return {
        "skeleton": skeleton,
        "endpoints": endpoints,
        "junctions": junctions,
        "pixel_graph": pixel_graph,
        "graph": healed_graph,
        "healing_edges": healing_edges
    }


# ============================================================
# CENTRALITY / CRITICALITY
# ============================================================

def calculate_criticality(G):

    if len(G) == 0:
        return {}

    if not nx.is_connected(G):

        largest_component = max(
            nx.connected_components(G),
            key=len
        )

        G_analysis = G.subgraph(
            largest_component
        ).copy()

    else:
        G_analysis = G

    centrality = nx.betweenness_centrality(
        G_analysis,
        weight="weight"
    )

    return centrality


# ============================================================
# RESILIENCE
# ============================================================

def average_shortest_path(G):

    if len(G) < 2:
        return float("inf")

    if not nx.is_connected(G):
        return float("inf")

    return nx.average_shortest_path_length(
        G,
        weight="weight"
    )


def resilience_index(
    baseline_graph,
    failed_graph
):

    baseline_apl = average_shortest_path(
        baseline_graph
    )

    failed_apl = average_shortest_path(
        failed_graph
    )

    if (
        not math.isfinite(baseline_apl)
        or not math.isfinite(failed_apl)
        or failed_apl == 0
    ):
        return 0.0

    return (
        baseline_apl
        / failed_apl
    )


# ============================================================
# CONNECTIVITY RATIO
# ============================================================

def connectivity_ratio(G):

    if len(G) == 0:
        return 0.0

    components = list(
        nx.connected_components(G)
    )

    largest = max(
        len(component)
        for component in components
    )

    return largest / len(G)


# ============================================================
# FAILURE SIMULATION
# ============================================================

def simulate_node_failure(
    G,
    node
):

    failed_graph = G.copy()

    if node in failed_graph:
        failed_graph.remove_node(node)

    return failed_graph


# ============================================================
# EMERGENCY ROUTE
# ============================================================

def find_emergency_route(
    G,
    source,
    target
):

    if (
        source not in G
        or target not in G
    ):
        return None

    try:

        route = nx.shortest_path(
            G,
            source,
            target,
            weight="weight"
        )

        distance = nx.shortest_path_length(
            G,
            source,
            target,
            weight="weight"
        )

        return {
            "route": route,
            "distance": distance
        }

    except nx.NetworkXNoPath:

        return None


# ============================================================
# TRAVEL TIME
# ============================================================

def distance_to_travel_time(
    distance,
    speed_kmh=40
):

    # Graph distance is currently in pixels.
    # We use a normalized conversion for simulation.
    #
    # 1 pixel = 1 metre for demonstration purposes.

    distance_km = distance / 1000.0

    time_hours = (
        distance_km / speed_kmh
    )

    return time_hours * 60.0


def route_travel_time(
    G,
    route,
    speed_kmh=40
):

    if route is None or len(route) < 2:
        return 0.0

    distance = 0.0

    for i in range(
        len(route) - 1
    ):

        a = route[i]
        b = route[i + 1]

        distance += G[a][b]["weight"]

    return distance_to_travel_time(
        distance,
        speed_kmh
    )