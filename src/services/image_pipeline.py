"""
RoadVision AI

End-to-end image processing and road network intelligence pipeline.
"""

import os
from pathlib import Path

import numpy as np
import networkx as nx
from PIL import Image

from src.inference.predict import predict
from uuid import uuid4

from src.graph.graph_engine import (
    build_road_graph,
    calculate_criticality,
)

from src.graph.graph_visualization import (
    create_road_graph_visualization,
)

from src.graph.stress_test import run_stress_test


def make_json_safe(value):
    """
    Convert NumPy / tuple / array values into
    standard Python values that FastAPI can serialize.
    """

    if isinstance(value, dict):
        safe_items = {}

        for key, item in value.items():
            safe_key = make_json_safe(key)

            if isinstance(safe_key, (list, dict)):
                safe_key = str(safe_key)

            safe_items[safe_key] = make_json_safe(item)

        return safe_items

    if isinstance(value, (list, tuple)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    return value


class ImagePipeline:
    """
    Complete RoadVision processing pipeline.

    Flow:

        Upload
          ↓
        U-Net road extraction
          ↓
        Binary road mask
          ↓
        Skeletonization + graph construction
          ↓
        Criticality analysis
          ↓
        Network stress test
          ↓
        Frontend-ready results
    """

    MODEL_PATH = "src/models/ROADVISION_FINAL_BEST.pth"

    def process(self, file):

        # ====================================================
        # DIRECTORIES
        # ====================================================

        upload_dir = Path("data/uploads")
        output_dir = Path("data/processed")

        upload_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # ====================================================
        # SAVE UPLOADED IMAGE
        # ====================================================
        #
        # The original filename is kept for the API response,
        # but the file is written to disk under a unique name.
        #
        # Why: if the browser's source file already lives inside
        # data/uploads (same folder the backend writes into) and
        # we save using the original filename, the write can land
        # on the exact same path the browser is still streaming
        # from. Chrome detects the source file changing mid-upload
        # and aborts with ERR_UPLOAD_FILE_CHANGED, which surfaces
        # to the frontend as "Failed to fetch". A unique on-disk
        # name removes any chance of that path collision.

        filename = os.path.basename(
            file.filename
        )

        saved_filename = (
            f"{uuid4().hex}_{filename}"
        )

        file_path = upload_dir / saved_filename

        with open(file_path, "wb") as buffer:
            buffer.write(
                file.file.read()
            )

        # ====================================================
        # AI ROAD EXTRACTION
        # ====================================================

        mask, original_size = predict(
            str(file_path),
            model_path=self.MODEL_PATH
        )

        # ====================================================
        # SAVE ROAD MASK
        # ====================================================

        mask_image = Image.fromarray(
            (
                mask.numpy() * 255
            ).astype("uint8")
        )

        prediction_path = (
            output_dir / "prediction.png"
        )

        mask_image.save(
            prediction_path
        )

        # ====================================================
        # SAVE INPUT PREVIEW
        # ====================================================

        image = Image.open(
            file_path
        ).convert("RGB")

        image.thumbnail(
            (1200, 1200)
        )

        preview_path = (
            output_dir / "input_preview.jpg"
        )

        image.save(
            preview_path,
            "JPEG"
        )

        # ====================================================
        # BUILD ROAD GRAPH
        # ====================================================

        mask_np = mask.numpy().astype(
            "uint8"
        )

        graph_result = build_road_graph(
            mask_np,
            max_gap_distance=20,
            max_gap_angle=60
        )

        graph = graph_result["graph"]

        # ====================================================
        # GRAPH METRICS
        # ====================================================

        graph_nodes = graph.number_of_nodes()

        graph_edges = graph.number_of_edges()

        skeleton_pixels = int(
            graph_result["skeleton"].sum()
        )

        healing_connections = len(
            graph_result.get(
                "healing_edges",
                []
            )
        )

        component_count = (
            nx.number_connected_components(graph)
            if graph_nodes > 0
            else 0
        )

        # ====================================================
        # CRITICALITY ANALYSIS
        # ====================================================

        criticality = {}

        if graph_nodes > 0:

            centrality = calculate_criticality(
                graph
            )

            if centrality:

                critical_node = max(
                    centrality,
                    key=centrality.get
                )

                critical_score = float(
                    centrality[critical_node]
                )

                criticality = {
                    "critical_node": critical_node,
                    "betweenness": critical_score
                }

            else:

                criticality = {
                    "critical_node": None,
                    "betweenness": 0.0
                }

        else:

            criticality = {
                "critical_node": None,
                "betweenness": 0.0
            }

        # ====================================================
        # ROAD GRAPH VISUALIZATION
        # ====================================================

        graph_visualization_path = (
            output_dir / "road_graph.png"
        )

        create_road_graph_visualization(
            graph_result=graph_result,
            output_path=str(
                graph_visualization_path
            ),
            critical_node=criticality[
                "critical_node"
            ]
        )

        # ====================================================
        # NETWORK STRESS TEST
        # ====================================================

        stress_results = None
        stress_error = None

        if graph_nodes >= 2:

            try:

                stress_results = run_stress_test(
                    graph
                )

            except Exception as error:

                stress_error = str(error)

                print(
                    "Stress test warning:",
                    stress_error
                )

        # ====================================================
        # DEFAULT RESILIENCE VALUES
        # ====================================================

        resilience = {

            "baseline_connectivity": None,

            "post_failure_connectivity": None,

            "connectivity_loss": None,

            "resilience_index": None,

            "time_before": None,

            "time_after": None,

            "percentage_increase": None,

            "alternate_route": None
        }

        # ====================================================
        # COPY STRESS TEST RESULTS
        # ====================================================

        if stress_results:

            for key in resilience:

                if key in stress_results:

                    resilience[key] = (
                        stress_results[key]
                    )

            # Keep additional stress-test
            # information available.

            for key, value in stress_results.items():

                if key not in resilience:

                    resilience[key] = value

        # ====================================================
        # FINAL FRONTEND RESPONSE
        # ====================================================

        response = {

            "status": "prediction_successful",

            "filename": filename,

            "original_size": original_size,

            "prediction": (
                "/processed/prediction.png"
            ),

            "preview": (
                "/processed/input_preview.jpg"
            ),

            "graph": {

                "nodes": graph_nodes,

                "edges": graph_edges,

                "components": component_count,

                "skeleton_pixels": skeleton_pixels,

                "healing_connections": (
                    healing_connections
                ),

                "visualization": (
                    "/processed/road_graph.png"
                )
            },

            "criticality": criticality,

            "resilience": resilience,

            "stress_test_error": stress_error
        }

        # ====================================================
        # JSON-SAFE RESPONSE
        # ====================================================

        return make_json_safe(response)