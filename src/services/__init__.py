from pathlib import Path
import shutil

from src.image_processing.image_reader import read_image
from src.image_processing.preprocessor import preprocess_image
from src.image_processing.edge_detector import detect_edges
from src.image_processing.image_saver import (
    save_processed_image,
    save_edge_image,
)

UPLOAD_FOLDER = Path("data/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


class ImagePipeline:

    def process(self, file):

        # Step 1
        file_path = UPLOAD_FOLDER / file.filename

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 2
        image = read_image(file_path)

        # Step 3
        height, width, channels = image.shape

        # Step 4
        processed = preprocess_image(image)

        # Step 5
        processed_path = save_processed_image(
            processed,
            file.filename
        )

        # Step 6
        edges = detect_edges(processed)

        # Step 7
        edge_path = save_edge_image(
            edges,
            file.filename
        )

        return {
            "message": "Pipeline completed successfully!",
            "filename": file.filename,
            "width": width,
            "height": height,
            "channels": channels,
            "uploaded_image": str(file_path),
            "processed_image": str(processed_path),
            "edge_image": str(edge_path),
        }