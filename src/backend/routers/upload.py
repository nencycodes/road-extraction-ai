from fastapi import APIRouter, UploadFile, File
from pathlib import Path
import shutil

from src.image_processing.image_reader import read_image
from src.image_processing.preprocessor import preprocess_image
from src.image_processing.edge_detector import detect_edges
from src.image_processing.image_saver import (
    save_processed_image,
    save_edge_image
)

router = APIRouter()

# Folder where original uploaded images are stored
UPLOAD_FOLDER = Path("data/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    # Step 1: Save uploaded image
    file_path = UPLOAD_FOLDER / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Step 2: Read image using OpenCV
    image = read_image(file_path)

    # Step 3: Get image dimensions
    height, width, channels = image.shape

    # Step 4: Preprocess image
    processed_image = preprocess_image(image)

    # Step 5: Save processed image
    processed_path = save_processed_image(
        processed_image,
        file.filename
    )

    # Step 6: Detect edges
    edge_image = detect_edges(processed_image)

    # Step 7: Save edge image
    edge_path = save_edge_image(
        edge_image,
        file.filename
    )

    # Step 8: Return response
    return {
        "message": "Pipeline completed successfully!",
        "filename": file.filename,
        "width": width,
        "height": height,
        "channels": channels,
        "uploaded_image": str(file_path),
        "processed_image": str(processed_path),
        "edge_image": str(edge_path)
    }