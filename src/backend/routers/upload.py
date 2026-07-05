from fastapi import APIRouter, UploadFile, File
import shutil
from pathlib import Path

router = APIRouter()

# Folder where uploaded images will be saved
UPLOAD_FOLDER = Path("data/uploads")

# Create the folder if it doesn't exist
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):

    file_path = UPLOAD_FOLDER / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "message": "Image uploaded successfully!",
        "filename": file.filename,
        "saved_at": str(file_path)
    }