from fastapi import APIRouter, UploadFile, File
from src.services.image_pipeline import ImagePipeline

router = APIRouter()

pipeline = ImagePipeline()


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    return pipeline.process(file)