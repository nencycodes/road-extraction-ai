from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.backend.routers.upload import router as upload_router


app = FastAPI(
    title="RoadVision AI",
    description="AI-powered Road Extraction Backend",
    version="0.1.0"
)


# Register the upload router
app.include_router(upload_router)


# Serve generated prediction images
app.mount(
    "/processed",
    StaticFiles(directory="data/processed"),
    name="processed"
)


@app.get("/")
def home():
    return {
        "status": "running",
        "project": "RoadVision AI",
        "message": "Backend is working successfully!"
    }