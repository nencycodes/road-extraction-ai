from fastapi import FastAPI
from src.backend.routers.upload import router as upload_router

app = FastAPI(
    title="RoadVision AI",
    description="AI-powered Road Extraction Backend",
    version="0.1.0"
)

# Register the upload router
app.include_router(upload_router)


@app.get("/")
def home():
    return {
        "status": "running",
        "project": "RoadVision AI",
        "message": "Backend is working successfully!"
    }