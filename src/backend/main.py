from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from src.backend.routers.upload import router as upload_router


app = FastAPI(
    title="RoadVision AI",
    description="AI-powered Road Extraction Backend",
    version="0.1.0"
)


# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register upload router
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