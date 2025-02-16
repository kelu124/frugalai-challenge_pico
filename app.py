from fastapi import FastAPI
from dotenv import load_dotenv
from tasks import audio

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Frugal AI Challenge API",
    description="API for the Frugal AI Challenge evaluation endpoints - using a rp2040 for inference"
)

audio.checkmodelplace()

# Include all routers

app.include_router(audio.router)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Frugal AI Challenge API",
        "endpoints": {
            "audio": "/audio - Audio classification task (now around 0.87 accuracy)"
        },
        "details": {
            "title": "Running a real-time sound classifier algorithm for chainsaw detection on a 2$ microcontroler",
            "description": "Using the rp2040 signals processing with tflite to get small models running on cheap energy/materials setup.",
            "power_use": "It runs with 50mA at 5V, that's 0.25W power for real-time detection =)"
        },
        "demo_notebook": "https://huggingface.co/spaces/kelu124/pico-sound/blob/main/notebooks/template-audio.ipynb",
        "README.ME": "https://huggingface.co/spaces/kelu124/pico-sound/blob/main/README.md"
    }