import os
import replicate
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PhotoRequest(BaseModel):
    image: str


# Картинка-шаблон (костюм)
TARGET_SUIT_IMAGE = "https://images.unsplash.com/photo-1507679799987-c73779587ccf?q=80&w=1000&auto=format&fit=crop"


@app.get("/")
def home():
    return {"status": "ok", "message": "Face Swap Backend"}


@app.post("/api/process-photo")
async def process_photo(req: PhotoRequest):
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise HTTPException(status_code=500, detail="Replicate token not set")

        client = replicate.Client(api_token=api_token, timeout=120.0)

        # Используем доступную модель Face Swap
        output = client.run(
            "subminds/face-swap:9e30a5822e1b19102c7102554746f36531be3ee037e909d94f29a03195f4c20f",
            input={
                "target_image": TARGET_SUIT_IMAGE,  # Шаблон в костюме
                "swap_image": req.image,            # Твоё лицо
            },
        )

        if output:
            return {"status": "success", "output_url": str(output)}

        return {"status": "error", "error": "Не удалось пересадить лицо"}

    except Exception as e:
        return {"status": "error", "error": str(e)}
