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


@app.get("/")
def home():
    return {"status": "ok", "message": "Backend is running!"}


@app.post("/api/process-photo")
async def process_photo(req: PhotoRequest):
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise HTTPException(status_code=500, detail="Replicate token not set")

        # Настраиваем клиент с увеличенным таймаутом (120 секунд)
        client = replicate.Client(api_token=api_token, timeout=120.0)

        # Вызываем молниеносную модель flux-schnell
        output = client.run(
            "black-forest-labs/flux-schnell",
            input={
                "prompt": "a professional headshot of a handsome man wearing a luxury dark business suit, white shirt and tie, clean background, passport photo style, high details",
                "go_fast": True,
                "megapixels": "1",
                "num_outputs": 1,
                "aspect_ratio": "1:1",
                "output_format": "webp",
                "output_quality": 90,
            },
        )

        if output:
            results = list(output)
            if len(results) > 0:
                return {"status": "success", "output_url": str(results[0])}

        return {"status": "error", "error": "Изображение не сгенерировано"}

    except Exception as e:
        return {"status": "error", "error": str(e)}
