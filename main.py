import os
import io
import base64
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


# Фото отличного костюма для шаблона
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

        # Конвертируем входной base64 в поток
        raw_image_data = req.image
        if "," in raw_image_data:
            raw_image_data = raw_image_data.split(",")[1]

        image_bytes = base64.b64decode(raw_image_data)
        user_image_file = io.BytesIO(image_bytes)

        # Вызываем топовую модель easel/advanced-face-swap
        output = client.run(
            "easel/advanced-face-swap:95fa91eb008b8fbe7769efaa9c7c7fdd810cb955dfc0d640b388e2283cb0a544",
            input={
                "target_image": TARGET_SUIT_IMAGE,
                "swap_image": user_image_file,
            },
        )

        url = None
        if hasattr(output, "url"):
            url = str(output.url)
        elif isinstance(output, list) and len(output) > 0:
            url = str(output[0])
        elif isinstance(output, str):
            url = output

        if url:
            return {"status": "success", "output_url": url}

        return {"status": "error", "error": f"Пустой ответ от Replicate: {output}"}

    except Exception as e:
        return {"status": "error", "error": str(e)}
