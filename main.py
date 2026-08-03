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


TARGET_SUIT_IMAGE = "https://images.unsplash.com/photo-1507679799987-c73779587ccf?q=80&w=1000&auto=format&fit=crop"


@app.get("/")
def home():
    return {"status": "ok", "message": "Face Swap Backend is Ready!"}


@app.post("/api/process-photo")
async def process_photo(req: PhotoRequest):
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise HTTPException(status_code=500, detail="Replicate token not set")

        client = replicate.Client(api_token=api_token, timeout=120.0)

        # Обрабатываем изображение: отрезаем префикс base64 и преобразуем в файловый поток
        raw_image_data = req.image
        if "," in raw_image_data:
            raw_image_data = raw_image_data.split(",")[1]

        image_bytes = base64.b64decode(raw_image_data)
        image_file = io.BytesIO(image_bytes)

        output = client.run(
            "codeplugtech/face-swap:278a81e7ebb22db98bcba54de985d22cc1abeead2754eb1f2af717247be69b34",
            input={
                "input_image": TARGET_SUIT_IMAGE,
                "swap_image": image_file,
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
