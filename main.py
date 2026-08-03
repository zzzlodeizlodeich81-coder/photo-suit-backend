import base64
import io
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
    return {"status": "ok", "message": "FLUX Photo Suit Backend is Live"}


@app.post("/api/process-photo")
async def process_photo(req: PhotoRequest):
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise HTTPException(
                status_code=500, detail="REPLICATE_API_TOKEN not set"
            )

        client = replicate.Client(api_token=api_token, timeout=120.0)

        # 1. Декодируем base64 во внутренний поток байтов
        raw_image_data = req.image
        if "," in raw_image_data:
            raw_image_data = raw_image_data.split(",")[1]

        image_bytes = base64.b64decode(raw_image_data)
        file_obj = io.BytesIO(image_bytes)

        # 2. Промпт для генерации идеального бизнес-костюма
        prompt = (
            "A professional studio portrait of the person from the input image, "
            "wearing a modern, perfectly tailored dark navy blue business suit with a crisp white shirt and tie. "
            "Cinematic studio lighting, 8k resolution, highly detailed face, photo realistic, sharp focus."
        )

        # 3. Передаем байтовый объект прямо в FLUX (SDK сам его упакует)
        output = client.run(
            "black-forest-labs/flux-dev",
            input={
                "image": file_obj,
                "prompt": prompt,
                "prompt_strength": 0.65,
                "num_inference_steps": 28,
                "guidance_scale": 3.5,
                "output_format": "jpg",
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

        return {
            "status": "error",
            "error": f"Модель не вернула картинку: {output}",
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}
