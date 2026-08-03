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


# Картинка шаблона костюма
TARGET_SUIT_IMAGE = "https://images.unsplash.com/photo-1507679799987-c73779587ccf?q=80&w=1000&auto=format&fit=crop"


@app.get("/")
def home():
    return {"status": "ok", "message": "Face Swap Backend is Online"}


@app.post("/api/process-photo")
async def process_photo(req: PhotoRequest):
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise HTTPException(
                status_code=500, detail="REPLICATE_API_TOKEN not set"
            )

        client = replicate.Client(api_token=api_token, timeout=120.0)

        # Декодируем base64 в байты
        raw_image_data = req.image
        if "," in raw_image_data:
            raw_image_data = raw_image_data.split(",")[1]

        image_bytes = base64.b64decode(raw_image_data)
        file_obj = io.BytesIO(image_bytes)

        # 1. Загружаем файл НАПРЯМУЮ в официальное хранилище Replicate
        uploaded_file = client.files.create(file_obj)

        # 2. Запускаем модель lucataco/modelscope-facefusion через официальную ссылку Replicate
        output = client.run(
            "lucataco/modelscope-facefusion:9a429854842207b4f3c163fac45732d841196929f214f440536c0a0cbe5c3459",
            input={
                "template_image": TARGET_SUIT_IMAGE,
                "user_image": uploaded_file.urls["get"],
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
