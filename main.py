import base64
import os
import requests
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


def upload_base64_to_tmp(base64_str: str) -> str:
    """Загружает base64 фото во временное хранилище и возвращает прямую ссылку URL."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    image_bytes = base64.b64decode(base64_str)

    # Загружаем на временный сервисный хостинг catbox
    response = requests.post(
        "https://catbox.moe/user/api.php",
        data={"reqtype": "fileupload"},
        files={"fileToUpload": ("face.jpg", image_bytes, "image/jpeg")},
        timeout=15,
    )

    if response.status_code == 200 and response.text.startswith("http"):
        return response.text.strip()
    else:
        raise Exception("Не удалось загрузить временное фото лица")


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

        # 1. Получаем прямую URL-ссылку на фото лица
        user_image_url = upload_base64_to_tmp(req.image)

        # 2. Вызываем проверенную модель codeplugtech/face-swap со 100% точными именами ключей
        output = client.run(
            "codeplugtech/face-swap:278a81e7ebb22db98bcba54de985d22cc1abeead2754eb1f2af717247be69b34",
            input={
                "input_image": TARGET_SUIT_IMAGE,  # Фоновый костюм
                "swap_image": user_image_url,  # Лицо
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
