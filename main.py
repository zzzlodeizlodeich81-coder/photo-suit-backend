import base64
import json
import os
import urllib.parse
import urllib.request
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


# Шаблон костюма
TARGET_SUIT_IMAGE = "https://images.unsplash.com/photo-1507679799987-c73779587ccf?q=80&w=1000&auto=format&fit=crop"


def upload_base64_to_tmp(base64_str: str) -> str:
    """Загружает base64 фото во временное хранилище через встроенные библиотеки Python."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]

    image_bytes = base64.b64decode(base64_str)

    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = []

    body.append(f"--{boundary}".encode())
    body.append(
        b'Content-Disposition: form-data; name="reqtype"\r\n\r\nfileupload'
    )

    body.append(f"--{boundary}".encode())
    body.append(
        b'Content-Disposition: form-data; name="fileToUpload";'
        b' filename="face.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'
    )
    body.append(image_bytes)
    body.append(b"\r\n")
    body.append(f"--{boundary}--\r\n".encode())

    payload = b"\r\n".join(
        [
            body[0],
            body[1],
            body[2],
            body[3] + body[4] + body[5],
            body[6],
        ]
    )

    req = urllib.request.Request(
        "https://catbox.moe/user/api.php",
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )

    with urllib.request.urlopen(req, timeout=15) as response:
        res_text = response.read().decode("utf-8").strip()
        if res_text.startswith("http"):
            return res_text
        raise Exception("Ошибка при загрузке фото во временное хранилище")


@app.get("/")
def home():
    return {"status": "ok", "message": "Face Swap Backend is Ready!"}


@app.post("/api/process-photo")
async def process_photo(req: PhotoRequest):
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise HTTPException(
                status_code=500, detail="REPLICATE_API_TOKEN not set"
            )

        client = replicate.Client(api_token=api_token, timeout=120.0)

        # 1. Получаем прямую URL-ссылку на загруженное фото
        user_image_url = upload_base64_to_tmp(req.image)

        # 2. Передаем ссылки в Replicate
        output = client.run(
            "codeplugtech/face-swap:278a81e7ebb22db98bcba54de985d22cc1abeead2754eb1f2af717247be69b34",
            input={
                "input_image": TARGET_SUIT_IMAGE,
                "swap_image": user_image_url,
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
