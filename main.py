import os
import replicate
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Настройки CORS для GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PhotoRequest(BaseModel):
    image: str  # Входящее фото пользователя (Base64)


# Эталонное фото мужчины в отличном деловом костюме.
# Твоё лицо будет наложено на это тело.
TARGET_SUIT_IMAGE = "https://replicate.delivery/pbxt/JRjZ68O056N2a26sP6K2a26sP6K2a26sP6K2a26sP6K2a26sP/output.png"


@app.get("/")
def home():
    return {"status": "ok", "message": "Face Swap Backend for Passport Photo"}


@app.post("/api/process-photo")
async def process_photo(req: PhotoRequest):
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise HTTPException(status_code=500, detail="Replicate token not set")

        # Настраиваем клиент с токеном
        client = replicate.Client(api_token=api_token, timeout=120.0)

        # Мы используем модель 'Face Swapper' (она супер-точная для сохранения лица)
        output = client.run(
            "pnm-company/face-swap:95dfc07218671607593d7c48529323c96a7b3d30421e4284566c7f8976b32525",
            input={
                "target_image": TARGET_SUIT_IMAGE,  # Фото костюма
                "swap_image": req.image,            # Твоё лицо
                "smooth_face": True,               # Сгладить переходы
                "align_faces": True,                # Выровнять лица
            }
        )

        # Результат этой модели — прямая ссылка на итоговое фото
        if output:
            return {"status": "success", "output_url": str(output)}
        else:
            return {"status": "error", "error": "No image generated"}

    except Exception as e:
        return {"status": "error", "error": str(e)}
