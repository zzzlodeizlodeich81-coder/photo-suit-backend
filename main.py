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


class ContentRequest(BaseModel):
    prompt: str
    mode: str = "photo"  # "photo" или "video"
    aspect_ratio: str = "16:9"


@app.get("/")
def home():
    return {"status": "ok", "message": "AI Content Studio API"}


# 1. Запуск генерации (Синхронно для Фото, Асинхронно для Видео)
@app.post("/api/generate")
@app.post("/api/process-photo")
async def generate_content(req: ContentRequest):
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise HTTPException(
                status_code=500, detail="REPLICATE_API_TOKEN not set"
            )

        client = replicate.Client(api_token=api_token, timeout=300.0)

        # РЕЖИМ 1: ФОТО (Flux Schnell - готовится за 3-5 секунд)
        if req.mode == "photo":
            output = client.run(
                "black-forest-labs/flux-schnell",
                input={
                    "prompt": req.prompt,
                    "aspect_ratio": req.aspect_ratio,
                    "output_format": "jpg",
                },
            )
            url = output[0] if isinstance(output, list) else str(output)
            return {"status": "success", "mode": "photo", "output_url": url}

        # РЕЖИМ 2: ВИДЕО (Запускаем через Predictions API, чтобы не ловить таймауты)
        elif req.mode == "video":
            # Используем быструю и стабильную модель Luma Ray
            prediction = client.predictions.create(
                version="luma/ray",
                input={
                    "prompt": req.prompt,
                    "aspect_ratio": req.aspect_ratio,
                },
            )
            # Возвращаем ID задачи фронтенду моментально!
            return {
                "status": "processing",
                "mode": "video",
                "prediction_id": prediction.id,
            }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# 2. Эндпоинт проверки статуса видео по ID
@app.get("/api/status/{prediction_id}")
async def check_status(prediction_id: str):
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        client = replicate.Client(api_token=api_token)

        prediction = client.predictions.get(prediction_id)

        if prediction.status == "succeeded":
            output = prediction.output
            url = output if isinstance(output, str) else output[0]
            return {"status": "success", "output_url": url}
        elif prediction.status == "failed":
            return {
                "status": "error",
                "error": prediction.error or "Генерация отменена или завершилась ошибкой",
            }
        else:
            return {"status": "processing", "progress": prediction.status}

    except Exception as e:
        return {"status": "error", "error": str(e)}
