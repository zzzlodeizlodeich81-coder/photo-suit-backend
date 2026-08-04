import os
import replicate
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

# 1. Железобетонный CORS для работы с GitHub Pages
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
    duration: int = 5


# Обработчик корневого урла
@app.get("/")
def home():
    return {"status": "ok", "message": "AI Content Studio API is running"}


# Глобальный обработчик ошибок, чтобы CORS не отваливался при ошибках 500/404
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=200,
        content={"status": "error", "error": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"},
    )


# Основной эндпоинт генерации
@app.post("/api/generate")
@app.post("/api/process-photo")
async def generate_content(req: ContentRequest):
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            return {
                "status": "error",
                "error": "REPLICATE_API_TOKEN environment variable is missing on Render",
            }

        client = replicate.Client(api_token=api_token)

        # РЕЖИМ 1: ФОТО (Flux Schnell)
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

        # РЕЖИМ 2: ВИДЕО (Runway Gen-4 Turbo)
        elif req.mode == "video":
            valid_ratios = ["16:9", "9:16", "1:1", "3:4", "4:3", "21:9"]
            target_ratio = (
                req.aspect_ratio if req.aspect_ratio in valid_ratios else "16:9"
            )
            video_duration = 10 if req.duration == 10 else 5

            prediction = client.predictions.create(
                model="runwayml/gen4-turbo",
                input={
                    "prompt": req.prompt,
                    "aspect_ratio": target_ratio,
                    "duration": video_duration,
                },
            )

            return {
                "status": "processing",
                "mode": "video",
                "prediction_id": prediction.id,
            }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# 2. Опрос статуса видео (Polling)
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
                "error": prediction.error or "Ошибка генерации видео",
            }
        else:
            return {"status": "processing", "progress": prediction.status}

    except Exception as e:
        return {"status": "error", "error": str(e)}
