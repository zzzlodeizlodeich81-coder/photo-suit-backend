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


# 1. Запуск генерации
@app.post("/api/generate")
@app.post("/api/process-photo")
async def generate_content(req: ContentRequest):
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise HTTPException(
                status_code=500, detail="REPLICATE_API_TOKEN not set"
            )

        client = replicate.Client(api_token=api_token)

        # РЕЖИМ 1: ФОТО (Flux Schnell — поддерживает 16:9, 9:16, 1:1, 3:4, 4:3 и т.д.)
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

        # РЕЖИМ 2: ВИДЕО (Запуск модели Luma Ray)
        elif req.mode == "video":
            # Настройка фрейминга для видео
            video_prompt = req.prompt
            if req.aspect_ratio == "9:16":
                video_prompt += (
                    ", vertical 9:16 portrait orientation, vertical composition"
                )
            elif req.aspect_ratio == "16:9":
                video_prompt += ", horizontal 16:9 widescreen orientation"
            elif req.aspect_ratio == "1:1":
                video_prompt += ", square 1:1 format"

            # Запускаем через predictions.create с использованием правильной модели
            prediction = client.predictions.create(
                model="luma/ray",
                input={
                    "prompt": video_prompt,
                    "aspect_ratio": req.aspect_ratio,
                },
            )

            return {
                "status": "processing",
                "mode": "video",
                "prediction_id": prediction.id,
            }

    except Exception as e:
        # Если модель luma/ray требует прямую ссылку или версию, делаем фоллбек
        try:
            if req.mode == "video":
                # Резервный запуск luma/ray через стандартную модель
                model_obj = client.models.get("luma/ray")
                version = model_obj.latest_version
                prediction = client.predictions.create(
                    version=version.id,
                    input={
                        "prompt": req.prompt,
                        "aspect_ratio": req.aspect_ratio,
                    },
                )
                return {
                    "status": "processing",
                    "mode": "video",
                    "prediction_id": prediction.id,
                }
        except Exception as fallback_err:
            return {"status": "error", "error": str(fallback_err)}

        return {"status": "error", "error": str(e)}


# 2. Эндпоинт проверки статуса видео
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
