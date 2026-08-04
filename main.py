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

        # РЕЖИМ 1: ФОТО (Flux Schnell - оставили как есть, так как качество идеальное!)
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

        # РЕЖИМ 2: ВИДЕО (Luma Ray с выбором формата кадра)
        elif req.mode == "video":
            # Формируем промпт с явным указанием ориентации кадра
            video_prompt = req.prompt
            if req.aspect_ratio == "9:16":
                video_prompt += ", vertical video shot 9:16 framing, portrait mode"
            elif req.aspect_ratio == "16:9":
                video_prompt += (
                    ", horizontal widescreen 16:9 video shot, cinematic"
                )
            elif req.aspect_ratio == "1:1":
                video_prompt += ", square 1:1 video shot"

            # Запускаем через predictions.create для luma/ray
            prediction = client.predictions.create(
                model="luma/ray",
                input={
                    "prompt": video_prompt,
                    "aspect_ratio": req.aspect_ratio,
                },
            )

            # Возвращаем ID задачи фронтенду моментально
            return {
                "status": "processing",
                "mode": "video",
                "prediction_id": prediction.id,
            }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# 2. Проверка статуса генерации видео по ID
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
                "error": prediction.error
                or "Генерация отменена или завершилась ошибкой",
            }
        else:
            return {"status": "processing", "progress": prediction.status}

    except Exception as e:
        return {"status": "error", "error": str(e)}
