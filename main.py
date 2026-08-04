import os
import replicate
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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
    duration: int = 5


@app.get("/")
def home():
    return {"status": "ok", "message": "AI Content Studio API is running"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=200,
        content={"status": "error", "error": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"},
    )


@app.post("/api/generate")
@app.post("/api/process-photo")
async def generate_content(req: ContentRequest):
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            return {
                "status": "error",
                "error": "REPLICATE_API_TOKEN environment variable is missing",
            }

        client = replicate.Client(api_token=api_token)

        # -------------------------------------------------------------
        # 1. РЕЖИМ ФОТО: Flux Schnell
        # -------------------------------------------------------------
        if req.mode == "photo":
            output = client.run(
                "black-forest-labs/flux-schnell",
                input={
                    "prompt": req.prompt,
                    "aspect_ratio": req.aspect_ratio,
                    "output_format": "jpg",
                },
            )

            # Вытаскиваем URL из результата
            if isinstance(output, list) and len(output) > 0:
                item = output[0]
                url_str = getattr(item, "url", str(item))
            else:
                url_str = getattr(output, "url", str(output))

            return {"status": "success", "mode": "photo", "output_url": str(url_str)}

        # -------------------------------------------------------------
        # 2. РЕЖИМ ВИДЕО: xAI Grok Imagine Video
        # -------------------------------------------------------------
        elif req.mode == "video":
            video_duration = req.duration if 1 <= req.duration <= 15 else 5

            # Создаем асинхронный prediction для фоновой генерации
            prediction = client.predictions.create(
                model="xai/grok-imagine-video",
                input={
                    "prompt": req.prompt,
                    "aspect_ratio": req.aspect_ratio,
                    "duration": int(video_duration),
                },
            )

            return {
                "status": "processing",
                "mode": "video",
                "prediction_id": prediction.id,
            }

    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.get("/api/status/{prediction_id}")
async def check_status(prediction_id: str):
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        client = replicate.Client(api_token=api_token)

        prediction = client.predictions.get(prediction_id)

        if prediction.status == "succeeded":
            output = prediction.output

            if isinstance(output, list) and len(output) > 0:
                res_item = output[0]
            else:
                res_item = output

            # Читаем .url согласно документации Replicate
            final_url = getattr(res_item, "url", str(res_item))

            return {"status": "success", "output_url": str(final_url)}

        elif prediction.status == "failed":
            return {
                "status": "error",
                "error": prediction.error or "Ошибка генерации видео",
            }
        else:
            return {"status": "processing", "progress": prediction.status}

    except Exception as e:
        return {"status": "error", "error": str(e)}
