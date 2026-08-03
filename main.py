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


@app.get("/")
def home():
    return {"status": "ok", "message": "AI Photo & Video Generator API"}


@app.post("/api/generate")
async def generate_content(req: ContentRequest):
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise HTTPException(
                status_code=500, detail="REPLICATE_API_TOKEN not set"
            )

        client = replicate.Client(api_token=api_token, timeout=300.0)

        # 1. Если запросили ФОТО
        if req.mode == "photo":
            output = client.run(
                "black-forest-labs/flux-schnell",
                input={
                    "prompt": req.prompt,
                    "aspect_ratio": "16:9",
                    "output_format": "jpg",
                },
            )

        # 2. Если запросили ВИДЕО
        elif req.mode == "video":
            output = client.run(
                "minimax/video-01",
                input={
                    "prompt": req.prompt,
                    "prompt_optimizer": True,
                },
            )
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid mode. Use 'photo' or 'video'.",
            )

        # Парсим ответ
        url = None
        if hasattr(output, "url"):
            url = str(output.url)
        elif isinstance(output, list) and len(output) > 0:
            url = str(output[0])
        elif isinstance(output, str):
            url = output

        if url:
            return {
                "status": "success",
                "mode": req.mode,
                "output_url": url,
            }

        return {
            "status": "error",
            "error": f"Модель не вернула результат: {output}",
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}
