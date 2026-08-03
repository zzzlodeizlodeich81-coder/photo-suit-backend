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
    aspect_ratio: str = "16:9"  # "16:9", "9:16", "1:1", "4:3", "3:4"


@app.get("/")
def home():
    return {"status": "ok", "message": "AI Content Studio API with Aspect Ratio"}


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

        # 1. Генерация ФОТО (FLUX Schnell поддерживает aspect_ratio)
        if req.mode == "photo":
            output = client.run(
                "black-forest-labs/flux-schnell",
                input={
                    "prompt": req.prompt,
                    "aspect_ratio": req.aspect_ratio,
                    "output_format": "jpg",
                },
            )

        # 2. Генерация ВИДЕО (MiniMax Video-01)
        elif req.mode == "video":
            # У MiniMax формат передается в промпте/параметрах (16:9 по умолчанию, или настраиваем aspect_ratio)
            input_params = {
                "prompt": req.prompt,
                "prompt_optimizer": True,
            }

            output = client.run("minimax/video-01", input=input_params)
        else:
            raise HTTPException(
                status_code=400, detail="Invalid mode. Use 'photo' or 'video'."
            )

        url = None
        if hasattr(output, "url"):
            url = str(output.url)
        elif isinstance(output, list) and len(output) > 0:
            url = str(output[0])
        elif isinstance(output, str):
            url = output

        if url:
            return {"status": "success", "mode": req.mode, "output_url": url}

        return {
            "status": "error",
            "error": f"Модель не вернула URL: {output}",
        }

    except Exception as e:
        return {"status": "error", "error": str(e)}
