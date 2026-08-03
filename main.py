import base64
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


class PhotoRequest(BaseModel):
    image: str


@app.get("/")
def home():
    return {"status": "ok", "message": "Backend is running!"}


@app.post("/api/process-photo")
async def process_photo(req: PhotoRequest):
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise HTTPException(status_code=500, detail="Replicate token not set")

        # --- ВОТ ЭТУ ЧАСТЬ МЫ ОБНОВИЛИ ---
        # Мы используем новую, стабильную версию модели Flux.
        output = replicate.run(
            "lucataco/flux-dev-photorealistic:0e1ff44c77508f71253c0724a29a1b55977a4192b95b871216d0012e8b62842c",
            input={
                "prompt": "man in a formal, high-quality black business suit, white shirt, black tie, sharp focus, professional headshot style",
                "image": req.image,
                "prompt_strength": 0.6
            }
        )
        # --------------------------------

        if isinstance(output, list) and len(output) > 0:
            return {"status": "success", "output_url": str(output[0])}
        elif isinstance(output, str):
            return {"status": "success", "output_url": output}
        else:
            return {"status": "error", "error": "No image generated"}

    except Exception as e:
        return {"status": "error", "error": str(e)}
