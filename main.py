import base64
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import replicate

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

class PhotoRequest(BaseModel):
    image: str

@app.post("/api/process-photo")
async def process_photo(req: PhotoRequest):
    if not REPLICATE_API_TOKEN:
        raise HTTPException(status_code=500, detail="Replicate API Token not configured")

    try:
        output = replicate.run(
            "stability-ai/stable-diffusion-inpainting:c28b92a7ecd66ee4a1e94d1d45f63697f1708ed74615a91f37e41259e86e1088",
            input={
                "image": req.image,
                "prompt": "a professional business suit, formal shirt and tie, passport photo style, white background, high quality",
                "negative_prompt": "pajamas, casual clothes, distorted face, bad anatomy, dark background",
                "num_inference_steps": 25
            }
        )
        return {"status": "success", "output_url": output[0]}

    except Exception as e:
        return {"status": "error", "error": str(e)}
