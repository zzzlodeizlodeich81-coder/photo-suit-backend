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


# Фото отличного делового костюма на прозрачном/нейтральном фоне
TARGET_SUIT_IMAGE = "https://images.unsplash.com/photo-1507679799987-c73779587ccf?q=80&w=1000&auto=format&fit=crop"


@app.get("/")
def home():
    return {"status": "ok", "message": "Face Swap Backend is Ready!"}


@app.post("/api/process-photo")
async def process_photo(req: PhotoRequest):
    try:
        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            raise HTTPException(status_code=500, detail="Replicate token not set")

        client = replicate.Client(api_token=api_token, timeout=120.0)

        # Вызываем строго проверенную модель со страницы Replicate
        output = client.run(
            "codeplugtech/face-swap:278a81e7ebb22db98bcba54de985d22cc1abeead2754eb1f2af717247be69b34",
            input={
                "input_image": TARGET_SUIT_IMAGE,  # Костюм
                "swap_image": req.image,            # Твоё лицо
            },
        )

        if output:
            # Replicate возвращает объект FileOutput, берем его URL
            output_url = str(output.url) if hasattr(output, "url") else str(output)
            return {"status": "success", "output_url": output_url}

        return {"status": "error", "error": "Не удалось сгенерировать фото"}

    except Exception as e:
        return {"status": "error", "error": str(e)}
