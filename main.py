import os
import replicate
from fastapi import FastAPI, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ContentRequest(BaseModel):
    prompt: Optional[str] = ""
    mode: str = "photo"  # "photo" или "video"
    aspect_ratio: str = "9:16"
    duration: int = 15
    image: Optional[str] = None  # Base64 или URL картинки для Image-to-Image / Image-to-Video


@app.get("/")
def home():
    return {"status": "ok", "message": "AI Content Studio API (Grok Suite) is running"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=200,
        content={"status": "error", "error": str(exc)},
        headers={"Access-Control-Allow-Origin": "*"},
    )


# -------------------------------------------------------------
# ОБРАБОТЧИК ПЛАТЕЖЕЙ ВКОНТАКТЕ (WEBHOOK)
# -------------------------------------------------------------
@app.post("/api/vk-payment")
async def vk_payment(request: Request):
    try:
        # ВКонтакте отправляет данные в формате x-www-form-urlencoded
        content_type = request.headers.get("content-type", "")
        if "application/x-www-form-urlencoded" in content_type:
            data = await request.form()
        else:
            data = await request.json()

        notification_type = data.get("notification_type")

        # 1. Запрос информации о товаре (при инициализации оплаты или тесте VK)
        if notification_type in ["get_item", "get_item_test"]:
            item = data.get("item")
            
            # Логика цен и названий (настраивается под ваши товары)
            items_db = {
                "photo_1": {"title": "1 генерация фото", "price": 1},
                "video_1": {"title": "1 генерация видео", "price": 5},
                "pack_10_5": {"title": "Пакет: 10 фото + 5 видео", "price": 20}
            }
            
            item_info = items_db.get(item, {"title": "Генерация контента", "price": 1})

            return {
                "response": {
                    "item_id": item,
                    "title": item_info["title"],
                    "price": item_info["price"]
                }
            }

        # 2. Изменение статуса заказа (успешная оплата)
        elif notification_type in ["order_status_change", "order_status_change_test"]:
            status = data.get("status")
            if status == "chargeable":
                order_id = data.get("order_id")
                user_id = data.get("user_id")
                item = data.get("item")

                # ЗДЕСЬ ДОБАВЛЯЕТСЯ ЛОГИКА НАЧИСЛЕНИЯ БАЛАНСА В БАЗУ ДАННЫХ
                # print(f"Пользователь {user_id} успешно купил {item}")

                return {
                    "response": {
                        "order_id": int(order_id),
                        "app_order_id": int(order_id)
                    }
                }

        return {"error": {"error_code": 100, "error_msg": "Неизвестный тип уведомления"}}

    except Exception as e:
        return {"error": {"error_code": 10, "error_msg": str(e)}}


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
        # 1. РЕЖИМ ФОТО: xAI Grok Imagine Image
        # -------------------------------------------------------------
        if req.mode == "photo":
            input_params = {
                "prompt": req.prompt or "high quality image",
                "aspect_ratio": req.aspect_ratio,
            }
            if req.image:
                input_params["image"] = req.image

            output = client.run(
                "xai/grok-imagine-image",
                input=input_params,
            )

            if isinstance(output, list) and len(output) > 0:
                item = output[0]
                url_str = getattr(item, "url", str(item))
            else:
                url_str = getattr(output, "url", str(output))

            return {"status": "success", "mode": "photo", "output_url": str(url_str)}

        # -------------------------------------------------------------
        # 2. РЕЖИМ ВИДЕО: xAI Grok Imagine Video (поддержка до 15 сек)
        # -------------------------------------------------------------
        elif req.mode == "video":
            video_duration = req.duration if 1 <= req.duration <= 15 else 15

            input_params = {
                "prompt": req.prompt or "cinematic motion",
                "aspect_ratio": req.aspect_ratio,
                "duration": int(video_duration),
            }
            if req.image:
                input_params["image"] = req.image

            prediction = client.predictions.create(
                model="xai/grok-imagine-video",
                input=input_params,
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
