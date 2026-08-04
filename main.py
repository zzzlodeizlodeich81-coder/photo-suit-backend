import os
import replicate
from fastapi import FastAPI, Request
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

# Хранилище балансов пользователей в ГОЛОСАХ VK: { "vk_user_id": balance_in_votes }
# В продакшене рекомендуется использовать SQLite / PostgreSQL.
USER_BALANCES = {}


class ContentRequest(BaseModel):
    user_id: str
    prompt: Optional[str] = ""
    mode: str = "photo"  # "photo" или "video"
    aspect_ratio: str = "9:16"
    duration: int = 5  # 5, 10 или 15 сек
    image: Optional[str] = None


# Расчет стоимости в ГОЛОСАХ исходя из цен Replicate ($0.06 за фото, $0.05/сек за видео)
def get_cost(mode: str, duration: int) -> int:
    if mode == "photo":
        return 2  # 2 голоса (~10 ₽ доход при себестоимости ~5.4 ₽)
    elif mode == "video":
        if duration <= 5:
            return 6   # 6 голосов за 5 сек (~30 ₽ доход при себестоимости ~22.5 ₽)
        elif duration <= 10:
            return 11  # 11 голосов за 10 сек (~55 ₽ доход при себестоимости ~45 ₽)
        else:
            return 16  # 16 голосов за 15 сек (~80 ₽ доход при себестоимости ~67.5 ₽)
    return 2


@app.get("/")
def home():
    return {"status": "ok", "message": "AI Content Studio API is running"}


# -------------------------------------------------------------
# ЭНДПОИНТ: Проверка баланса пользователя
# -------------------------------------------------------------
@app.get("/api/balance/{user_id}")
async def get_user_balance(user_id: str):
    balance = USER_BALANCES.get(str(user_id), 0)
    return {"status": "success", "balance": balance}


# -------------------------------------------------------------
# ПЛАТЕЖНЫЙ WEBHOOK ВКОНТАКТЕ
# -------------------------------------------------------------
@app.post("/api/vk-payment")
async def vk_payment(request: Request):
    try:
        try:
            form_data = await request.form()
            data = dict(form_data)
        except Exception:
            data = await request.json()

        notification_type = data.get("notification_type")

        # 1. VK запрашивает информацию о товаре перед покупкой (get_item / get_item_test)
        if notification_type in ["get_item", "get_item_test"]:
            item = data.get("item")

            # Кастомные наборы голосов (пакеты пополнения)
            items_db = {
                "votes_2": {"title": "2 голоса (1 фото)", "price": 2},
                "votes_6": {"title": "6 голосов (видео 5 сек)", "price": 6},
                "votes_11": {"title": "11 голосов (видео 10 сек)", "price": 11},
                "votes_16": {"title": "16 голосов (видео 15 сек)", "price": 16},
                "votes_30": {"title": "Пакет 30 голосов (со скидкой)", "price": 30},
            }
            
            # Фоллбэк: если товар не найден в словаре, цена берётся равной названию (например, votes_10 -> 10)
            item_info = items_db.get(item)
            if not item_info:
                try:
                    parsed_price = int(str(item).replace("votes_", ""))
                    item_info = {"title": f"{parsed_price} голосов", "price": parsed_price}
                except Exception:
                    item_info = {"title": "Пополнение баланса", "price": 2}

            return JSONResponse(content={
                "response": {
                    "item_id": str(item),
                    "title": str(item_info["title"]),
                    "price": int(item_info["price"])
                }
            })

        # 2. Успешная оплата — зачисляем купленные голоса
        elif notification_type in ["order_status_change", "order_status_change_test"]:
            status = data.get("status")
            if status == "chargeable":
                order_id = data.get("order_id")
                user_id = str(data.get("user_id"))
                item = data.get("item")

                votes_to_add = 2
                try:
                    votes_to_add = int(str(item).replace("votes_", ""))
                except Exception:
                    votes_to_add = 2

                USER_BALANCES[user_id] = USER_BALANCES.get(user_id, 0) + votes_to_add

                return JSONResponse(content={
                    "response": {
                        "order_id": int(order_id),
                        "app_order_id": int(order_id)
                    }
                })

        return JSONResponse(content={"error": {"error_code": 100, "error_msg": "Unknown notification"}})

    except Exception as e:
        return JSONResponse(content={"error": {"error_code": 10, "error_msg": str(e)}})


# -------------------------------------------------------------
# ЭНДПОИНТ ГЕНЕРАЦИИ (С ПРОВЕРКОЙ И СПИСАНИЕМ БАЛАНСА)
# -------------------------------------------------------------
@app.post("/api/generate")
@app.post("/api/process-photo")
async def generate_content(req: ContentRequest):
    try:
        user_id = str(req.user_id)
        current_balance = USER_BALANCES.get(user_id, 0)
        required_cost = get_cost(req.mode, req.duration)

        # БЛОКИРОВКА ГЕНЕРАЦИИ ПРИ НЕДОСТАТКЕ СРЕДСТВ
        if current_balance < required_cost:
            return {
                "status": "error",
                "error": f"Недостаточно голосов! Требуется {required_cost} голосов, а у вас на балансе {current_balance}."
            }

        api_token = os.environ.get("REPLICATE_API_TOKEN")
        if not api_token:
            return {"status": "error", "error": "REPLICATE_API_TOKEN environment variable is missing"}

        client = replicate.Client(api_token=api_token)

        # 1. РЕЖИМ ФОТО: xAI Grok Imagine Image
        if req.mode == "photo":
            input_params = {
                "prompt": req.prompt or "high quality image",
                "aspect_ratio": req.aspect_ratio,
            }
            if req.image:
                input_params["image"] = req.image

            output = client.run("xai/grok-imagine-image", input=input_params)

            # Списываем голоса после запуска
            USER_BALANCES[user_id] -= required_cost

            if isinstance(output, list) and len(output) > 0:
                item = output[0]
                url_str = getattr(item, "url", str(item))
            else:
                url_str = getattr(output, "url", str(output))

            return {
                "status": "success",
                "mode": "photo",
                "output_url": str(url_str),
                "remaining_balance": USER_BALANCES[user_id]
            }

        # 2. РЕЖИМ ВИДЕО: xAI Grok Imagine Video
        elif req.mode == "video":
            video_duration = req.duration if 1 <= req.duration <= 15 else 5

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

            # Списываем голоса после запуска
            USER_BALANCES[user_id] -= required_cost

            return {
                "status": "processing",
                "mode": "video",
                "prediction_id": prediction.id,
                "remaining_balance": USER_BALANCES[user_id]
            }

    except Exception as e:
        return {"status": "error", "error": str(e)}


# -------------------------------------------------------------
# ЭНДПОИНТ ПРОВЕРКИ СТАТУСА ВИДЕО
# -------------------------------------------------------------
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
