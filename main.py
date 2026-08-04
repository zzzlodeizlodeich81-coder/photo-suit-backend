import os
import replicate
from urllib.parse import parse_qs
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

# Хранилище балансов пользователей в КАДРАХ: { "vk_user_id": balance_in_frames }
USER_BALANCES = {}


class ContentRequest(BaseModel):
    user_id: str
    prompt: Optional[str] = ""
    mode: str = "photo"  # "photo" или "video"
    aspect_ratio: str = "9:16"
    duration: int = 5  # 5, 10 или 15 сек
    image: Optional[str] = None


class AdRewardRequest(BaseModel):
    user_id: str


# Расчет стоимости в КАДРАХ (40 кадров = 2 голоса = ~10 руб)
def get_cost(mode: str, duration: int) -> int:
    if mode == "photo":
        return 40  # 40 кадров
    elif mode == "video":
        if duration <= 5:
            return 120   # 120 кадров (6 голосов)
        elif duration <= 10:
            return 220  # 220 кадров (11 голосов)
        else:
            return 320  # 320 кадров (16 голосов)
    return 40


@app.get("/")
def home():
    return {"status": "ok", "message": "AI Content Studio API is running"}


# -------------------------------------------------------------
# ЭНДПОИНТ: Проверка баланса пользователя (в кадрах)
# -------------------------------------------------------------
@app.get("/api/balance/{user_id}")
async def get_user_balance(user_id: str):
    balance = USER_BALANCES.get(str(user_id), 0)
    return {"status": "success", "balance": int(balance)}


# -------------------------------------------------------------
# ЭНДПОИНТ: Начисление за просмотр рекламы (+5 кадров)
# -------------------------------------------------------------
@app.post("/api/add-reward-ad")
async def add_reward_ad(req: AdRewardRequest):
    try:
        user_id = str(req.user_id)
        current_balance = USER_BALANCES.get(user_id, 0)
        
        # +5 кадров за ролик (8 роликов = 40 кадров = 1 фото)
        USER_BALANCES[user_id] = current_balance + 5
        
        return {
            "status": "success", 
            "new_balance": USER_BALANCES[user_id]
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# -------------------------------------------------------------
# ПЛАТЕЖНЫЙ WEBHOOK ВКОНТАКТЕ
# -------------------------------------------------------------
@app.post("/api/vk-payment")
async def vk_payment(request: Request):
    try:
        raw_body = await request.body()
        body_str = raw_body.decode("utf-8")
        
        parsed_data = parse_qs(body_str)
        data = {k: v[0] for k, v in parsed_data.items()}

        if not data:
            try:
                data = await request.json()
            except Exception:
                data = {}

        notification_type = data.get("notification_type")

        # 1. Запрос информации о товаре перед покупкой
        if notification_type in ["get_item", "get_item_test"]:
            item = str(data.get("item", ""))

            # Тарифные пакеты в Кадрах
            items_db = {
                "votes_2": {"title": "40 кадров (1 фото)", "price": 2, "frames": 40},
                "votes_6": {"title": "120 кадров (видео 5 сек)", "price": 6, "frames": 120},
                "votes_11": {"title": "220 кадров (видео 10 сек)", "price": 11, "frames": 220},
                "votes_16": {"title": "320 кадров (видео 15 сек)", "price": 16, "frames": 320},
                "votes_30": {"title": "600 кадров (Выгодно!)", "price": 30, "frames": 600},
            }
            
            item_info = items_db.get(item)
            if not item_info:
                try:
                    parsed_price = int(item.replace("votes_", ""))
                    item_info = {"title": f"{parsed_price * 20} кадров", "price": parsed_price}
                except Exception:
                    item_info = {"title": "Пополнение баланса (40 кадров)", "price": 2}

            return JSONResponse(
                content={
                    "response": {
                        "item_id": item,
                        "title": item_info["title"],
                        "price": int(item_info["price"]),
                    }
                },
                status_code=200,
            )

        # 2. Успешная оплата — зачисляем Кадры
        elif notification_type in ["order_status_change", "order_status_change_test"]:
            status = data.get("status")
            if status == "chargeable":
                order_id = data.get("order_id")
                user_id = str(data.get("user_id"))
                item = str(data.get("item", ""))

                # Конвертируем купленный пакет в Кадры
                frames_map = {
                    "votes_2": 40,
                    "votes_6": 120,
                    "votes_11": 220,
                    "votes_16": 320,
                    "votes_30": 600,
                }
                frames_to_add = frames_map.get(item, 40)

                USER_BALANCES[user_id] = USER_BALANCES.get(user_id, 0) + frames_to_add

                return JSONResponse(
                    content={
                        "response": {
                            "order_id": int(order_id),
                            "app_order_id": int(order_id),
                        }
                    },
                    status_code=200,
                )

        return JSONResponse(
            content={"error": {"error_code": 100, "error_msg": "Unknown notification"}},
            status_code=200,
        )

    except Exception as e:
        return JSONResponse(
            content={"error": {"error_code": 10, "error_msg": str(e)}},
            status_code=200,
        )


# -------------------------------------------------------------
# ЭНДПОИНТ ГЕНЕРАЦИИ (С СПИСАНИЕМ КАДРОВ)
# -------------------------------------------------------------
@app.post("/api/generate")
@app.post("/api/process-photo")
async def generate_content(req: ContentRequest):
    try:
        user_id = str(req.user_id)
        current_balance = USER_BALANCES.get(user_id, 0)
        required_cost = get_cost(req.mode, req.duration)

        if current_balance < required_cost:
            return {
                "status": "error",
                "error": f"Недостаточно кадров! Требуется {required_cost} кадров, а у вас на балансе {current_balance}.",
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

            # Списываем кадры
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
                "remaining_balance": USER_BALANCES[user_id],
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

            # Списываем кадры
            USER_BALANCES[user_id] -= required_cost

            return {
                "status": "processing",
                "mode": "video",
                "prediction_id": prediction.id,
                "remaining_balance": USER_BALANCES[user_id],
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
