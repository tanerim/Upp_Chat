from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
import asyncio
import ollama
import sqlite3
import uuid
import json
import time
from datetime import datetime

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

DB_PATH = "conversations.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        model_left TEXT,
        model_right TEXT,
        temperature REAL,
        top_k INTEGER,
        top_p REAL,
        conversation TEXT,
        timestamp TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        data = ollama.list()

        # data.models is a list of Model objects
        if hasattr(data, "models"):
            models = [m.model for m in data.models]
        else:
            # Fallback for older API versions
            models = [m.get("model") or m.get("name") for m in data.get("models", [])]

    except Exception as e:
        print("⚠️ Error retrieving models:", e)
        models = ["llama2", "mistral", "phi3"]  # safe defaults

    return templates.TemplateResponse("index.html", {"request": request, "models": models})



@app.post("/api/start_conversation")
async def start_conversation(
    left_model: str = Form(...),
    right_model: str = Form(...),
    temperature: float = Form(0.7),
    top_k: int = Form(40),
    top_p: float = Form(0.9),
):
    return JSONResponse({"status": "ready", "left_model": left_model, "right_model": right_model})

from fastapi import Request
from fastapi.responses import JSONResponse
import ollama

@app.post("/api/chat-stream")
async def chat_stream(request: Request):
    """
    Stream a goal-driven conversation between two Ollama models.
    User provides a system prompt defining the left model's role.
    Left model begins the conversation with its first message.
    """
    data = await request.json()
    left_model = data["left_model"]
    right_model = data["right_model"]
    temperature = float(data.get("temperature", 0.7))
    top_k = int(data.get("top_k", 40))
    top_p = float(data.get("top_p", 0.9))
    system_prompt = data["prompt"].strip()

    async def event_generator():
        def send_chunk(role, token):
            return f"data: {json.dumps({'role': role, 'token': token})}\n\n"

        def model_stream(model, messages):
            for chunk in ollama.chat(
                model=model,
                messages=messages,
                options={"temperature": temperature, "top_k": top_k, "top_p": top_p},
                stream=True,
            ):
                if "message" in chunk:
                    yield chunk["message"]["content"]

        yield send_chunk("system", f"Initializing duel: {left_model} 🧠 vs {right_model}")

        # Left model receives system prompt first
        left_init_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Begin the conversation based on your role."}
        ]

        left_reply = ""
        for token in model_stream(left_model, left_init_messages):
            left_reply += token
            yield send_chunk(left_model, token)

        # Now alternate dialogue
        for _ in range(15):  # 15 full exchanges as default limit
            right_messages = [{"role": "user", "content": left_reply}]
            right_reply = ""
            for token in model_stream(right_model, right_messages):
                right_reply += token
                yield send_chunk(right_model, token)

            left_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": right_reply},
            ]
            left_reply = ""
            for token in model_stream(left_model, left_messages):
                left_reply += token
                yield send_chunk(left_model, token)

        yield send_chunk("system", "🏁 Duel finished.")

    return StreamingResponse(event_generator(), media_type="text/event-stream")




@app.post("/api/save")
async def save_conversation(request: Request):
    data = await request.json()
    cid = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO conversations VALUES (?,?,?,?,?,?,?,?)",
        (
            cid,
            data["left_model"],
            data["right_model"],
            data["temperature"],
            data["top_k"],
            data["top_p"],
            json.dumps(data["conversation"], ensure_ascii=False),
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()
    return JSONResponse({"status": "saved", "id": cid})
