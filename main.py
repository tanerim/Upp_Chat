from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse
import ollama
import uuid
import json
import re
from collections import Counter
from datetime import datetime
from db import init_db, get_connection, DB_PATH


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

init_db()


def compute_word_frequency(text):
    """
    Build a word frequency map for a single response.
    Keeps unicode word characters and lowercases for normalized counting.
    """
    words = re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)
    return dict(Counter(words))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        data = ollama.list()
        if hasattr(data, "models"):
            models = [
                {
                    "name": m.model,
                    "label": f"{m.model} ({round(m.size / 1_000_000_000, 2)} GB)"
                }
                for m in data.models
            ]
        else:
            models = [
                {
                    "name": m.get("model", m.get("name", "unknown")),
                    "label": f"{m.get('model', m.get('name', 'unknown'))} "
                             f"({round(m.get('size', 0) / 1_000_000_000, 2)} GB)"
                }
                for m in data.get("models", [])
            ]
    except Exception as e:
        print("⚠️ Error retrieving models:", e)
        models = [
            {"name": "llama2", "label": "llama2 (3.9 GB)"},
            {"name": "mistral", "label": "mistral (7.2 GB)"},
            {"name": "phi3", "label": "phi3 (2.3 GB)"}
        ]

    return templates.TemplateResponse("index.html", {"request": request, "models": models})





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
    left_temperature = float(data.get("left_temperature", 0.7))
    left_top_k = int(data.get("left_top_k", 40))
    left_top_p = float(data.get("left_top_p", 0.9))
    right_temperature = float(data.get("right_temperature", 0.7))
    right_top_k = int(data.get("right_top_k", 40))
    right_top_p = float(data.get("right_top_p", 0.9))
    turns = max(1, min(200, int(data.get("turns", 15))))
    keep_alive = data.get("keep_alive", "20m")
    left_host = data.get("left_host", "http://127.0.0.1:11434").strip() or "http://127.0.0.1:11434"
    right_host = data.get("right_host", left_host).strip() or left_host
    prompt_left = data.get("prompt_left", "").strip()
    prompt_right = data.get("prompt_right", "").strip()

    async def event_generator():
        left_client = ollama.Client(host=left_host)
        right_client = ollama.Client(host=right_host)

        def send_chunk(role, token):
            return f"data: {json.dumps({'role': role, 'token': token})}\n\n"

        def model_stream(client, model, messages, options):
            for chunk in client.chat(
                    model=model,
                    messages=messages,
                    options=options,
                    keep_alive=keep_alive,
                    stream=True,
            ):
                if "message" in chunk:
                    yield chunk["message"]["content"]

        yield send_chunk(
            "system",
            (
                f"Initializing Conversation: {left_model} ({left_host}) 🧠 "
                f"vs {right_model} ({right_host}) | turns={turns}, keep_alive={keep_alive}"
            ),
        )

        # Left model starts with its system prompt
        left_init = [
            {"role": "system", "content": prompt_left},
            {"role": "user", "content": "Begin the conversation based on your role."}
        ]

        left_reply = ""
        for token in model_stream(
                left_client,
                left_model,
                left_init,
                {"temperature": left_temperature, "top_k": left_top_k, "top_p": left_top_p},
        ):
            left_reply += token
            yield send_chunk(left_model, token)

        # Alternate dialogue between models
        for _ in range(turns):
            right_messages = [
                {"role": "system", "content": prompt_right},
                {"role": "user", "content": left_reply},
            ]
            right_reply = ""
            for token in model_stream(
                    right_client,
                    right_model,
                    right_messages,
                    {"temperature": right_temperature, "top_k": right_top_k, "top_p": right_top_p},
            ):
                right_reply += token
                yield send_chunk(right_model, token)

            left_messages = [
                {"role": "system", "content": prompt_left},
                {"role": "user", "content": right_reply},
            ]
            left_reply = ""
            for token in model_stream(
                    left_client,
                    left_model,
                    left_messages,
                    {"temperature": left_temperature, "top_k": left_top_k, "top_p": left_top_p},
            ):
                left_reply += token
                yield send_chunk(left_model, token)

        yield send_chunk("system", "🏁 Conversation finished.")

    return StreamingResponse(event_generator(), media_type="text/event-stream")




@app.post("/api/save")
async def save_conversation(request: Request):
    data = await request.json()
    cid = str(uuid.uuid4())
    conversation = data.get("conversation", [])

    try:
        conn = get_connection()
        c = conn.cursor()
        c.execute("""
            INSERT INTO conversations (
                id, left_model, right_model, temperature, top_k, top_p, conversation, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cid,
            data["left_model"],
            data["right_model"],
            data["temperature"],
            data["top_k"],
            data["top_p"],
            json.dumps(conversation, ensure_ascii=False),
            datetime.now().isoformat(),
        ))

        for idx, message in enumerate(conversation):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            word_frequency = compute_word_frequency(content)
            c.execute("""
                INSERT INTO conversation_messages (
                    conversation_id, message_index, role, content, word_frequency, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                cid,
                idx,
                role,
                content,
                json.dumps(word_frequency, ensure_ascii=False),
                datetime.now().isoformat(),
            ))

        conn.commit()
        conn.close()

        print(f"💾 Saved conversation {cid} to {DB_PATH}")
        return JSONResponse({
            "status": "saved",
            "id": cid,
            "messages_saved": len(conversation)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
