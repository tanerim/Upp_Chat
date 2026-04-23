from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import ollama
import uuid
import json
import re
import time
import math
from collections import Counter
from typing import List
from datetime import datetime
from functools import lru_cache
from gensim.models import Word2Vec
from db import init_db, get_connection, DB_PATH


WORD_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)
MODEL_CACHE_TTL_SECONDS = 15
_model_cache = {"models": None, "expires_at": 0.0}

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

init_db()


def compute_word_frequency(text):
    words = tokenize_text(text)
    return dict(Counter(words))


def tokenize_text(text: str) -> List[str]:
    return WORD_PATTERN.findall(text.lower())


def compute_response_embedding(model: Word2Vec, tokens: List[str]) -> List[float]:
    if not tokens:
        return []

    valid_tokens = [token for token in tokens if token in model.wv]
    if not valid_tokens:
        return []

    embedding = model.wv[valid_tokens].mean(axis=0)
    return [float(value) for value in embedding]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_token_centroid(model: Word2Vec, tokens: List[str]) -> List[float]:
    valid_tokens = [token for token in tokens if token in model.wv]
    if not valid_tokens:
        return []

    embedding = model.wv[valid_tokens].mean(axis=0)
    return [float(value) for value in embedding]


def get_lexical_frontier(
    model: Word2Vec, left_centroid: List[float], right_centroid: List[float], limit: int = 12
):
    if not left_centroid or not right_centroid:
        return []

    frontier = []
    for token in model.wv.index_to_key:
        word_vec = [float(v) for v in model.wv[token]]
        left_score = cosine_similarity(word_vec, left_centroid)
        right_score = cosine_similarity(word_vec, right_centroid)
        delta = abs(left_score - right_score)
        frontier.append(
            {
                "word": token,
                "left_similarity": round(left_score, 6),
                "right_similarity": round(right_score, 6),
                "delta": round(delta, 6),
                "lean": "left" if left_score >= right_score else "right",
            }
        )

    frontier.sort(key=lambda item: item["delta"], reverse=True)
    return frontier[:limit]


def build_word2vec_model(tokenized_messages: List[List[str]], vector_size: int = 64) -> Word2Vec | None:
    usable_sentences = [tokens for tokens in tokenized_messages if tokens]
    if not usable_sentences:
        return None

    return Word2Vec(
        sentences=usable_sentences,
        vector_size=vector_size,
        window=5,
        min_count=1,
        workers=1,
        sg=1,
        epochs=20,
    )


@lru_cache(maxsize=1)
def get_fallback_models():
    return [
        {"name": "llama2", "label": "llama2 (3.9 GB)"},
        {"name": "mistral", "label": "mistral (7.2 GB)"},
        {"name": "phi3", "label": "phi3 (2.3 GB)"},
    ]


def _serialize_ollama_models(data):
    if hasattr(data, "models"):
        return [
            {
                "name": m.model,
                "label": f"{m.model} ({round(m.size / 1_000_000_000, 2)} GB)",
            }
            for m in data.models
        ]

    return [
        {
            "name": m.get("model", m.get("name", "unknown")),
            "label": (
                f"{m.get('model', m.get('name', 'unknown'))} "
                f"({round(m.get('size', 0) / 1_000_000_000, 2)} GB)"
            ),
        }
        for m in data.get("models", [])
    ]


def get_available_models():
    now = time.monotonic()
    if _model_cache["models"] and now < _model_cache["expires_at"]:
        return _model_cache["models"]

    data = ollama.list()
    models = _serialize_ollama_models(data)
    _model_cache["models"] = models
    _model_cache["expires_at"] = now + MODEL_CACHE_TTL_SECONDS
    return models


def normalize_chat_params(data):
    return {
        "temperature": float(data.get("temperature", data.get("left_temperature", 0.7))),
        "top_k": int(data.get("top_k", data.get("left_top_k", 40))),
        "top_p": float(data.get("top_p", data.get("left_top_p", 0.9))),
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        models = get_available_models()
    except Exception as e:
        print("⚠️ Error retrieving models:", e)
        models = get_fallback_models()

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "models": models, "css_version": int(time.time())},
    )


@app.post("/api/chat-stream")
async def chat_stream(request: Request):
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
    keep_alive = data.get("keep_alive", "5m")
    left_host = data.get("left_host", "http://127.0.0.1:11434").strip() or "http://127.0.0.1:11434"
    right_host = data.get("right_host", left_host).strip() or left_host
    prompt_left = data.get("prompt_left", "").strip()
    prompt_right = data.get("prompt_right", "").strip()

    async def event_generator():
        left_client = ollama.Client(host=left_host)
        right_client = ollama.Client(host=right_host)

        left_options = {"temperature": left_temperature, "top_k": left_top_k, "top_p": left_top_p}
        right_options = {"temperature": right_temperature, "top_k": right_top_k, "top_p": right_top_p}

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

        def stream_with_guard(client, model, messages, options):
            response_text = ""
            try:
                for token in model_stream(client, model, messages, options):
                    response_text += token
                    yield ("token", token)
            except Exception as exc:
                yield ("error", f"{model} failed: {exc}")
            else:
                yield ("done", response_text)

        yield send_chunk(
            "system",
            (
                f"Initializing Conversation: {left_model} ({left_host}) 🧠 "
                f"vs {right_model} ({right_host}) | turns={turns}, keep_alive={keep_alive}"
            ),
        )

        left_init = [
            {"role": "system", "content": prompt_left},
            {"role": "user", "content": "Begin the conversation based on your role."},
        ]

        left_reply = ""
        for event_type, value in stream_with_guard(left_client, left_model, left_init, left_options):
            if event_type == "token":
                yield send_chunk(left_model, value)
            elif event_type == "error":
                yield send_chunk("system", f"❌ {value}")
                yield send_chunk("system", "🏁 Conversation finished.")
                return
            elif event_type == "done":
                left_reply = value

        for _ in range(turns):
            right_messages = [
                {"role": "system", "content": prompt_right},
                {"role": "user", "content": left_reply},
            ]
            right_reply = ""
            for event_type, value in stream_with_guard(right_client, right_model, right_messages, right_options):
                if event_type == "token":
                    yield send_chunk(right_model, value)
                elif event_type == "error":
                    yield send_chunk("system", f"❌ {value}")
                    yield send_chunk("system", "🏁 Conversation finished.")
                    return
                elif event_type == "done":
                    right_reply = value

            left_messages = [
                {"role": "system", "content": prompt_left},
                {"role": "user", "content": right_reply},
            ]
            left_reply = ""
            for event_type, value in stream_with_guard(left_client, left_model, left_messages, left_options):
                if event_type == "token":
                    yield send_chunk(left_model, value)
                elif event_type == "error":
                    yield send_chunk("system", f"❌ {value}")
                    yield send_chunk("system", "🏁 Conversation finished.")
                    return
                elif event_type == "done":
                    left_reply = value

        yield send_chunk("system", "🏁 Conversation finished.")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/save")
async def save_conversation(request: Request):
    data = await request.json()
    cid = str(uuid.uuid4())
    conversation = data.get("conversation", [])
    params = normalize_chat_params(data)

    try:
        if not isinstance(conversation, list):
            return JSONResponse({"error": "conversation must be a list"}, status_code=400)

        created_at = datetime.now().isoformat()
        left_model = data.get("left_model", "unknown")
        right_model = data.get("right_model", "unknown")

        conn = get_connection()
        with conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO conversations (
                    id, left_model, right_model, temperature, top_k, top_p, conversation, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    cid,
                    left_model,
                    right_model,
                    params["temperature"],
                    params["top_k"],
                    params["top_p"],
                    json.dumps(conversation, ensure_ascii=False),
                    created_at,
                ),
            )

            tokenized_messages = [tokenize_text(str(message.get("content", ""))) for message in conversation]
            word2vec_model = build_word2vec_model(tokenized_messages)

            message_rows = []
            for idx, message in enumerate(conversation):
                role = str(message.get("role", "unknown"))
                content = str(message.get("content", ""))
                tokens = tokenized_messages[idx]
                word_frequency = compute_word_frequency(content)
                response_embedding = (
                    compute_response_embedding(word2vec_model, tokens) if word2vec_model else []
                )

                message_rows.append(
                    (
                        cid,
                        idx,
                        role,
                        content,
                        json.dumps(word_frequency, ensure_ascii=False),
                        json.dumps(response_embedding, ensure_ascii=False),
                        created_at,
                    )
                )

            c.executemany(
                """
                INSERT INTO conversation_messages (
                    conversation_id, message_index, role, content, word_frequency, embedding_vector, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                message_rows,
            )
        conn.close()

        print(f"💾 Saved conversation {cid} to {DB_PATH}")
        return JSONResponse({"status": "saved", "id": cid, "messages_saved": len(conversation)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/conversation-vectors")
async def conversation_vectors(request: Request):
    data = await request.json()
    conversation = data.get("conversation", [])
    left_model = str(data.get("left_model", "")).strip()
    right_model = str(data.get("right_model", "")).strip()

    if not isinstance(conversation, list) or not conversation:
        return JSONResponse({"error": "conversation must be a non-empty list"}, status_code=400)

    tokenized_messages = [tokenize_text(str(message.get("content", ""))) for message in conversation]
    word2vec_model = build_word2vec_model(tokenized_messages)
    if not word2vec_model:
        return JSONResponse({"error": "No usable text to build Word2Vec model"}, status_code=400)

    all_tokens = [token for tokens in tokenized_messages for token in tokens]
    left_tokens = []
    right_tokens = []

    for idx, message in enumerate(conversation):
        role = str(message.get("role", ""))
        if role == left_model:
            left_tokens.extend(tokenized_messages[idx])
        elif role == right_model:
            right_tokens.extend(tokenized_messages[idx])

    conversation_centroid = compute_token_centroid(word2vec_model, all_tokens)
    left_centroid = compute_token_centroid(word2vec_model, left_tokens)
    right_centroid = compute_token_centroid(word2vec_model, right_tokens)
    centroid_similarity = cosine_similarity(left_centroid, right_centroid)

    salient_words = []
    if conversation_centroid:
        for token in word2vec_model.wv.index_to_key:
            word_vec = [float(v) for v in word2vec_model.wv[token]]
            score = cosine_similarity(word_vec, conversation_centroid)
            salient_words.append({"word": token, "similarity": round(score, 6)})
        salient_words.sort(key=lambda item: item["similarity"], reverse=True)

    return JSONResponse(
        {
            "vector_size": word2vec_model.vector_size,
            "token_count": len(all_tokens),
            "vocabulary_size": len(word2vec_model.wv.index_to_key),
            "conversation_centroid": conversation_centroid,
            "left_centroid": left_centroid,
            "right_centroid": right_centroid,
            "left_right_similarity": round(centroid_similarity, 6),
            "salient_words": salient_words[:12],
            "lexical_frontier": get_lexical_frontier(word2vec_model, left_centroid, right_centroid),
        }
    )
