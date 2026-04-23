# 🧠 Upp Chat
### Local Model Conversation

Upp Chat is a **FastAPI web app** that lets two **locally hosted LLMs (via [Ollama](https://ollama.ai))** talk to each other in real time.  
You can define the **role and goal** of one model, then watch it start and sustain a conversation with another model through a **live, color-coded chat interface**.
By default, total responses for each model is limited to 15. It could be set to more 

---

## 🚀 Features

- **Dual Model Interface** – Select any two local Ollama models for a back-and-forth “duel”.
- **System Prompt Mode** – Define a *persona or goal* for the left model; it uses this as a system prompt to start the chat.
- **Live Streaming (SSE)** – Messages appear token by token in real time using Server-Sent Events.
- **Visual Clarity**  
  - Left model = 🟡 light yellow messages  
  - Right model = 🟢 light green messages  
  - Each message block is labeled with the model name.
- **Custom Parameters** – Set temperature, top-k, and top-p values per model.
- **Independent Runtime Controls** – Set per-model temperature/top-k/top-p, Ollama host per side, turn count, and keep_alive.
- **Conversation Control** – Stop or save any dialogue into a local SQLite database.

---

## 🏗️ Project Structure

- Upp Chat/
- │
- ├── main.py # FastAPI backend
- ├── db.py # SQLite setup and helpers
- ├── templates/
- │ └── index.html # Frontend interface
- ├── static/
- │ └── style.css # Styling for UI
- └── requirements.txt # Dependencies


---

## ⚙️ Installation

1. **Clone or copy the repository**

   ```bash
   git clone https://github.com/tanerim/Upp_Chat
   cd Upp_Chat
   pip install -r requirements
   ```
2. **Make sure ollama is running**   
    ```bash
   ollama serve
   ollama list
   ```
3. ** RUn the app **
    ```bash
   uvicorn main:app --reload --port 8008
   ```

## ⚡ Performance Mode (recommended for multi-GPU)

If both models run on the same Ollama instance, it may unload/reload models between turns.
To reduce model swapping:

1. Run **two Ollama servers** (one per GPU) on different ports.
2. Assign one model to each host in the UI.
3. Increase `keep_alive` (example: `60m`) so models stay warm in VRAM.

Example (Linux):

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11434 ollama serve

# GPU 1 (second terminal)
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 ollama serve
```

Then set:
- Left host: `http://127.0.0.1:11434`
- Right host: `http://127.0.0.1:11435`


## How It Works

### Choose models
- Pick your left and right models from your installed Ollama list.

### Set model parameters
- You can set temperature, Top K, Top P, host, turn count, and keep_alive parameters.

### Define the left model’s role
- When prompted, describe the left model’s persona or mission, e.g.

```
You are an economist explaining the future of automation and universal basic income to another AI model who disagrees with you.
```
```
You are a cognitive scientist specializing in artificial intelligence alignment.
Your goal is to teach a curious student AI about cultural differences.
``` 
```
You are an alien anthropologist studying human humor.
Start a conversation by asking what makes people laugh.
``` 

## Watch the conversation
- The left model starts the conversation using your role definition,
and the right model responds — all streamed live in the browser.

## Stop or save
- You can stop the dialogue anytime and save the conversation (with parameters and model names) to a local SQLite database.

## 🛠️ TO_DO and  Ideas

- Add a “swap roles” button to reverse model order.
- Include a temperature sync toggle.
- Visualize conversation history in a separate page.
- Export saved conversations as Markdown or JSON.


## 🔎 Embedding Vectors: Görme ve Kullanma

Kaydedilen her mesaj artık iki analiz alanıyla gelir:
- `word_frequency`: kelime frekans sözlüğü
- `embedding_vector`: vektör alanı (JSON float listesi). `gensim` kuruluysa Word2Vec ile, kurulu değilse hash tabanlı fallback ile üretilir.

### 1) Mesajları ve embedding alanını çekme

```bash
curl "http://127.0.0.1:8008/api/conversations/<conversation_id>/messages?include_vectors=true"
```

### 2) Bir mesaj için benzer mesajları bulma (cosine similarity)

```bash
curl -X POST "http://127.0.0.1:8008/api/conversations/<conversation_id>/similarity" \
  -H "Content-Type: application/json" \
  -d '{"reference_index": 3, "top_k": 5}'
```

Bu endpoint, seçilen mesaj indeksine göre en benzer mesajları döndürür.
