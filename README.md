# 🧠 LLM Duel – Local Model Conversation Arena

LLM Duel is a **FastAPI web app** that lets two **locally hosted LLMs (via [Ollama](https://ollama.ai))** talk to each other in real time.  
You can define the **role and goal** of one model, then watch it start and sustain a conversation with another model through a **live, color-coded chat interface**.

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
- **Conversation Control** – Stop or save any dialogue into a local SQLite database.

---

## 🏗️ Project Structure

- ollama_duel/
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
   git clone https://github.com/yourusername/ollama_duel.git
   cd ollama_duel
   pip install fastapi uvicorn jinja2 ollama
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


## How It Works

### Choose models
- Pick your left and right models from your installed Ollama list.

### Set model parameters
- You can set temperature, Top K and Top P parameters for each model.

### Define the left model’s role
- When prompted, describe the left model’s persona or mission, e.g.

        You are a philosopher debating ethics.

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
