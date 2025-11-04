# db.py
import sqlite3
from pathlib import Path

DB_PATH = Path("data/conversations.db")
DB_PATH.parent.mkdir(exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            left_model TEXT,
            right_model TEXT,
            temperature REAL,
            top_k INTEGER,
            top_p REAL,
            conversation TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def get_connection():
    return sqlite3.connect(DB_PATH)
