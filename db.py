#db.py

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "conversations.db")


def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    """Initialize the database and create table if it doesn’t exist."""
    conn = get_connection()
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
