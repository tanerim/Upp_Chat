import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "conversations.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=10000;")
    return conn


def _ensure_column(cursor, table_name, column_name, column_type):
    columns = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
    existing_column_names = {col[1] for col in columns}
    if column_name not in existing_column_names:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def init_db():
    """Initialize the database and create table if it doesn’t exist."""
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        """
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
    """
    )
    _ensure_column(c, "conversations", "left_model", "TEXT")
    _ensure_column(c, "conversations", "right_model", "TEXT")
    _ensure_column(c, "conversations", "temperature", "REAL")
    _ensure_column(c, "conversations", "top_k", "INTEGER")
    _ensure_column(c, "conversations", "top_p", "REAL")
    _ensure_column(c, "conversations", "conversation", "TEXT")
    _ensure_column(c, "conversations", "created_at", "TEXT")
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            message_index INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            word_frequency TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
    """
    )
    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation_id
        ON conversation_messages(conversation_id)
    """
    )
    c.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_conversation_messages_created_at
        ON conversation_messages(created_at)
    """
    )
    conn.commit()
    conn.close()
