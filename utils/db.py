"""
utils/db.py
-----------
Database Connection Utility for LangGraph + pgvector.
Handles:
- PostgreSQL + LangGraph store setup
- Connection health check
- Data CRUD operations (put, get, delete)
"""

import os
from dotenv import load_dotenv
from psycopg import Connection
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.errors import OperationalError
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# --------------------------------------------------------
# Load environment variables
# --------------------------------------------------------
load_dotenv()

PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DB = os.getenv("PG_DB", "rag")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_HOST = os.getenv("PG_HOST", "localhost")

DB_URI = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}?sslmode=disable"

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

# --------------------------------------------------------
# Setup pgvector store + LangGraph checkpointer
# --------------------------------------------------------

def create_store_connections():
    """Initialize LangGraph store & checkpointer with pgvector."""
    # --- Initialize embeddings ---
    embedding_dim = 3072
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        embedding_kwargs={"output_dimensionality": embedding_dim},
    )

    # --- Connect to Postgres ---
    store_conn = Connection.connect(DB_URI, **connection_kwargs)
    saver_conn = Connection.connect(DB_URI, **connection_kwargs)

    # --- Initialize Store + Saver ---
    store = PostgresStore(
        store_conn,
        index={"embed": embeddings, "dims": embedding_dim, "hnsw": False}
    )
    checkpointer = PostgresSaver(saver_conn)

    # --- Setup tables if not exist ---
    store.setup()
    checkpointer.setup()

    print("Database connections established & store initialized.")
    return store, checkpointer, store_conn, saver_conn


# Initialize on import
store, checkpointer, store_conn, saver_conn = create_store_connections()

# --------------------------------------------------------
# Utility Functions
# --------------------------------------------------------

def check_connection() -> bool:
    """Check if PostgreSQL connection is alive."""
    try:
        with store_conn.cursor() as cur:
            cur.execute("SELECT 1;")
        print("Database connection active.")
        return True
    except OperationalError as e:
        print(f"Database connection error: {e}")
        return False


def put_data(namespace: tuple, key: str, value: dict):
    """Insert or update record in LangGraph store."""
    try:
        store.put(namespace, key=key, value=value, index=["text"])
        print(f"Data inserted for {namespace} → {key}")
    except Exception as e:
        print(f"Failed to insert data: {e}")


def get_data(namespace: tuple, key: str):
    """Retrieve record from LangGraph store."""
    try:
        record = store.get(namespace, key=key)
        if record:
            print(f"Data fetched for {namespace}:{key}")
            return record.value
        print(f"No record found for {namespace}:{key}")
        return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def delete_data(namespace: tuple, key: str):
    """Delete a record from LangGraph store."""
    try:
        store.delete(namespace, key=key)
        print(f"Data deleted for {namespace}:{key}")
    except Exception as e:
        print(f"Error deleting data: {e}")


def close_connections():
    """Gracefully close Postgres connections."""
    try:
        store_conn.close()
        saver_conn.close()
        print("Database connections closed.")
    except Exception as e:
        print(f"Error closing connections: {e}")

# --------------------------------------------------------
# Example manual test (optional)
# --------------------------------------------------------
if __name__ == "__main__":
    check_connection()
    ns = ("demo_user", "test_memory")
    put_data(ns, "chunk_1", {"text": "This is a test chunk"})
    print(get_data(ns, "chunk_1"))
    delete_data(ns, "chunk_1")
    close_connections()
