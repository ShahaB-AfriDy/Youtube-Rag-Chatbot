"""
Database Connection Utility for LangGraph + pgvector
---------------------------------------------------
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
from langgraph.store.base import BaseStore
from psycopg.errors import OperationalError

# --------------------------------------------------------
# Load Environment Variables
# --------------------------------------------------------
load_dotenv()

PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
PG_DB = os.getenv("PG_DB", "YoutubeRagChatbot")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_HOST = os.getenv("PG_HOST", "localhost")

DB_URI = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}?sslmode=disable"

connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}


# --------------------------------------------------------
# Connection Setup
# --------------------------------------------------------
def create_store_connections():
    """Create connections for LangGraph Store and Checkpointer."""
    store_conn = Connection.connect(DB_URI, **connection_kwargs)
    saver_conn = Connection.connect(DB_URI, **connection_kwargs)

    store = PostgresStore(store_conn)
    checkpointer = PostgresSaver(saver_conn)

    store.setup()
    checkpointer.setup()

    print("✅ Database connections established and setup complete.")
    return store, checkpointer, store_conn, saver_conn


# Initialize on import
store, checkpointer, store_conn, saver_conn = create_store_connections()


# --------------------------------------------------------
# Utility Functions
# --------------------------------------------------------

def check_connection() -> bool:
    """Check if the database connection is active."""
    try:
        with store_conn.cursor() as cur:
            cur.execute("SELECT 1;")
        print("Database connection active.")
        return True
    except OperationalError as e:
        print(f"Database connection error: {e}")
        return False


def put_data(namespace: tuple, key: str, value: dict):
    """Store data in LangGraph store."""
    try:
        store.put(namespace, key=key, value=value)
        print(f"Data inserted: {namespace} -> {key}")
    except Exception as e:
        print(f"Failed to insert data: {e}")


def get_data(namespace: tuple, key: str):
    """Retrieve data from LangGraph store."""
    try:
        record = store.get(namespace, key=key)
        if record:
            print(f"Data fetched for {namespace}:{key}")
            return record.value
        else:
            print(f"No record found for {namespace}:{key}")
            return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None


def delete_data(namespace: tuple, key: str):
    """Delete a record from LangGraph store."""
    try:
        store.delete(namespace, key=key)
        print(f"Data deleted: {namespace}:{key}")
    except Exception as e:
        print(f"Error deleting data: {e}")


def close_connections():
    """Close all Postgres connections."""
    try:
        store_conn.close()
        saver_conn.close()
        print("Database connections closed.")
    except Exception as e:
        print(f"Error closing connections: {e}")


# --------------------------------------------------------
# Example Usage (You can remove this part later)
# --------------------------------------------------------
if __name__ == "__main__":
    # Connection test
    check_connection()

    # Test Namespace
    ns = ("test_user", "memory")

    # Insert sample
    put_data(ns, "profile", {"name": "Shahab", "city": "Peshawar"})

    # Get sample
    print(get_data(ns, "profile"))

    # Delete sample
    delete_data(ns, "profile")

    # Close
    close_connections()
