import os
from dotenv import load_dotenv
from psycopg import Connection
from langgraph.store.postgres import PostgresStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# ---- EMBEDDING SETUP ----
embedding_dim = 3072
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    embedding_kwargs={"output_dimensionality": embedding_dim},
)

# ---- DATABASE CONNECTION ----
user_name = "postgres"
password = "postgres"
database_name = "rag"
port = "5432"

DB_URI = f"postgresql://{user_name}:{password}@localhost:{port}/{database_name}?sslmode=disable"
connection_kwargs = {"autocommit": True, "prepare_threshold": 0}

conn = Connection.connect(DB_URI, **connection_kwargs)
store = PostgresStore(conn, index={"embed": embeddings, "dims": embedding_dim,"hnsw": False})
store.setup()

# ---- TEXT TO STORE ----

with open("Story.txt", "r", encoding="utf-8") as f:
    long_text = f.read()
namespace = ("1", "documents")

# ---- SPLIT LONG TEXT INTO CHUNKS ----
# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# chunks = splitter.split_text(long_text)

# # ---- STORE EACH CHUNK AS EMBEDDING ----
# namespace = ("1", "documents")
# for i, chunk in enumerate(chunks):
#     key = f"chunk_{i+1}"
#     metadata = {"text": chunk}
#     store.put(namespace, key, metadata, index=["text"])

# print(f"✅ Stored {len(chunks)} text chunks successfully in PostgreSQL!")

# # ---- USER QUERY ----


while True:
    user_query = input("\nAsk a question: ")
    if user_query.lower() in ["exit", "quit"]:
        print("Exiting...")
        break
# ---- SEARCH SIMILAR CHUNKS ----

    results = store.search(namespace, query=user_query, limit=3)

    print("\n🔍 Top matches:")
    for i, r in enumerate(results, start=1):
        print(f"\nResult {i}:")
        print(f"Score: {r.score:.3f}")
        print(f"Text snippet: {r.value['text'][:300]}...")

