from datetime import datetime
import sqlite3
import uuid
from typing import Optional, List
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import os

class MemoryAnalysis(BaseModel):
    is_important: bool
    formatted_memory: Optional[str] = None

class Message(BaseModel):
    content: str
    type: str  # "human" or "ai"
    timestamp: datetime

class MemoryManager:
    SIMILARITY_THRESHOLD = 0.9
    
    def __init__(self, qdrant_url: str = "http://localhost:6333", sqlite_path: str = "data/short_term.db"):
        # Initialize vector store (Qdrant)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.qdrant = QdrantClient(url=qdrant_url)
        
        # Ensure the collection exists
        try:
            self.qdrant.get_collection("memories")
        except:
            self.qdrant.create_collection(
                collection_name="memories",
                vectors_config={
                    "size": self.model.get_sentence_embedding_dimension(),
                    "distance": "Cosine"
                }
            )
        
        # Initialize SQLite for short-term memory
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        self.sqlite_conn = sqlite3.connect(sqlite_path)
        self._init_sqlite()
    
    def _init_sqlite(self):
        """Initialize SQLite tables for short-term memory."""
        cursor = self.sqlite_conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                type TEXT NOT NULL,
                timestamp DATETIME NOT NULL
            )
        """)
        self.sqlite_conn.commit()
    
    async def store_short_term(self, message: Message) -> None:
        """Store a message in short-term memory (SQLite)."""
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (id, content, type, timestamp) VALUES (?, ?, ?, ?)",
            (str(uuid.uuid4()), message.content, message.type, message.timestamp.isoformat())
        )
        self.sqlite_conn.commit()
    
    def get_recent_messages(self, limit: int = 10) -> List[Message]:
        """Retrieve recent messages from short-term memory."""
        cursor = self.sqlite_conn.cursor()
        cursor.execute(
            "SELECT content, type, timestamp FROM conversations ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        return [
            Message(content=row[0], type=row[1], timestamp=datetime.fromisoformat(row[2]))
            for row in cursor.fetchall()
        ]
    
    def store_long_term(self, text: str, metadata: dict) -> None:
        """Store a memory in long-term memory (Qdrant)."""
        embedding = self.model.encode(text)
        self.qdrant.upsert(
            collection_name="memories",
            points=[PointStruct(
                id=metadata["id"],
                vector=embedding.tolist(),
                payload={"text": text, **metadata}
            )]
        )
    
    def find_similar_memory(self, text: str) -> Optional[str]:
        """Find similar existing memory using cosine similarity."""
        embedding = self.model.encode(text)
        results = self.qdrant.search(
            collection_name="memories",
            query_vector=embedding.tolist(),
            limit=1
        )
        if results and results[0].score > self.SIMILARITY_THRESHOLD:
            return results[0].payload["text"]
        return None
    
    def close(self):
        """Close database connections."""
        self.sqlite_conn.close() 