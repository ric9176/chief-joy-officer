from datetime import datetime
from .manager import MemoryManager, Message
from .analysis import MemoryAnalyzer
import uuid

async def main():
    # Initialize the memory system
    memory_manager = MemoryManager(
        qdrant_url="http://localhost:6333",
        sqlite_path="data/short_term.db"
    )
    memory_analyzer = MemoryAnalyzer()
    
    try:
        # Example: Store a message in short-term memory
        message = Message(
            content="I really enjoy programming in Python and building AI applications.",
            type="human",
            timestamp=datetime.now()
        )
        await memory_manager.store_short_term(message)
        
        # Analyze the message for long-term storage
        analysis = await memory_analyzer.analyze_memory(message.content)
        
        if analysis.is_important and analysis.formatted_memory:
            # Check for similar existing memories
            similar = memory_manager.find_similar_memory(analysis.formatted_memory)
            
            if not similar:
                # Store in long-term memory if no similar memory exists
                memory_manager.store_long_term(
                    text=analysis.formatted_memory,
                    metadata={
                        "id": str(uuid.uuid4()),
                        "timestamp": datetime.now().isoformat(),
                        "source_message": message.content
                    }
                )
        
        # Retrieve recent messages from short-term memory
        recent_messages = memory_manager.get_recent_messages(limit=5)
        print("Recent messages:")
        for msg in recent_messages:
            print(f"- {msg.timestamp}: {msg.content}")
            
    finally:
        memory_manager.close()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 