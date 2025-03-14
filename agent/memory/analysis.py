from typing import Optional
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

class MemoryAnalysis(BaseModel):
    is_important: bool
    formatted_memory: Optional[str] = None

MEMORY_ANALYSIS_PROMPT = PromptTemplate.from_template("""Extract and format important personal facts about the user from their message.
Focus on the actual information, not meta-commentary or requests.

Important facts include:
- Personal details (name, age, location)
- Professional info (job, education, skills)
- Preferences (likes, dislikes, favorites)
- Life circumstances (family, relationships)
- Significant experiences or achievements
- Personal goals or aspirations

Rules:
1. Only extract actual facts, not requests or commentary
2. Convert facts into clear, third-person statements
3. If no actual facts are present, mark as not important
4. Remove conversational elements and focus on core information

Examples:
Input: "Hey, could you remember that I love Star Wars?"
Output: {
    "is_important": true,
    "formatted_memory": "Loves Star Wars"
}

Input: "Can you remember my details for next time?"
Output: {
    "is_important": false,
    "formatted_memory": null
}

Message: {message}
Output:""")

class MemoryAnalyzer:
    def __init__(self, temperature: float = 0.1):
        self.llm = ChatOpenAI(temperature=temperature)
    
    async def analyze_memory(self, message: str) -> MemoryAnalysis:
        """Analyze a message to determine importance and format if needed."""
        prompt = MEMORY_ANALYSIS_PROMPT.format(message=message)
        response = await self.llm.ainvoke(prompt)
        
        # Parse the response into a MemoryAnalysis object
        try:
            # Extract the JSON-like content from the response
            content = response.content
            if isinstance(content, str):
                # Convert string representation to dict
                import json
                content = json.loads(content)
            
            return MemoryAnalysis(
                is_important=content.get("is_important", False),
                formatted_memory=content.get("formatted_memory")
            )
        except Exception as e:
            # If parsing fails, return a safe default
            return MemoryAnalysis(is_important=False, formatted_memory=None) 