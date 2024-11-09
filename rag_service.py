from typing import List

class RAGService:
    async def get_similar_statements(self, query: str) -> List[str]:
        """Dummy RAG implementation"""
        import random

        sample_statements = [
            "Democracy needs more direct participation from citizens.",
            "Educational reform should focus on critical thinking.",
            "We need better mechanisms for public discourse.",
            "Technology should enhance democratic processes.",
            "Civic engagement should be taught in schools.",
            "Public policy should be more evidence-based."
        ]

        return random.sample(sample_statements, 3)
