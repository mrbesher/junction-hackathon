from conversation_types import ConversationReference, ConversationHistory

class ConversationManager:
    def __init__(self):
        self.history = ConversationHistory(messages=[])

    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.history.messages.append({
            "role": role,
            "content": content
        })

    def check_existing_conversation(self, summary: str) -> ConversationReference:
        """Check if similar conversation exists using RAG."""
        # In production, this would search through existing conversations
        return ConversationReference(
            conversation_id=None,
            comment_id=None
        )

    def get_confirmation_prompt(self, summary: str) -> str:
        """Generate confirmation message for the summary."""
        return f"I understood that you want to: {summary}\nDoes this accurately reflect your question? (yes/no)"
