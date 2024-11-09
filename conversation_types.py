from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ConversationReference:
    conversation_id: Optional[str]
    comment_id: Optional[str]

@dataclass
class ConversationHistory:
    messages: List[dict]
    system_prompt: str = "You are a helpful AI assistant."