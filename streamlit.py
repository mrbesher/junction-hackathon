import streamlit as st
from dataclasses import dataclass
from typing import List, Optional
import asyncio
from pathlib import Path
import tempfile
import yaml

from voice_assistant import VoiceAssistant
from components import StatementDisplay


@dataclass
class Message:
    content: str
    role: str

@dataclass
class Statement:
    text: str
    votes: int = 0
    id: str = ""

class DummyRAGService:
    """Temporary RAG implementation"""
    def get_similar_statements(self, query: str) -> List[str]:
        import random
        statements = [
            "We should improve public transportation",
            "Healthcare should be more accessible",
            "Education needs more funding"
        ]
        return random.choices(statements, k=3) if random.random() > 0.5 else []

class SessionState:
    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "current_suggestions" not in st.session_state:
            st.session_state.current_suggestions = []
        if "recording_active" not in st.session_state:
            st.session_state.recording_active = False
        if "awaiting_confirmation" not in st.session_state:
            st.session_state.awaiting_confirmation = False

class MultimodalPolisApp:
    def __init__(self):
        self.state = SessionState()
        self.voice_assistant = VoiceAssistant(
            api_key=st.secrets.get("openrouter_api_key"),
            tts_model_path="/home/besher/Documents/tools/piper/en_US-ljspeech-high.onnx",
            tts_config_path="/home/besher/Documents/tools/piper/en_en_US_ljspeech_high_en_US-ljspeech-high.onnx.json"
        )
        self.rag_service = DummyRAGService()
        self.statement_display = StatementDisplay()

    async def process_text_input(self, text_input: str) -> None:
        """Process text input and generate response"""
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content=text_input)
        ]

        response = await self.voice_assistant.llm_client.get_response(messages)
        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})
            similar_statements = self.rag_service.get_similar_statements(text_input)
            st.session_state.current_suggestions = [
                Statement(text=stmt) for stmt in similar_statements
            ]

    async def handle_voice_interaction(self) -> None:
        """Handle voice interaction and process response"""
        with st.spinner("Listening..."):
            final_statement = await self.voice_assistant.process_voice_input()

            if final_statement:
                similar_statements = self.rag_service.get_similar_statements(final_statement)
                st.session_state.current_suggestions = [
                    Statement(text=stmt) for stmt in similar_statements
                ]
                st.session_state.messages.append({
                    "role": "user",
                    "content": final_statement
                })

    def render_suggestions(self):
        """Render suggestion cards with voting options"""
        if st.session_state.current_suggestions:
            st.markdown("### Similar Statements")
            st.markdown("If any of these statements match your intent, you can upvote them:")

            for idx, suggestion in enumerate(st.session_state.current_suggestions):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(suggestion.text)
                with col2:
                    if st.button("⬆️ Upvote", key=f"upvote_{idx}"):
                        suggestion.votes += 1
                        st.success("Vote recorded!")
                        st.session_state.current_suggestions = []
                        st.session_state.recording_active = False

    def render_chat_history(self):
        """Render chat history"""
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    def render_main_interface(self):
        """Render main application interface"""
        st.title("Multimodal Polis 📈")

        # Input section
        col1, col2 = st.columns([4, 1])

        with col1:
            text_input = st.text_input("Type your statement:", key="text_input")

        with col2:
            voice_button = st.button("🎤 Voice Input")

        # Process inputs
        if text_input:
            asyncio.run(self.process_text_input(text_input))

        if voice_button:
            st.session_state.recording_active = True
            asyncio.run(self.handle_voice_interaction())

        # Display suggestions and chat history
        self.render_suggestions()
        self.render_chat_history()

    def run(self):
        """Main application entry point"""
        self.render_main_interface()

if __name__ == "__main__":
    app = MultimodalPolisApp()
    app.run()
