import streamlit as st
from dataclasses import dataclass
from typing import List
import asyncio
import requests
from voice_assistant import VoiceAssistant, AssistantState
from components import StatementDisplay
from prompt_processor import PromptProcessor
from helpers import read_config

# Read config and set up
config = read_config("config.yaml")
RAG_URL = "https://69a9-91-194-240-2.ngrok-free.app/post-data"


@dataclass
class Message:
    content: str
    role: str


@dataclass
class Statement:
    text: str
    votes: int = 0
    id: str = ""


class RAGService:
    """RAG implementation"""

    def get_similar_statements(self, query: str):
        response = requests.post(RAG_URL, params={"text": query}, timeout=10)
        response = response.json()
        return response


class SessionState:
    def __init__(self):
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "system", "content": config["system_prompt"]}
            ]
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
            tts_config_path="/home/besher/Documents/tools/piper/en_en_US_ljspeech_high_en_US-ljspeech-high.onnx.json",
            state_callback=self.update_state,
        )
        self.rag_service = RAGService()
        self.statement_display = StatementDisplay()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def update_state(self, state: AssistantState) -> None:
        if self.progress_container:
            with self.progress_container:
                if state == AssistantState.LISTENING:
                    st.warning("Listening...", icon="👂")
                elif state == AssistantState.PROCESSING:
                    st.warning("Processing your input...", icon="⚙️")
                elif state == AssistantState.VERIFYING:
                    st.warning("Please confirm if I understood correctly...", icon="🤔")
                elif state == AssistantState.COMPLETED:
                    st.success("Successfully captured your statement!", icon="✅")
                elif state == AssistantState.FAILED:
                    st.error("Could not complete the interaction", icon="❌")

    async def process_text_input(self, text_input: str) -> None:
        self.progress_container = st.empty()

        # Get initial processing result
        statement, state = self.loop.run_until_complete(
            self.voice_assistant.process_text_input(text_input)
        )

        if state == AssistantState.VERIFYING and statement:
            # Show confirmation buttons
            col1, col2 = st.columns(2)
            with col1:
                confirm = st.button("Yes, that's correct")
            with col2:
                deny = st.button("No, try again")

            if confirm or deny:
                final_statement, is_confirmed = self.loop.run_until_complete(
                    self.voice_assistant.process_voice_input()
                )

                if is_confirmed and final_statement:
                    # Process final confirmed statement
                    similar_statements = self.rag_service.get_similar_statements(
                        final_statement
                    )
                    st.session_state.current_suggestions = [
                        Statement(text=stmt["text"], id=stmt["hit_id"])
                        for stmt in similar_statements
                    ]
                    st.session_state.messages.append(
                        {"role": "user", "content": final_statement}
                    )
                    return

        if state == AssistantState.FAILED:
            st.error("Failed to process input", icon="❌")

    async def handle_voice_interaction(self) -> None:
        self.progress_container = st.empty()

        final_statement, state = await self.voice_assistant.process_voice_input()
        if state == AssistantState.COMPLETED:
            # Process final statement
            if final_statement:
                similar_statements = self.rag_service.get_similar_statements(
                    final_statement
                )
                st.session_state.current_suggestions = [
                    Statement(text=stmt["text"], id=stmt["hit_id"])
                    for stmt in similar_statements
                ]
                st.session_state.messages.append(
                    {"role": "user", "content": final_statement}
                )

    def render_suggestions(self):
        if st.session_state.current_suggestions:
            st.markdown("### 💭 Similar Statements")

            # Generate text for TTS
            statements_text = "\n".join(
                [
                    f"{idx + 1}. {suggestion.text}"
                    for idx, suggestion in enumerate(
                        st.session_state.current_suggestions
                    )
                ]
            )
            self.voice_assistant.tts.speak(
                "Here are some similar statements. Do you feel any of these represent your views? "
                f"{statements_text}"
            )

            # Custom CSS for the suggestion cards
            st.markdown(
                """
                <style>
                .suggestion-card {
                    background-color: #f0f2f6;
                    border-radius: 10px;
                    padding: 15px;
                    margin: 10px 0;
                    border: 1px solid #e0e3e9;
                }
                .suggestion-text {
                    font-size: 16px;
                    color: #1f2937;
                }
                </style>
            """,
                unsafe_allow_html=True,
            )

            for idx, suggestion in enumerate(st.session_state.current_suggestions):
                # Create a card-like container for each suggestion
                with st.container():
                    st.markdown(
                        f"""
                        <div class="suggestion-card">
                            <p class="suggestion-text">{suggestion.text}</p>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                    # Right-aligned upvote button with custom styling
                    col1, col2, col3 = st.columns([6, 1, 1])
                    with col3:
                        if st.button(
                            "👍 Agree",
                            key=f"upvote_{idx}",
                            help="Click if you agree with this statement",
                        ):
                            suggestion.votes += 1
                            st.success("✨ Thanks for your input!")
                            st.session_state.current_suggestions = []
                            st.session_state.recording_active = False

    def render_chat_history(self):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    def render_main_interface(self):
        st.title("Multimodal Polis 📈")

        with st.sidebar:
            st.page_link("main_page.py", label="Home", icon="🏠")
            st.page_link("pages/Explore_Data.py", label="Explore Data", icon="🔍")
            st.page_link("pages/Multimodal_Polis.py", label="Chat", icon="👋")

        col1, col2 = st.columns([4, 1])

        with col1:
            text_input = st.text_input("Type your statement:", key="text_input")
            if text_input:
                asyncio.run(self.process_text_input(text_input))

        with col2:
            voice_button = st.button("🎤 Voice Input")
            if voice_button:
                st.session_state.recording_active = True
                asyncio.run(self.handle_voice_interaction())

        self.render_suggestions()
        self.render_chat_history()

        if st.button("Clear Chat"):
            st.session_state.messages = [
                {"role": "system", "content": config["system_prompt"]}
            ]
            st.rerun()

    def run(self):
        self.render_main_interface()

    def __del__(self):
        # Clean up the event loop
        self.loop.close()


# Initialize and run the app
if __name__ == "__main__":
    app = MultimodalPolisApp()
    app.run()
