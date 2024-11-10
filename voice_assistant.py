from typing import Optional, Union
import requests
from conversation_manager import ConversationManager
from transcriber import TranscriptionService, WhisperTranscriber
from text_to_speech import TextToSpeech
from prompt_processor import PromptProcessor

from enum import Enum


class AssistantState(Enum):
    IDLE = "Waiting to start..."
    LISTENING = "Listening for your input..."
    PROCESSING = "Processing your input..."
    VERIFYING = "Please confirm if I understood correctly..."
    COMPLETED = "Successfully captured your statement!"
    FAILED = "Could not complete the interaction"


class LLMClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def get_response(self, messages: list) -> Optional[str]:
        try:
            response = requests.post(
                url=self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": "qwen/qwen-2.5-72b-instruct",
                    "messages": messages,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error getting LLM response: {e}")
            return None


class VoiceAssistant:
    def __init__(
        self,
        api_key: str,
        tts_model_path: str,
        tts_config_path: str,
        use_gpu: bool = False,
        state_callback=None,
    ):
        self.state = AssistantState.IDLE
        self.state_callback = state_callback
        self.llm_client = LLMClient(api_key)
        self.transcriber = TranscriptionService()
        self.tts = TextToSpeech(
            use_gpu=use_gpu, model_path=tts_model_path, config_path=tts_config_path
        )
        self.conversation_history = []
        self.MAX_CLARIFICATION_TURNS = 3

    async def get_voice_input(self) -> Optional[str]:
        """Record and transcribe voice input."""
        result = await self.transcriber.transcribe_speech()
        print(result)
        if not result.segments:
            return None
        return " ".join(segment.text for segment in result.segments)

    async def confirm_summary(self, summary: str) -> bool:
        """Ask for user confirmation of the summary."""
        confirmation_prompt = f"I understood that: {summary}. Is this correct?"
        self.tts.speak(confirmation_prompt)
        print(confirmation_prompt)

        response = await self.get_voice_input()
        if not response:
            return False

        response = response.lower()
        return "yes" in response or "correct" in response

    def set_state(self, state: AssistantState) -> None:
        """Set the current state of the assistant."""
        self.state = state
        if self.state_callback:
            self.state_callback(state)

    async def process_voice_input(
        self,
    ) -> Optional[tuple[Union[str, None], AssistantState]]:
        """Returns (final_statement, state) tuple"""
        # Initial voice input
        self.set_state(AssistantState.LISTENING)
        initial_input = await self.get_voice_input()
        if not initial_input:
            self.set_state(AssistantState.FAILED)
            return None, self.state

        # Get initial summary from LLM
        self.set_state(AssistantState.PROCESSING)
        summary_prompt = PromptProcessor.create_summary_prompt(initial_input)
        self.conversation_history = [{"role": "user", "content": summary_prompt}]

        # Get initial LLM response before entering the loop
        llm_response = self.llm_client.get_response(self.conversation_history)
        if not llm_response:
            self.set_state(AssistantState.FAILED)
            return None, self.state

        attempts = 0
        while attempts < self.MAX_CLARIFICATION_TURNS:
            # Extract summary from current LLM response
            summary_data = PromptProcessor.extract_yaml_content(llm_response)
            if not summary_data or "statement" not in summary_data:
                self.set_state(AssistantState.FAILED)
                return None, self.state

            # Ask user for confirmation
            self.set_state(AssistantState.VERIFYING)
            confirmation_prompt = f"I understood your statement as: {summary_data['statement']}. Is this correct?"
            self.tts.speak(confirmation_prompt)

            # Get user's confirmation response
            self.set_state(AssistantState.LISTENING)
            user_confirmation = await self.get_voice_input()
            if not user_confirmation:
                self.set_state(AssistantState.FAILED)
                return None, self.state

            # Ask LLM to verify user's response and potentially modify statement
            verification_prompt = (
                f"Based on the my response: '{user_confirmation}', provide a YAML response with:\n"
                "'statement': the original statement if confirmed, or a modified version based on feedback\n"
                "'confirmed': true if I confirmed, false if requested changes or need clarification"
            )

            self.conversation_history.append(
                {"role": "assistant", "content": llm_response}
            )
            self.conversation_history.append(
                {"role": "user", "content": verification_prompt}
            )

            # Get LLM's verification and potential modification
            llm_response = self.llm_client.get_response(self.conversation_history)
            if not llm_response:
                self.set_state(AssistantState.FAILED)
                return None, self.state

            # Check if user confirmed
            verification_data = PromptProcessor.extract_yaml_content(llm_response)
            if verification_data and verification_data.get("confirmed", False):
                self.set_state(AssistantState.COMPLETED)
                return verification_data["statement"], self.state

            attempts += 1

        self.set_state(AssistantState.FAILED)
        return None, self.state

    async def process_text_input(
        self, text_input: str
    ) -> Optional[tuple[Union[str, None], AssistantState]]:
        """Process text input similar to voice input"""
        self.set_state(AssistantState.PROCESSING)

        # Create initial summary prompt
        summary_prompt = PromptProcessor.create_summary_prompt(text_input)
        self.conversation_history = [{"role": "user", "content": summary_prompt}]

        # Get initial LLM response
        llm_response = self.llm_client.get_response(self.conversation_history)
        if not llm_response:
            self.set_state(AssistantState.FAILED)
            return None, self.state

        attempts = 0
        while attempts < self.MAX_CLARIFICATION_TURNS:
            # Extract summary from current LLM response
            summary_data = PromptProcessor.extract_yaml_content(llm_response)
            if not summary_data or "statement" not in summary_data:
                self.set_state(AssistantState.FAILED)
                return None, self.state

            # Set verification state
            self.set_state(AssistantState.VERIFYING)

            # Return the current statement and state for UI confirmation
            return summary_data["statement"], self.state

        self.set_state(AssistantState.FAILED)
        return None, self.state

    def process_confirmation(
        self, statement: str, is_confirmed: bool
    ) -> Optional[tuple[str, bool]]:
        """Process user confirmation response"""
        verification_prompt = (
            f'Does this statement imply confirmation? {"yes" if is_confirmed else "no"}\n'
            f'Use the same YAML format with "confirmed" and "statement" keys.'
        )

        self.conversation_history.append(
            {"role": "user", "content": verification_prompt}
        )
        verify_response = self.llm_client.get_response(self.conversation_history)

        if not verify_response:
            return None, False

        confirmation_data = PromptProcessor.extract_yaml_content(verify_response)
        if confirmation_data and confirmation_data.get("confirmed", False):
            self.set_state(AssistantState.COMPLETED)
            return statement, True

        return None, False
