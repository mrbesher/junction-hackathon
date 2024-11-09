from typing import Optional
import requests
from conversation_manager import ConversationManager
from transcriber import TranscriptionService, WhisperTranscriber
from text_to_speech import TextToSpeech
from prompt_processor import PromptProcessor

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
                    "model": "qwen/qwen-2-7b-instruct:free",
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
    ):
        self.llm_client = LLMClient(api_key)
        self.transcriber = TranscriptionService()
        self.tts = TextToSpeech(use_gpu=use_gpu, model_path=tts_model_path, config_path=tts_config_path)
        self.conversation_history = []
        self.MAX_CLARIFICATION_TURNS = 3

    async def get_voice_input(self) -> Optional[str]:
        """Record and transcribe voice input."""
        result = await self.transcriber.transcribe_speech()
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

    async def process_voice_input(self) -> Optional[str]:
        """Main processing loop for voice input and summary clarification."""
        print("Speak...")

        # Get initial input
        question = await self.get_voice_input()
        if not question:
            print("No speech detected!")
            return None

        # Initialize conversation with system message and user's question
        self.conversation_history = [
            {"role": "user", "content": PromptProcessor.create_summary_prompt(question)}
        ]

        # First attempt - get initial summary
        llm_response = self.llm_client.get_response(self.conversation_history)
        if not llm_response:
            print("Failed to get response from LLM")
            return None

        summary_data = PromptProcessor.extract_yaml_content(llm_response)
        if not summary_data or 'statement' not in summary_data:
            print("Failed to parse LLM response")
            return None

        # Ask user to confirm the initial summary
        confirmation_prompt = f"I understood that: {summary_data['statement']}. Is this correct?"
        self.tts.speak(confirmation_prompt)

        self.conversation_history.append(
            {"role": "assistant", "content": confirmation_prompt}
        )

        confirmation = await self.get_voice_input()
        if not confirmation:
            return None

        # Add user's confirmation response to history
        self.conversation_history.append(
            {"role": "user", "content": confirmation}
        )

        # Get LLM to evaluate the confirmation
        llm_response = self.llm_client.get_response(self.conversation_history)
        if not llm_response:
            return None

        confirmation_data = PromptProcessor.extract_yaml_content(llm_response)
        if confirmation_data and confirmation_data.get('confirmed', False):
            return summary_data['statement']

        # If not confirmed, enter clarification loop
        for attempt in range(self.MAX_CLARIFICATION_TURNS):
            # Ask for clarification
            confirmation_prompt = f"I understood that your statement is: {summary_data['statement']}. Is this correct?"
            self.tts.speak(confirmation_prompt)
            print("Please clarify...")

            clarification = await self.get_voice_input()
            if not clarification:
                return None

            # Add user's clarification to history
            self.conversation_history.append(
                {"role": "user", "content": PromptProcessor.create_summary_prompt(clarification)}
            )

            # Get new LLM response
            llm_response = self.llm_client.get_response(self.conversation_history)
            if not llm_response:
                print("Failed to get response from LLM")
                return None

            summary_data = PromptProcessor.extract_yaml_content(llm_response)
            if not summary_data or 'statement' not in summary_data:
                print("Failed to parse LLM response")
                return None

            # Ask user to confirm the new summary
            confirmation_prompt = f"I understood that: {summary_data['statement']}. Is this correct?"
            self.tts.speak(confirmation_prompt)
            print(confirmation_prompt)

            confirmation = await self.get_voice_input()
            if not confirmation:
                return None

            # Add user's confirmation response to history
            self.conversation_history.append(
                {"role": "user", "content": confirmation}
            )

            # Get LLM to evaluate the confirmation
            llm_response = self.llm_client.get_response(self.conversation_history)
            if not llm_response:
                return None

            confirmation_data = PromptProcessor.extract_yaml_content(llm_response)
            if confirmation_data and confirmation_data.get('confirmed', False):
                return summary_data['statement']

        self.tts.speak(
            "I'm still having trouble understanding completely. "
            "Let's start over with a new question."
        )
        return None
