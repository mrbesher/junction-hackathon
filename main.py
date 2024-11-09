import os
from voice_assistant import VoiceAssistant

def main():
    # Initialize components
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENROUTER_API_KEY environment variable")

    use_gpu = False
    try:
        import torch
        if torch.cuda.is_available():
            use_gpu = True
            print("GPU available. Using GPU for TTS.")
    except ImportError:
        pass

    # Initialize voice assistant
    assistant = VoiceAssistant(api_key,
                               tts_model_path="/home/besher/Documents/tools/piper/en_US-ljspeech-high.onnx",
                               tts_config_path="/home/besher/Documents/tools/piper/en_en_US_ljspeech_high_en_US-ljspeech-high.onnx.json",
                               use_gpu=use_gpu)

    # Process voice input
    assistant.process_voice_input()

if __name__ == "__main__":
    main()
