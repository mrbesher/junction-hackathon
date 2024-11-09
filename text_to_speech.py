import json
import logging
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime
import sounddevice as sd


class TextToSpeech:
    def __init__(
        self,
        model_path: str,
        config_path: str,
        use_gpu: bool = False,
        sample_rate: int = 22050,
    ):
        """Initialize the text to speech engine.

        Args:
            model_path: Path to the ONNX model file
            config_path: Path to the model config JSON file
            use_gpu: Whether to use GPU acceleration if available
            sample_rate: Audio sample rate to use
        """
        self.sample_rate = sample_rate
        self._init_logger()
        self.model, self.config = self._load_model_and_config(
            model_path, config_path, use_gpu
        )

    def _init_logger(self) -> None:
        """Initialize logging configuration."""
        self.logger = logging.getLogger("TextToSpeech")
        self.logger.setLevel(logging.INFO)

    def _load_model_and_config(
        self, model_path: str, config_path: str, use_gpu: bool
    ) -> Tuple[onnxruntime.InferenceSession, dict]:
        """Load the ONNX model and its configuration.

        Args:
            model_path: Path to ONNX model file
            config_path: Path to config JSON file
            use_gpu: Whether to use GPU acceleration

        Returns:
            Tuple of model session and config dict
        """
        # Set up ONNX runtime options
        providers = [
            "CPUExecutionProvider"
            if not use_gpu
            else ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"})
        ]
        sess_options = onnxruntime.SessionOptions()

        # Load model
        self.logger.info(f"Loading model from {model_path}")
        model = onnxruntime.InferenceSession(
            str(model_path), sess_options=sess_options, providers=providers
        )

        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)

        return model, config

    def _phonemize(self, text: str) -> List[List[str]]:
        """Convert text to phonemes.

        Args:
            text: Input text to convert

        Returns:
            List of phoneme sequences grouped by sentence
        """
        if self.config["phoneme_type"] == "espeak":
            from piper_phonemize import phonemize_espeak

            return phonemize_espeak(text, self.config["espeak"]["voice"])
        else:
            from piper_phonemize import phonemize_codepoints

            return phonemize_codepoints(text)

    def _phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """Convert phonemes to model input IDs.

        Args:
            phonemes: List of phoneme strings

        Returns:
            List of integer IDs
        """
        id_map = self.config["phoneme_id_map"]
        PAD = "_"
        BOS = "^"
        EOS = "$"

        ids = list(id_map[BOS])
        for phoneme in phonemes:
            if phoneme not in id_map:
                self.logger.warning(f"Missing phoneme from id map: {phoneme}")
                continue
            ids.extend(id_map[phoneme])
            ids.extend(id_map[PAD])
        ids.extend(id_map[EOS])
        return ids

    def _audio_float_to_int16(self, audio: np.ndarray) -> np.ndarray:
        """Convert float audio data to 16-bit integers.

        Args:
            audio: Float audio data

        Returns:
            16-bit integer audio data
        """
        audio = np.clip(audio * 32768, -32768, 32767)
        return audio.astype(np.int16)

    def synthesize(
        self,
        text: str,
        speed: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        speaker_id: Optional[int] = None,
    ) -> np.ndarray:
        """Synthesize speech from text.

        Args:
            text: Input text to synthesize
            speed: Speech rate multiplier (1.0 = normal speed)
            noise_scale: Amount of noise in the output
            noise_w: Noise width parameter
            speaker_id: Optional speaker ID for multi-speaker models

        Returns:
            Audio data as numpy array
        """
        # Convert text to phonemes
        phoneme_sequences = self._phonemize(text)

        # Process each sequence
        for phonemes in phoneme_sequences:
            # Convert to IDs
            phoneme_ids = self._phonemes_to_ids(phonemes)

            # Prepare model inputs
            text_array = np.expand_dims(np.array(phoneme_ids, dtype=np.int64), 0)
            text_lengths = np.array([text_array.shape[1]], dtype=np.int64)
            scales = np.array([noise_scale, 1.0 / speed, noise_w], dtype=np.float32)

            # Add speaker ID if needed
            sid = None
            if speaker_id is not None:
                sid = np.array([speaker_id], dtype=np.int64)

            # Run inference
            audio = self.model.run(
                None,
                {
                    "input": text_array,
                    "input_lengths": text_lengths,
                    "scales": scales,
                    "sid": sid,
                },
            )[0].squeeze((0, 1))

            # Convert to 16-bit
            audio = self._audio_float_to_int16(audio.squeeze())
            yield audio

    def speak(
        self,
        text: str,
        speed: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        speaker_id: Optional[int] = None,
    ) -> None:
        """Synthesize and play speech from text.

        Args:
            text: Input text to speak
            speed: Speech rate multiplier (1.0 = normal speed)
            noise_scale: Amount of noise in the output
            noise_w: Noise width parameter
            speaker_id: Optional speaker ID for multi-speaker models
        """
        try:
            for audio_segment in self.synthesize(text, speed, noise_scale, noise_w, speaker_id):
                sd.play(audio_segment, self.sample_rate)
                sd.wait()
        except Exception as e:
            self.logger.error(f"Error playing audio: {e}")
