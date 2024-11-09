import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import sounddevice as sd
import wave
from faster_whisper import WhisperModel

@dataclass
class TranscriptionSegment:
    start: float
    end: float
    text: str

@dataclass
class TranscriptionResult:
    language: str
    language_probability: float
    segments: List[TranscriptionSegment]

class AudioRecorder:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.background_noise_level = None
        self.silence_threshold = None
        self.calibration_duration = 1.0  # 1 second calibration

    def _calculate_rms(self, audio_chunk: np.ndarray) -> float:
        if audio_chunk.size == 0 or np.all(audio_chunk == 0):
            return 0.0
        return np.sqrt(np.mean(np.square(audio_chunk.astype(float))))

    def _calibrate_noise(self, chunk_duration: float = 0.1) -> None:
        """Calibrate the background noise level."""
        print("Calibrating background noise... Please stay quiet.")
        chunk_samples = int(chunk_duration * self.sample_rate)
        calibration_chunks = int(self.calibration_duration / chunk_duration)

        noise_levels = []
        for _ in range(calibration_chunks):
            chunk = sd.rec(chunk_samples, samplerate=self.sample_rate,
                         channels=1, dtype=np.int16)
            sd.wait()
            rms = self._calculate_rms(chunk)
            noise_levels.append(rms)

        # Calculate baseline noise level and set threshold
        self.background_noise_level = np.mean(noise_levels)
        self.silence_threshold = self.background_noise_level * 2.5
        print(f"Calibration complete. Background noise level: {self.background_noise_level:.2f}")

    def record_until_silence(
        self,
        max_duration: float = 30,
        chunk_duration: float = 0.1,
        min_silence_duration: float = 1.0
    ) -> np.ndarray:
        # First, calibrate the noise level
        self._calibrate_noise(chunk_duration)

        chunk_samples = int(chunk_duration * self.sample_rate)
        chunks = []

        # Parameters for speech detection
        speech_started = False
        silence_chunks = 0
        required_silence_chunks = int(min_silence_duration / chunk_duration)

        print("Recording... Speak now.")
        start_time = time.time()

        while (time.time() - start_time) < max_duration:
            chunk = sd.rec(chunk_samples, samplerate=self.sample_rate,
                         channels=1, dtype=np.int16)
            sd.wait()

            current_rms = self._calculate_rms(chunk)
            chunks.append(chunk)

            # Detect if this chunk is silence
            is_silence = current_rms <= self.silence_threshold

            # If we haven't detected speech yet, check if this chunk is speech
            if not speech_started:
                if not is_silence:
                    speech_started = True
                    print("Speech detected!")
                    silence_chunks = 0
            # If we have detected speech, check for ending silence
            elif is_silence:
                silence_chunks += 1
                if silence_chunks >= required_silence_chunks:
                    print("End of speech detected.")
                    break
            else:
                silence_chunks = 0

        # If we never detected speech, return empty array
        if not speech_started:
            print("No speech detected!")
            return np.array([], dtype=np.int16)

        return np.concatenate(chunks)

class WhisperTranscriber:
    def __init__(self, model_size: str = "large-v3"):
        # Attempt to detect GPU support with CUDA; default to CPU if unavailable
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                compute_type = "float16"
                print("CUDA available: Using GPU with float16 precision.")
            else:
                device = "cpu"
                compute_type = "int8"
                print("CUDA not available: Using CPU with int8 precision.")
        except ImportError:
            device = "cpu"
            compute_type = "int8"
            print("Torch not installed: Defaulting to CPU with int8 precision.")

        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.recorder = AudioRecorder()

    def transcribe_audio(self, audio_data: np.ndarray) -> TranscriptionResult:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            self._save_wav(tmp_file.name, audio_data)
            result = self._process_audio_file(tmp_file.name)
            os.unlink(tmp_file.name)
            return result

    def transcribe_file(self, audio_path: str | Path) -> TranscriptionResult:
        return self._process_audio_file(str(audio_path))

    def record_and_transcribe(self,
                            max_duration: float = 30,
                            silence_duration: float = 3.0) -> TranscriptionResult:
        audio_data = self.recorder.record_until_silence(
            max_duration=max_duration,
            min_silence_duration=silence_duration
        )
        return self.transcribe_audio(audio_data)

    def _save_wav(self, filename: str, audio_data: np.ndarray) -> None:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.recorder.sample_rate)
            wf.writeframes(audio_data.tobytes())

    def _process_audio_file(self, audio_path: str) -> TranscriptionResult:
        segments, info = self.model.transcribe(audio_path, beam_size=5)
        return TranscriptionResult(
            language=info.language,
            language_probability=info.language_probability,
            segments=[
                TranscriptionSegment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text
                )
                for segment in segments
            ]
        )

def print_transcription(result: TranscriptionResult) -> None:
    print(f"\nDetected language: {result.language} "
          f"(probability: {result.language_probability:.2f})")

    for segment in result.segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

def main():
    transcriber = WhisperTranscriber()
    print("Initializing speech recognition...")

    result = transcriber.record_and_transcribe(
        max_duration=30,
        silence_duration=2
    )

    print_transcription(result)

if __name__ == "__main__":
    main()
