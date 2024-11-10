import asyncio
import base64
import json
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable, AsyncIterator

import numpy as np
import sounddevice as sd
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


class AudioConfig:
    def __init__(
        self, sample_rate: int = 16000, channels: int = 1, dtype: np.dtype = np.int16
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype


class AudioRecorder:
    def __init__(
        self,
        config: AudioConfig,
        on_speech_start: Optional[Callable] = None,
        on_speech_end: Optional[Callable] = None,
    ):
        self.config = config
        self.background_noise_level = None
        self.silence_threshold = None
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end

    async def _calibrate_noise(self, chunk_duration: float = 0.1) -> None:
        """Calibrate background noise asynchronously"""
        chunk_samples = int(chunk_duration * self.config.sample_rate)
        calibration_chunks = int(1.0 / chunk_duration)  # 1 second calibration

        noise_levels = []
        for _ in range(calibration_chunks):
            chunk = sd.rec(
                chunk_samples,
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
            )
            sd.wait()
            rms = self._calculate_rms(chunk)
            noise_levels.append(rms)
            await asyncio.sleep(0.001)  # Allow other tasks to run

        self.background_noise_level = np.mean(noise_levels)
        self.silence_threshold = self.background_noise_level * 2.5

    def _calculate_rms(self, audio_chunk: np.ndarray) -> float:
        if audio_chunk.size == 0 or np.all(audio_chunk == 0):
            return 0.0
        return np.sqrt(np.mean(np.square(audio_chunk.astype(float))))

    async def record_until_silence(
        self,
        max_duration: float = 30,
        chunk_duration: float = 0.1,
        min_silence_duration: float = 1.0,
    ) -> AsyncIterator[np.ndarray]:
        """Record audio asynchronously until silence is detected"""
        await self._calibrate_noise(chunk_duration)

        chunk_samples = int(chunk_duration * self.config.sample_rate)
        speech_started = False
        silence_chunks = 0
        required_silence_chunks = int(min_silence_duration / chunk_duration)

        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < max_duration:
            chunk = sd.rec(
                chunk_samples,
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
            )
            sd.wait()

            current_rms = self._calculate_rms(chunk)
            is_silence = current_rms <= float(
                self.silence_threshold if not None else 0.0
            )

            if not speech_started and not is_silence:
                speech_started = True
                if self.on_speech_start:
                    self.on_speech_start()

            if speech_started:
                yield chunk

                if is_silence:
                    silence_chunks += 1
                    if silence_chunks >= required_silence_chunks:
                        if self.on_speech_end:
                            self.on_speech_end()
                        break
                else:
                    silence_chunks = 0

            await asyncio.sleep(0.001)


class WhisperTranscriber:
    def __init__(self, model_size: str = "large-v3"):
        self.model = self._initialize_model(model_size)

    def _initialize_model(self, model_size: str) -> WhisperModel:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
        except ImportError:
            device = "cpu"
            compute_type = "int8"

        return WhisperModel(model_size, device=device, compute_type=compute_type)

    async def transcribe_stream(
        self, audio_stream: AsyncIterator[np.ndarray], temp_file: Path
    ) -> TranscriptionResult:
        """Transcribe audio from async stream"""
        # Collect all chunks first to ensure we have valid audio data
        chunks = []
        async for chunk in audio_stream:
            if chunk is not None and chunk.size > 0:
                chunks.append(chunk)

        # Check if we have any valid audio data
        if not chunks:
            raise ValueError("No valid audio data received")

        # Get channel count from the first chunk
        first_chunk = chunks[0]
        n_channels = first_chunk.shape[1] if len(first_chunk.shape) > 1 else 1

        with wave.open(str(temp_file), "wb") as wf:
            wf.setnchannels(n_channels)
            wf.setsampwidth(first_chunk.dtype.itemsize)
            wf.setframerate(16000)  # Default sample rate

            for chunk in chunks:
                wf.writeframes(chunk.tobytes())

        return await self.transcribe_file(temp_file)

    async def transcribe_file(self, audio_path: Path) -> TranscriptionResult:
        """Transcribe audio file asynchronously"""
        segments, info = self.model.transcribe(str(audio_path))
        return TranscriptionResult(
            language=info.language,
            language_probability=info.language_probability,
            segments=[
                TranscriptionSegment(
                    start=segment.start, end=segment.end, text=segment.text
                )
                for segment in segments
            ],
        )


class TranscriptionService:
    def __init__(
        self, audio_config: Optional[AudioConfig] = None, model_size: str = "large-v3"
    ):
        self.audio_config = audio_config or AudioConfig()
        self.transcriber = WhisperTranscriber(model_size)
        self.recorder = AudioRecorder(
            self.audio_config,
            on_speech_start=lambda: print("Speech detected"),
            on_speech_end=lambda: print("Speech ended"),
        )

    async def transcribe_speech(
        self,
        max_duration: float = 30,
        silence_duration: float = 1.5,
        temp_path: Optional[Path] = None,
    ) -> TranscriptionResult:
        temp_path = temp_path or Path("temp_audio.wav")

        audio_stream = self.recorder.record_until_silence(
            max_duration=max_duration, min_silence_duration=silence_duration
        )

        return await self.transcriber.transcribe_stream(audio_stream, temp_path)


# Example usage
async def main():
    service = TranscriptionService()
    result = await service.transcribe_speech()
    print(f"\nDetected language: {result.language}")
    for segment in result.segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")


if __name__ == "__main__":
    asyncio.run(main())
