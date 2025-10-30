#!/usr/bin/env python3
"""
Silero VAD - ML-based Voice Activity Detection
Filters noise from real human speech with high accuracy
"""

import torch
import numpy as np
import logging
from typing import Optional


class SileroVAD:
    """
    Lightweight Silero VAD wrapper for speech detection
    Each instance maintains state for continuous audio stream
    """

    def __init__(self, threshold: float = 0.5, sample_rate: int = 8000,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Silero VAD

        Args:
            threshold: Speech probability threshold (0.0-1.0)
                      0.0 = accept almost all audio
                      0.3 = very sensitive (catches whispers)
                      0.5 = balanced (recommended)
                      0.7 = strict (only clear speech)
            sample_rate: Audio sample rate (8000 or 16000)
            logger: Logger instance
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.logger = logger or logging.getLogger(__name__)

        # Load Silero VAD model
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                verbose=False
            )
            self.model.eval()

            self.logger.info(f"✅ Silero VAD loaded (threshold={threshold}, rate={sample_rate}Hz)")

        except Exception as e:
            self.logger.error(f"Failed to load Silero VAD: {e}")
            raise

    def is_speech(self, audio_chunk: bytes) -> bool:
        """
        Check if audio chunk contains human speech

        Args:
            audio_chunk: Raw PCM audio bytes (16-bit), variable size

        Returns:
            True if speech detected, False if noise/silence
        """
        try:
            # Convert bytes to float32 numpy array
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
            audio_array = audio_array / 32768.0  # Normalize to [-1, 1]

            # Silero requires fixed chunk sizes: 256 for 8kHz, 512 for 16kHz
            chunk_size = 512 if self.sample_rate == 16000 else 256

            # If audio is shorter than chunk size, pad with zeros
            if len(audio_array) < chunk_size:
                audio_array = np.pad(audio_array, (0, chunk_size - len(audio_array)), mode='constant')

            # Split into fixed-size chunks and process each
            max_speech_prob = 0.0
            num_chunks = len(audio_array) // chunk_size

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                chunk = audio_array[start_idx:end_idx]

                # Convert to tensor and get speech probability
                chunk_tensor = torch.from_numpy(chunk)
                with torch.no_grad():
                    speech_prob = self.model(chunk_tensor, self.sample_rate).item()

                # Track maximum probability across all chunks
                max_speech_prob = max(max_speech_prob, speech_prob)

            # Determine if speech based on maximum probability
            is_speech = max_speech_prob >= self.threshold

            # Optional debug logging
            if not is_speech:
                self.logger.debug(f"Silero rejected: max_prob={max_speech_prob:.3f} < {self.threshold} ({num_chunks} chunks)")

            return is_speech

        except Exception as e:
            self.logger.error(f"Silero VAD error: {e}")
            # On error, pass through (don't filter)
            return True

    def reset_state(self):
        """Reset internal state for new audio stream"""
        try:
            self.model.reset_states()
        except Exception as e:
            self.logger.warning(f"Failed to reset Silero state: {e}")

    def get_speech_probability(self, audio_chunk: bytes) -> float:
        """
        Get raw speech probability without thresholding

        Args:
            audio_chunk: Raw PCM audio bytes, variable size

        Returns:
            Speech probability (0.0-1.0), maximum across all chunks
        """
        try:
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
            audio_array = audio_array / 32768.0

            # Silero requires fixed chunk sizes: 256 for 8kHz, 512 for 16kHz
            chunk_size = 512 if self.sample_rate == 16000 else 256

            # If audio is shorter than chunk size, pad with zeros
            if len(audio_array) < chunk_size:
                audio_array = np.pad(audio_array, (0, chunk_size - len(audio_array)), mode='constant')

            # Split into fixed-size chunks and get maximum probability
            max_speech_prob = 0.0
            num_chunks = len(audio_array) // chunk_size

            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size
                chunk = audio_array[start_idx:end_idx]

                chunk_tensor = torch.from_numpy(chunk)
                with torch.no_grad():
                    speech_prob = self.model(chunk_tensor, self.sample_rate).item()

                max_speech_prob = max(max_speech_prob, speech_prob)

            return max_speech_prob

        except Exception as e:
            self.logger.error(f"Silero probability error: {e}")
            return 0.5  # Return neutral probability on error
