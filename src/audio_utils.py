# src/audio_utils.py

import numpy as np

def mix_audio(audio1: bytes, audio2: bytes, volume1: float = 1.0, volume2: float = 0.3) -> bytes:
    """
    Mixes two raw audio byte streams together with specified volumes.
    Assumes both are 16-bit PCM audio.

    Args:
        audio1: Primary audio (bytes).
        audio2: Background audio (bytes).
        volume1: Volume multiplier for primary audio (0.0 to 1.0+).
        volume2: Volume multiplier for background audio (0.0 to 1.0+).

    Returns:
        The mixed audio as a byte stream.
    """
    if not audio1:
        return audio2
    if not audio2:
        return audio1

    # Convert to numpy arrays for manipulation
    samples1 = np.frombuffer(audio1, dtype=np.int16).astype(np.float32)
    samples2 = np.frombuffer(audio2, dtype=np.int16).astype(np.float32)

    # Ensure both arrays are the same length for mixing
    min_length = min(len(samples1), len(samples2))
    samples1 = samples1[:min_length]
    samples2 = samples2[:min_length]

    # Apply volumes
    samples1 *= volume1
    samples2 *= volume2

    # Mix the audio by adding the samples
    mixed_samples = samples1 + samples2

    # Prevent clipping by clamping values to the 16-bit range
    mixed_samples = np.clip(mixed_samples, -32768, 32767)

    # Convert back to int16 and then to bytes
    return mixed_samples.astype(np.int16).tobytes()