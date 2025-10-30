#!/usr/bin/env python3
"""
Framework-Agnostic Voicemail Detection Core
Extracted from pjsua2 implementation for reuse across different SIP frameworks

Components:
- VoicemailModelManager: Thread-safe singleton for ML model management
- VoicemailClassifier: Framework-independent classification logic
"""

import os
import time
import tempfile
import threading
import numpy as np
from typing import Optional, Tuple
from pydub import AudioSegment
import torch
import logging
import gc

class VoicemailModelManager:
    """
    Thread-safe singleton for voicemail detection transformer model
    Features:
    - Lazy loading (model loaded only when first needed)
    - Reference counting (automatic cleanup when unused)
    - GPU/CPU detection
    - Automatic model download and caching
    """
    _instance = None
    _instance_lock = threading.Lock()
    _model_pipeline = None
    _model_lock = threading.Lock()
    _load_event = threading.Event()
    _load_error = None
    _reference_count = 0
    _cleanup_timer = None

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_pipeline(self, model_path: str = "models/voicemail_detector"):
        """
        Get or create the model pipeline with reference counting

        Args:
            model_path: Local path to cached model

        Returns:
            Transformer pipeline for audio classification, or None if loading fails
        """
        # Fast path - model already loaded
        if self._model_pipeline is not None:
            with self._model_lock:
                self._reference_count += 1
            return self._model_pipeline

        with self._model_lock:
            # Double-check inside lock
            if self._model_pipeline is not None:
                self._reference_count += 1
                return self._model_pipeline

            # Check if another thread is loading
            if not self._load_event.is_set():
                try:
                    self.logger.info("Loading global VMD model (singleton)...")

                    from transformers import pipeline

                    # Determine device
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    self.logger.info(f"Using device: {device}")

                    # Try to load from local path first
                    if os.path.exists(model_path):
                        self.logger.info(f"Loading from local: {model_path}")
                        self._model_pipeline = pipeline(
                            "audio-classification",
                            model=model_path,
                            device=device
                        )
                    else:
                        # Download and save
                        self.logger.info("Downloading model...")
                        self._model_pipeline = pipeline(
                            "audio-classification",
                            model="jakeBland/wav2vec-vm-finetune",
                            device=device
                        )

                        # Save for future use
                        try:
                            os.makedirs(model_path, exist_ok=True)
                            self._model_pipeline.save_pretrained(model_path)
                            self.logger.info(f"Model saved to {model_path}")
                        except Exception as e:
                            self.logger.warning(f"Could not save model: {e}")

                    self._load_event.set()
                    self._reference_count = 1
                    self.logger.info("✅ VMD model loaded successfully")

                    # Start cleanup timer
                    self._reset_cleanup_timer()

                except Exception as e:
                    self._load_error = str(e)
                    self.logger.error(f"Failed to load VMD model: {e}")
                    return None

            return self._model_pipeline

    def release(self):
        """Release a reference to the model"""
        with self._model_lock:
            self._reference_count = max(0, self._reference_count - 1)
            if self._reference_count == 0:
                self._reset_cleanup_timer()

    def _reset_cleanup_timer(self):
        """Reset cleanup timer - cleanup model after 5 minutes of no use"""
        if self._cleanup_timer:
            self._cleanup_timer.cancel()

        def cleanup():
            with self._model_lock:
                if self._reference_count == 0 and self._model_pipeline:
                    self.logger.info("Cleaning up unused VMD model")
                    self._model_pipeline = None
                    self._load_event.clear()
                    gc.collect()

        self._cleanup_timer = threading.Timer(300, cleanup)  # 5 minutes
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()


# Global singleton instance
vmd_model_singleton = VoicemailModelManager()


class VoicemailClassifier:
    """
    Framework-agnostic voicemail classification using ML model

    Uses wav2vec transformer model to classify audio as voicemail or live person.
    Works with any SIP framework (pjsua2, FreeSwitch, etc.) - just needs audio bytes.
    """

    def __init__(self,
                 model_path: str = "models/voicemail_detector",
                 confidence_threshold: float = 0.60,
                 min_audio_length: int = 1600,
                 logger=None):
        """
        Initialize classifier

        Args:
            model_path: Path to model cache directory
            confidence_threshold: Minimum confidence for positive detection (0.0-1.0)
            min_audio_length: Minimum audio bytes required for analysis
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.min_audio_length = min_audio_length
        self.model_pipeline = None

        self.logger.info(
            f"VoicemailClassifier initialized "
            f"(threshold={confidence_threshold}, min_length={min_audio_length})"
        )

    def classify_audio(self, audio_bytes: bytes) -> Tuple[bool, float]:
        """
        Classify audio as voicemail or live person

        Args:
            audio_bytes: Raw PCM audio bytes (8kHz, 16-bit, mono)

        Returns:
            Tuple of (is_voicemail: bool, confidence: float)
            - is_voicemail: True if voicemail detected with high confidence
            - confidence: Model confidence score (0.0-1.0)
        """
        try:
            # === DIAGNOSTIC: Input validation ===
            audio_duration = len(audio_bytes) / (8000 * 2)  # 8kHz, 16-bit
            self.logger.info(
                f"[VMD CORE] classify_audio() called:\n"
                f"  Input size: {len(audio_bytes):,} bytes\n"
                f"  Audio duration: {audio_duration:.2f}s\n"
                f"  Minimum required: {self.min_audio_length:,} bytes\n"
                f"  Confidence threshold: {self.confidence_threshold}"
            )

            # Check minimum length
            if len(audio_bytes) < self.min_audio_length:
                self.logger.warning(
                    f"[VMD CORE] ❌ Insufficient audio data "
                    f"({len(audio_bytes):,} < {self.min_audio_length:,})"
                )
                return False, 0.0

            # === DIAGNOSTIC: Audio quality analysis ===
            try:
                import numpy as np
                audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_array.astype(float)**2))
                peak = np.max(np.abs(audio_array))

                self.logger.info(
                    f"[VMD CORE] Input audio quality:\n"
                    f"  Samples: {len(audio_array):,}\n"
                    f"  RMS energy: {rms:.1f}\n"
                    f"  Peak amplitude: {peak:.0f}"
                )
            except Exception as e:
                self.logger.warning(f"[VMD CORE] Could not analyze input audio: {e}")

            # Get model pipeline
            self.logger.info("[VMD CORE] Loading model pipeline...")
            self.model_pipeline = vmd_model_singleton.get_pipeline(self.model_path)
            if not self.model_pipeline:
                self.logger.error("[VMD CORE] ❌ Model unavailable - assuming live person")
                return False, 0.0

            self.logger.info("[VMD CORE] ✅ Model pipeline ready")

            # Prepare audio file for model
            self.logger.info("[VMD CORE] Preparing audio file (8kHz -> 16kHz upsampling)...")
            prep_start = time.time()
            temp_file = self._prepare_audio_file(audio_bytes)
            prep_time = time.time() - prep_start

            if not temp_file:
                self.logger.error("[VMD CORE] ❌ Audio preparation failed")
                return False, 0.0

            self.logger.info(f"[VMD CORE] ✅ Audio prepared in {prep_time:.2f}s: {temp_file}")

            try:
                # Run classification
                self.logger.info("[VMD CORE] Running transformer model inference...")
                inference_start = time.time()
                result = self.model_pipeline(temp_file)
                inference_time = time.time() - inference_start

                self.logger.info(f"[VMD CORE] ✅ Inference complete in {inference_time:.2f}s")

                # Process result
                if result and len(result) > 0:
                    label = result[0]["label"]
                    confidence = result[0]["score"]

                    self.logger.info(
                        f"[VMD CORE] Raw model output:\n"
                        f"  Label: {label}\n"
                        f"  Confidence: {confidence:.3f}"
                    )

                    # Interpret result
                    is_voicemail, final_confidence = self._interpret_result(label, confidence)

                    self.logger.info(
                        f"[VMD CORE] Final decision:\n"
                        f"  Classification: {'VOICEMAIL' if is_voicemail else 'LIVE PERSON'}\n"
                        f"  Confidence: {final_confidence:.3f}\n"
                        f"  Threshold: {self.confidence_threshold}\n"
                        f"  Total time: {prep_time + inference_time:.2f}s"
                    )

                    return is_voicemail, final_confidence
                else:
                    self.logger.error("[VMD CORE] ❌ No model result returned")
                    return False, 0.0

            finally:
                # Cleanup temp file
                try:
                    os.unlink(temp_file)
                    self.logger.debug(f"[VMD CORE] Cleaned up temp file: {temp_file}")
                except Exception as e:
                    self.logger.warning(f"[VMD CORE] Could not cleanup temp file: {e}")

        except Exception as e:
            self.logger.error(f"[VMD CORE] Classification error: {e}", exc_info=True)
            return False, 0.0

    def _prepare_audio_file(self, audio_data: bytes) -> Optional[str]:
        """
        Prepare audio file for model input

        Converts raw PCM audio (8kHz, 16-bit, mono) to WAV file at 16kHz
        Model expects 16kHz audio, so we upsample from 8kHz

        Args:
            audio_data: Raw PCM bytes (8kHz, 16-bit, mono)

        Returns:
            Path to temporary WAV file, or None on error
        """
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)

            # Upsample from 8kHz to 16kHz (simple repeat method)
            upsampled = np.repeat(audio_array, 2)

            # Create AudioSegment
            audio_segment = AudioSegment(
                data=upsampled.tobytes(),
                sample_width=2,
                frame_rate=16000,
                channels=1
            )

            # Export to temp file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.wav',
                prefix='vmd_'
            )

            audio_segment.export(temp_file.name, format="wav")
            temp_file.close()

            return temp_file.name

        except Exception as e:
            self.logger.error(f"Audio preparation error: {e}")
            return None

    def _interpret_result(self, label: str, confidence: float) -> Tuple[bool, float]:
        """
        Interpret model result

        Args:
            label: Model output label (contains keywords like "voicemail", "machine", etc.)
            confidence: Model confidence score (0.0-1.0)

        Returns:
            Tuple of (is_voicemail: bool, confidence: float)
        """
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            self.logger.info(
                f"Low confidence "
                f"({confidence:.3f} < {self.confidence_threshold})"
            )
            return False, confidence

        # Check for voicemail keywords
        label_lower = label.lower()
        voicemail_keywords = [
            "voicemail", "machine", "answering",
            "vm", "automated", "recording", "message"
        ]

        is_voicemail = any(keyword in label_lower for keyword in voicemail_keywords)

        if is_voicemail:
            self.logger.info(
                f"VOICEMAIL confirmed (label={label}, confidence={confidence:.3f})"
            )
        else:
            self.logger.info(
                f"LIVE PERSON confirmed (label={label}, confidence={confidence:.3f})"
            )

        return is_voicemail, confidence

    def cleanup(self):
        """Release model reference"""
        if self.model_pipeline:
            vmd_model_singleton.release()
            self.model_pipeline = None
