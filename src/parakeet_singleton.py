#!/usr/bin/env python3
"""
Parakeet Model Singleton
Thread-safe singleton for Parakeet ASR models (TDT and RNNT) with lazy loading
Extracted from deprecated sip_bot_server.py for standalone use
"""

import threading
import logging

# Setup logging
log = logging.getLogger(__name__)


class ParakeetModelSingleton:
    """Thread-safe singleton for Parakeet models (TDT and RNNT) with lazy loading"""
    _instance = None
    _lock = threading.Lock()
    _model = None
    _model_type = None  # Track which model type is loaded
    _model_lock = threading.Lock()
    _transcribe_lock = threading.Lock()  # Lock for thread-safe transcription
    _load_event = threading.Event()
    _load_error = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self, logger=None, confidence_threshold=None):
        """Get or create the Parakeet model instance (TDT or RNNT based on config)"""
        # Fast path - model already loaded
        if self._model is not None:
            return self._model

        with self._model_lock:
            # Double-check inside lock
            if self._model is not None:
                return self._model

            # Check if another thread is loading
            if not self._load_event.is_set():
                try:
                    import torch
                    from src.config import (
                        MODEL_PATH, USE_GPU, USE_LOCAL_MODEL,
                        USE_RNNT_MODEL, RNNT_MODEL_PATH,
                        RNNT_CONFIDENCE_THRESHOLD, RNNT_FALLBACK_TO_TDT
                    )

                    # Use provided logger or fallback to module logger
                    _logger = logger or log

                    # Set device
                    if USE_GPU and torch.cuda.is_available():
                        device = 'cuda'
                        _logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                    else:
                        device = 'cpu'
                        _logger.info("Using CPU for inference")

                    # Load RNNT model if enabled
                    if USE_RNNT_MODEL:
                        try:
                            from src.parakeet_rnnt import create_rnnt_model

                            _logger.info("Loading global Parakeet RNNT 1.1b model (singleton)...")

                            # Use provided threshold or fallback to config default
                            threshold = confidence_threshold if confidence_threshold is not None else RNNT_CONFIDENCE_THRESHOLD

                            self._model = create_rnnt_model(
                                model_path=RNNT_MODEL_PATH,
                                confidence_threshold=threshold,
                                device=device,
                                logger=logger or _logger
                            )

                            # Ensure model is ready
                            if self._model.ensure_model_loaded():
                                self._model_type = "RNNT"
                                _logger.info("✅ Global Parakeet RNNT 1.1b model loaded successfully")
                            else:
                                raise Exception("RNNT model failed to load")

                        except Exception as rnnt_error:
                            _logger.error(f"Failed to load RNNT model: {rnnt_error}")

                            if RNNT_FALLBACK_TO_TDT:
                                _logger.info("Falling back to TDT model...")
                                self._model = None  # Reset for TDT loading
                            else:
                                raise rnnt_error

                    # Load TDT model if RNNT not enabled or failed with fallback
                    if self._model is None:
                        import nemo.collections.asr as nemo_asr

                        _logger.info("Loading global Parakeet TDT model (singleton)...")

                        if USE_LOCAL_MODEL:
                            _logger.info(f"Loading local Parakeet model from {MODEL_PATH}")
                            self._model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH)
                        else:
                            self._model = nemo_asr.models.ASRModel.from_pretrained(
                                model_name="nvidia/parakeet-tdt-0.6b"
                            )

                        self._model = self._model.to(device)
                        self._model.eval()
                        self._model_type = "TDT"

                        # Disable CUDA graph optimization for multi-threaded inference
                        if device == 'cuda':
                            try:
                                # Disable graph optimization to prevent CUDAGraph replay errors
                                if hasattr(self._model, 'cfg'):
                                    if hasattr(self._model.cfg, 'use_cuda_graph'):
                                        self._model.cfg.use_cuda_graph = False
                                    if hasattr(self._model.cfg, 'enable_cuda_graph'):
                                        self._model.cfg.enable_cuda_graph = False

                                # Set threading mode for concurrent access
                                torch.backends.cudnn.benchmark = False
                                torch.backends.cudnn.deterministic = True

                                _logger.info("CUDA graph optimization disabled for multi-threading")
                            except Exception as e:
                                _logger.warning(f"Could not disable CUDA graph optimization: {e}")

                        _logger.info("✅ Global Parakeet TDT model loaded successfully")

                    self._load_event.set()

                except Exception as e:
                    self._load_error = str(e)
                    _logger.error(f"Failed to load Parakeet model: {e}")
                    return None

            return self._model

    def transcribe_safe(self, audio_paths, **kwargs):
        """Thread-safe wrapper for model transcription to prevent freeze/unfreeze errors"""
        if self._model is None:
            return None

        # Serialize access to the model to prevent concurrent freeze/unfreeze operations
        with self._transcribe_lock:
            try:
                return self._model.transcribe(audio_paths, **kwargs)
            except Exception as e:
                # Log the error but don't re-raise to allow caller to handle
                log.error(f"Parakeet ({self._model_type}) transcription error: {e}")
                return None

    def transcribe_with_confidence(self, audio_path, **kwargs):
        """
        Transcribe with confidence score (RNNT native or TDT fallback)
        Returns (text, confidence) tuple
        """
        if self._model is None:
            return None, 0.0

        # Use RNNT native confidence if available
        if self._model_type == "RNNT" and hasattr(self._model, 'transcribe_with_confidence'):
            with self._transcribe_lock:
                try:
                    return self._model.transcribe_with_confidence(audio_path, **kwargs)
                except Exception as e:
                    log.error(f"RNNT confidence transcription error: {e}")
                    return None, 0.0

        # Fallback to regular transcription for TDT or RNNT without confidence method
        else:
            with self._transcribe_lock:
                try:
                    # Regular transcription
                    hypotheses = self._model.transcribe(
                        [audio_path] if isinstance(audio_path, str) else audio_path,
                        return_hypotheses=True,
                        **kwargs
                    )

                    if hypotheses and len(hypotheses) > 0:
                        hypothesis = hypotheses[0]
                        if isinstance(hypothesis, list) and len(hypothesis) > 0:
                            hypothesis = hypothesis[0]

                        if hypothesis:
                            text = hypothesis.text if hasattr(hypothesis, 'text') else str(hypothesis)

                            # Extract confidence score (TDT method)
                            confidence = 0.0
                            if hasattr(hypothesis, 'score'):
                                confidence = hypothesis.score
                            elif hasattr(hypothesis, 'confidence'):
                                confidence = hypothesis.confidence

                            return text, confidence

                except Exception as e:
                    log.error(f"Parakeet ({self._model_type}) confidence fallback error: {e}")

        return None, 0.0

    def get_model_type(self):
        """Get the type of loaded model (TDT or RNNT)"""
        return self._model_type

    def is_loaded(self):
        """Check if model is loaded"""
        return self._model is not None
