#!/usr/bin/env python3
"""
Parakeet RNNT 1.1b Model Wrapper with Native Confidence Support

This module provides a drop-in replacement for Parakeet TDT with proper confidence scores.
The RNNT architecture natively supports word_confidence, token_confidence, and frame_confidence,
eliminating the need for score-based workarounds.

Designed to integrate seamlessly with the existing SIP bot threading and singleton architecture.
"""

import os
import logging
import threading
import time
from typing import Optional, List, Tuple, Union
import numpy as np
import tempfile
import soundfile as sf
import torch

# Import NeMo components for RNNT
try:
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.parts.utils.asr_confidence_utils import (
        ConfidenceConfig, 
        ConfidenceMethodConfig
    )
    from nemo.collections.asr.parts.submodules.rnnt_decoding import RNNTDecodingConfig
    NEMO_AVAILABLE = True
except ImportError as e:
    NEMO_AVAILABLE = False
    print(f"NeMo not available: {e}")


class ParakeetRNNTModel:
    """
    Parakeet RNNT 1.1b wrapper with native confidence support.
    
    This class provides a drop-in replacement for Parakeet TDT with the same API
    but with working confidence scores from the RNNT architecture.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 device: str = 'auto',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize Parakeet RNNT 1.1b model
        
        Args:
            model_path: Path to store/load model (auto-creates if None)
            confidence_threshold: Minimum confidence for valid transcription
            device: 'auto', 'cuda', or 'cpu'
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.model_path = model_path or os.path.join("models", "parakeet-rnnt-1.1b")
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = None
        self._model_lock = threading.Lock()
        
        # Ensure model directory exists
        os.makedirs(self.model_path, exist_ok=True)
        
        # Set device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
                self.logger.info(f"Auto-selected CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                self.logger.info("Auto-selected CPU device")
        else:
            self.device = device
            self.logger.info(f"Using specified device: {device}")
    
    def _download_and_setup_model(self) -> bool:
        """
        Download and setup Parakeet RNNT 1.1b model with confidence configuration
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not NEMO_AVAILABLE:
                self.logger.error("NeMo not available. Install with: pip install nemo_toolkit")
                return False
            
            self.logger.info("Downloading Parakeet RNNT 1.1b model...")
            start_time = time.time()
            
            # Download model
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                "nvidia/parakeet-rnnt-1.1b"
            )
            
            download_time = time.time() - start_time
            self.logger.info(f"Model downloaded in {download_time:.2f} seconds")
            
            # Configure RNNT for confidence extraction
            confidence_cfg = ConfidenceConfig(
                preserve_frame_confidence=True,
                preserve_token_confidence=True, 
                preserve_word_confidence=True,
                aggregation="prod",  # Product aggregation works well for RNNT
                exclude_blank=False,  # RNNT benefits from including blanks
                tdt_include_duration=False,  # Not needed for RNNT
                method_cfg=ConfidenceMethodConfig(
                    name="max_prob",  # Maximum probability method
                    entropy_type="gibbs",
                    alpha=0.5,
                    entropy_norm="lin"
                )
            )
            
            # Apply RNNT-specific decoding configuration
            decoding_cfg = RNNTDecodingConfig(
                strategy="greedy_batch",
                fused_batch_size=-1,
                preserve_alignments=True,
                confidence_cfg=confidence_cfg,
                compute_timestamps=False  # Enable timestamps
            )
            
            # Update decoding strategy
            self.model.change_decoding_strategy(decoding_cfg)
            
            # Move to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Configure for multi-threading (same as TDT setup)
            if self.device == 'cuda':
                try:
                    # Disable graph optimization to prevent CUDAGraph replay errors
                    if hasattr(self.model, 'cfg'):
                        if hasattr(self.model.cfg, 'use_cuda_graph'):
                            self.model.cfg.use_cuda_graph = False
                        if hasattr(self.model.cfg, 'enable_cuda_graph'):
                            self.model.cfg.enable_cuda_graph = False
                    
                    # Set threading mode for concurrent access
                    torch.backends.cudnn.benchmark = False
                    torch.backends.cudnn.deterministic = True
                    
                    self.logger.info("CUDA graph optimization disabled for multi-threading")
                except Exception as e:
                    self.logger.warning(f"Could not disable CUDA graph optimization: {e}")
            
            self.logger.info("✅ Parakeet RNNT 1.1b loaded with native confidence support")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download/setup RNNT model: {e}")
            return False
    
    def ensure_model_loaded(self) -> bool:
        """
        Ensure model is loaded, download if necessary
        
        Returns:
            True if model is ready, False otherwise
        """
        if self.model is not None:
            return True
        
        with self._model_lock:
            # Double-check inside lock
            if self.model is not None:
                return True
            
            # Download and setup model
            return self._download_and_setup_model()
    
    def transcribe_with_confidence(self, 
                                   audio_path: str,
                                   return_hypotheses: bool = True,
                                   **kwargs) -> Tuple[Optional[str], float]:
        """
        Transcribe audio file with native RNNT confidence scores
        
        Args:
            audio_path: Path to audio file
            return_hypotheses: Whether to return hypothesis objects (required for confidence)
            **kwargs: Additional arguments (batch_size, num_workers, verbose, etc.) - passed to transcribe()
            
        Returns:
            Tuple of (transcribed_text, confidence_score)
        """
        if not self.ensure_model_loaded():
            return None, 0.0
        
        try:
            with self._model_lock:
                # Transcribe with hypotheses for confidence
                # Extract known parameters and set defaults
                batch_size = kwargs.get('batch_size', 1)
                num_workers = kwargs.get('num_workers', 0)
                verbose = kwargs.get('verbose', False)
                
                # Try with return_hypotheses first, fallback if needed
                try:
                    hypotheses = self.model.transcribe(
                        [audio_path],
                        batch_size=batch_size,
                        return_hypotheses=return_hypotheses,
                        num_workers=num_workers,
                        verbose=verbose
                    )
                except Exception as transcribe_error:
                    self.logger.warning(f"Transcribe with hypotheses failed: {transcribe_error}")
                    # Fallback without return_hypotheses
                    try:
                        hypotheses = self.model.transcribe(
                            [audio_path],
                            batch_size=batch_size,
                            return_hypotheses=False,
                            num_workers=num_workers,
                            verbose=verbose
                        )
                        # Convert string results to simple objects
                        if hypotheses and isinstance(hypotheses[0], str):
                            class SimpleHypothesis:
                                def __init__(self, text):
                                    self.text = text
                                    self.score = None
                                    self.word_confidence = None
                                    self.token_confidence = None
                                    self.frame_confidence = None
                            hypotheses = [SimpleHypothesis(hypotheses[0])]
                    except Exception as fallback_error:
                        self.logger.error(f"Fallback transcription failed: {fallback_error}")
                        return None, 0.0
                
                if not hypotheses:
                    return None, 0.0
                
                # Handle RNNT hypothesis structure
                hypothesis = hypotheses[0]
                if isinstance(hypothesis, list) and len(hypothesis) > 0:
                    hypothesis = hypothesis[0]
                
                if hypothesis:
                    # Extract text
                    text = hypothesis.text if hasattr(hypothesis, 'text') else str(hypothesis)
                    
                    # Get native RNNT confidence scores
                    try:
                        confidence = self._extract_native_confidence(hypothesis)
                    except Exception as conf_error:
                        self.logger.warning(f"Confidence extraction error: {conf_error}")
                        confidence = 0.5  # Default confidence
                    
                    # Apply confidence threshold
                    if confidence >= self.confidence_threshold:
                        return text, confidence
                    else:
                        self.logger.debug(f"Low confidence ({confidence:.3f}): {text}")
                        return None, confidence
                
        except Exception as e:
            self.logger.error(f"RNNT transcription error: {e}")
            
        return None, 0.0
    
    def _extract_native_confidence(self, hypothesis) -> float:
        """
        Extract native confidence from RNNT hypothesis with robust error handling
        
        RNNT natively supports word_confidence, token_confidence, and frame_confidence
        unlike TDT which requires score-based workarounds.
        
        Returns:
            Confidence score between 0 and 1
        """
        try:
            
            # Priority 1: Word confidence (most reliable for RNNT)
            if hasattr(hypothesis, 'word_confidence') and hypothesis.word_confidence is not None:
                try:
                    confidences = hypothesis.word_confidence
                    
                    # Handle dictionary structure (common in NeMo RNNT)
                    if isinstance(confidences, dict):
                        # Look for common keys in confidence dictionaries
                        if 'confidence' in confidences:
                            confidences = confidences['confidence']
                        elif 'word_confidence' in confidences:
                            confidences = confidences['word_confidence']
                        else:
                            self.logger.debug(f"Word confidence dict keys: {list(confidences.keys())}")
                            # Try to get first available value
                            if confidences:
                                confidences = list(confidences.values())[0]
                    
                    if isinstance(confidences, torch.Tensor):
                        confidences = confidences.cpu().numpy()
                    elif isinstance(confidences, list):
                        confidences = np.array(confidences)
                    
                    # Ensure we have a valid numpy array
                    if isinstance(confidences, np.ndarray) and confidences.size > 0:
                        # Handle nested arrays or lists
                        if len(confidences.shape) > 1:
                            confidences = confidences.flatten()
                        
                        # Filter out invalid values
                        valid_confidences = confidences[~np.isnan(confidences) & ~np.isinf(confidences)]
                        if len(valid_confidences) > 0:
                            avg_confidence = float(np.mean(valid_confidences))
                            self.logger.debug(f"Word confidences: {len(valid_confidences)} values, Avg: {avg_confidence:.3f}")
                            return avg_confidence
                except Exception as e:
                    self.logger.debug(f"Word confidence extraction failed: {e}")
            
            # Priority 2: Token confidence
            if hasattr(hypothesis, 'token_confidence') and hypothesis.token_confidence is not None:
                try:
                    confidences = hypothesis.token_confidence
                    if isinstance(confidences, torch.Tensor):
                        confidences = confidences.cpu().numpy()
                    elif isinstance(confidences, list):
                        confidences = np.array(confidences)
                    
                    if isinstance(confidences, np.ndarray) and confidences.size > 0:
                        # Handle potential multidimensional arrays
                        if len(confidences.shape) > 1:
                            confidences = confidences.flatten()
                        
                        valid_confidences = confidences[~np.isnan(confidences) & ~np.isinf(confidences)]
                        if len(valid_confidences) > 0:
                            avg_confidence = float(np.mean(valid_confidences))
                            self.logger.debug(f"Token confidence: {avg_confidence:.3f}")
                            return avg_confidence
                except Exception as e:
                    self.logger.debug(f"Token confidence extraction failed: {e}")
            
            # Priority 3: Frame confidence
            if hasattr(hypothesis, 'frame_confidence') and hypothesis.frame_confidence is not None:
                try:
                    confidences = hypothesis.frame_confidence
                    if isinstance(confidences, torch.Tensor):
                        confidences = confidences.cpu().numpy()
                    elif isinstance(confidences, list):
                        confidences = np.array(confidences)
                    
                    if isinstance(confidences, np.ndarray) and confidences.size > 0:
                        # RNNT frame confidence is 2D: [time_steps, num_classes]
                        if len(confidences.shape) == 2:
                            # Take max probability at each timestep, then average
                            max_probs = np.max(confidences, axis=-1)
                            valid_probs = max_probs[~np.isnan(max_probs) & ~np.isinf(max_probs)]
                            if len(valid_probs) > 0:
                                avg_confidence = float(np.mean(valid_probs))
                                self.logger.debug(f"Frame confidence: {avg_confidence:.3f}")
                                return avg_confidence
                        elif len(confidences.shape) == 1:
                            valid_confidences = confidences[~np.isnan(confidences) & ~np.isinf(confidences)]
                            if len(valid_confidences) > 0:
                                avg_confidence = float(np.mean(valid_confidences))
                                self.logger.debug(f"Frame confidence (1D): {avg_confidence:.3f}")
                                return avg_confidence
                except Exception as e:
                    self.logger.debug(f"Frame confidence extraction failed: {e}")
            
            # Fallback: Use score field (better normalized in RNNT than TDT)
            if hasattr(hypothesis, 'score') and hypothesis.score is not None:
                try:
                    score = hypothesis.score
                    if isinstance(score, torch.Tensor):
                        score = score.item()
                    else:
                        score = float(score)
                    
                    # RNNT scores are better normalized than TDT
                    # Convert log probability to confidence using sigmoid
                    confidence = 1 / (1 + np.exp(-score / 10))
                    self.logger.debug(f"Score-based confidence: {confidence:.3f} (score: {score})")
                    return confidence
                except Exception as e:
                    self.logger.debug(f"Score-based confidence extraction failed: {e}")
            
            # Final fallback
            self.logger.debug("No usable confidence information available from RNNT hypothesis")
            return 0.5
            
        except Exception as e:
            self.logger.warning(f"Confidence extraction error: {e}")
            return 0.5
    
    def transcribe(self, 
                   audio_paths: Union[str, List[str]], 
                   batch_size: int = 1,
                   return_hypotheses: bool = False,
                   **kwargs) -> List[Union[str, object]]:
        """
        Transcribe audio files (compatible with TDT API)
        
        Args:
            audio_paths: Path(s) to audio files
            batch_size: Batch size for processing
            return_hypotheses: Whether to return hypothesis objects
            **kwargs: Additional arguments
            
        Returns:
            List of transcription results
        """
        if not self.ensure_model_loaded():
            if isinstance(audio_paths, str):
                return [None]
            else:
                return [None] * len(audio_paths)
        
        try:
            with self._model_lock:
                # Ensure audio_paths is a list
                if isinstance(audio_paths, str):
                    audio_paths = [audio_paths]
                
                # Use model's transcribe method
                results = self.model.transcribe(
                    audio_paths,
                    batch_size=batch_size,
                    return_hypotheses=return_hypotheses,
                    num_workers=0,
                    verbose=False,
                    **kwargs
                )
                
                return results if results else []
                
        except Exception as e:
            self.logger.error(f"RNNT batch transcription error: {e}")
            return [None] * len(audio_paths)
    
    def to(self, device):
        """Move model to device (compatible with TDT API)"""
        if self.model is not None:
            self.device = device
            self.model = self.model.to(device)
        return self
    
    def eval(self):
        """Set model to evaluation mode (compatible with TDT API)"""
        if self.model is not None:
            self.model.eval()
        return self
    
    def freeze(self):
        """Freeze model parameters (compatible with TDT API)"""
        if self.model is not None and hasattr(self.model, 'freeze'):
            self.model.freeze()
        return self
    
    @property
    def cfg(self):
        """Access model configuration (compatible with TDT API)"""
        if self.model is not None:
            return self.model.cfg
        return None


def create_rnnt_model(model_path: Optional[str] = None,
                      confidence_threshold: float = 0.5,
                      device: str = 'auto',
                      logger: Optional[logging.Logger] = None) -> ParakeetRNNTModel:
    """
    Factory function to create Parakeet RNNT model
    
    Args:
        model_path: Path to store/load model
        confidence_threshold: Minimum confidence threshold
        device: Device to use ('auto', 'cuda', 'cpu')
        logger: Logger instance
        
    Returns:
        Configured ParakeetRNNTModel instance
    """
    return ParakeetRNNTModel(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device,
        logger=logger
    )


# Test function for validation
def test_rnnt_confidence():
    """Test RNNT confidence extraction with sample audio"""
    import tempfile
    import soundfile as sf
    import numpy as np
    
    # Create test model
    model = create_rnnt_model(confidence_threshold=0.3)
    
    # Generate test audio (1 second of sine wave)
    sample_rate = 16000
    duration = 1.0
    frequency = 440  # A4 note
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio_data = 0.3 * np.sin(2 * np.pi * frequency * t)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        sf.write(tmp_file.name, audio_data, sample_rate)
        tmp_path = tmp_file.name
    
    try:
        print("Testing Parakeet RNNT confidence extraction...")
        text, confidence = model.transcribe_with_confidence(tmp_path)
        print(f"Transcription: '{text}' (confidence: {confidence:.3f})")
        print("✅ RNNT confidence test completed!")
        
    except Exception as e:
        print(f"❌ RNNT test failed: {e}")
    finally:
        # Cleanup
        try:
            os.unlink(tmp_path)
        except:
            pass


if __name__ == "__main__":
    test_rnnt_confidence()