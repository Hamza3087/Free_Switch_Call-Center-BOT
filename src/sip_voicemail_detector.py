#!/usr/bin/env python3
"""
Production-Grade Voicemail Detector
- Thread-safe singleton model management
- Robust error handling for all operations
- Zero resource leaks with automatic cleanup
"""

import os
import time
import tempfile
import threading
import numpy as np
from typing import Optional, Callable, List
from pydub import AudioSegment
import torch
import logging
import gc
import weakref
from contextlib import contextmanager
import queue
import pjsua2 as pj

# PJSIP Error codes
PJSIP_ERROR_NOT_FOUND = 70001  # PJ_ENOTFOUND
PJSIP_ERROR_INVALID = 70004     # PJ_EINVAL

class ModelSingleton:
    """Thread-safe singleton for transformer model with lazy loading"""
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
        """Get or create the model pipeline with reference counting"""
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
vmd_model_singleton = ModelSingleton()

class LocalVoicemailDetector:
    """
    Production-grade voicemail detector with robust error handling
    """
    
    def __init__(self, detection_duration: float = 7.0, 
                 model_path: str = "models/voicemail_detector", 
                 logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.detection_duration = detection_duration
        self.model_path = model_path
        
        # State management
        self.state_lock = threading.RLock()
        self.is_active = False
        self.detection_complete = False
        
        # Detection thread
        self.detection_thread = None
        self.stop_event = threading.Event()
        
        # Callbacks
        self.voicemail_callback = None
        self.callback_lock = threading.Lock()
        
        # Audio recording
        self.audio_recorder = None
        self.recorded_audio_data = []
        self.audio_lock = threading.Lock()
        
        # Model
        self.model_pipeline = None
        
        # Stats
        self.stats = {
            'detections_started': 0,
            'detections_completed': 0,
            'voicemails_detected': 0,
            'errors': 0,
            'confidence': 0.0,
            'last_label': None
        }
        
        # Thresholds
        self.confidence_threshold = 0.60
        self.min_audio_length = 1600  # Minimum bytes for analysis
        
        self.logger.info(f"VMD initialized (duration={detection_duration}s, threshold={self.confidence_threshold})")
    
    def set_voicemail_callback(self, callback: Callable):
        """Set callback for voicemail detection"""
        with self.callback_lock:
            self.voicemail_callback = callback
    
    def set_audio_recorder(self, audio_recorder):
        """Set audio recorder instance"""
        self.audio_recorder = audio_recorder
    
    def start_detection(self, audio_media) -> bool:
        """
        Start voicemail detection with error handling
        Returns True if started successfully
        """
        with self.state_lock:
            if self.is_active:
                self.logger.warning("Detection already active")
                return False
            
            if not self.audio_recorder:
                self.logger.error("No audio recorder set")
                return False
            
            self.logger.info("Starting voicemail detection...")
            
            # Reset state
            self.stop_event.clear()
            self.detection_complete = False
            
            with self.audio_lock:
                self.recorded_audio_data.clear()
            
            # Start detection thread
            self.detection_thread = threading.Thread(
                target=self._detection_loop_safe,
                args=(audio_media,),
                name=f"vmd_{id(self)}",
                daemon=True
            )
            self.detection_thread.start()
            
            self.is_active = True
            self.stats['detections_started'] += 1
            
            return True
    
    def stop_detection(self):
        """Stop detection with cleanup"""
        with self.state_lock:
            if not self.is_active:
                return
            
            self.logger.info("Stopping voicemail detection...")
            
            # Signal stop
            self.stop_event.set()
            self.is_active = False
        
        # Wait for thread outside lock
        if self.detection_thread and self.detection_thread.is_alive():
            if threading.current_thread() != self.detection_thread:
                self.detection_thread.join(timeout=2.0)
        
        # Release model reference
        if self.model_pipeline:
            vmd_model_singleton.release()
            self.model_pipeline = None
        
        self.logger.info("VMD stopped")
    
    def _detection_loop_safe(self, audio_media):
        """Detection loop with comprehensive error handling"""
        try:
            # Register thread with PJSIP
            self._register_pjsip_thread()
            
            # Run detection
            self._run_detection(audio_media)
            
        except Exception as e:
            self.logger.error(f"VMD detection error: {e}", exc_info=True)
            self.stats['errors'] += 1
            
        finally:
            with self.state_lock:
                self.is_active = False
                self.detection_complete = True
            
            self.stats['detections_completed'] += 1
    
    def _register_pjsip_thread(self):
        """Register thread with PJSIP"""
        try:
            if not hasattr(threading.current_thread(), '_pj_registered'):
                pj.Endpoint.instance().libRegisterThread("vmd_thread")
                threading.current_thread()._pj_registered = True
                
        except Exception as e:
            self.logger.warning(f"PJSIP registration warning: {e}")
    
    def _run_detection(self, audio_media):
        """Main detection logic"""
        recording_file = None
        
        try:
            # Start recording
            recording_file = self.audio_recorder.start_recording(audio_media)
            if not recording_file:
                self.logger.error("Failed to start recording for VMD")
                return
            
            self.logger.info("VMD recording started...")
            
            # Record for specified duration
            detection_start = time.time()
            last_file_size = 0
            
            while (time.time() - detection_start) < self.detection_duration:
                # Check if stopped
                if self.stop_event.is_set():
                    self.logger.info("VMD stopped by signal")
                    break
                
                # Get new audio data
                new_audio = self.audio_recorder.get_new_audio_data(last_file_size)
                if new_audio:
                    with self.audio_lock:
                        self.recorded_audio_data.append(new_audio)
                    
                    # Update file size
                    if recording_file and os.path.exists(recording_file):
                        last_file_size = os.path.getsize(recording_file)
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.1)
            
        except Exception as e:
            self.logger.error(f"Detection recording error: {e}")
            
        finally:
            # Stop recording with proper error handling
            try:
                self.audio_recorder.stop_recording()
            except Exception as e:
                self.logger.error(f"Failed to stop recording: {e}")
        
        # Analyze audio if not stopped
        if not self.stop_event.is_set():
            self.logger.info("Analyzing audio for voicemail patterns...")
            
            if self._classify_audio():
                self.logger.info("🤖 Voicemail detected")
                self.stats['voicemails_detected'] += 1
                self._trigger_callback()
            else:
                self.logger.info("👤 Live person detected")
    
    def _classify_audio(self) -> bool:
        """
        Classify audio as voicemail or live person
        Returns True if voicemail detected
        """
        try:
            # Get model pipeline
            self.model_pipeline = vmd_model_singleton.get_pipeline(self.model_path)
            if not self.model_pipeline:
                self.logger.warning("Model unavailable - assuming live person")
                return False
            
            # Combine audio data
            with self.audio_lock:
                if not self.recorded_audio_data:
                    self.logger.info("No audio data to analyze")
                    return False
                
                combined_audio = b''.join(self.recorded_audio_data)
            
            # Check minimum length
            if len(combined_audio) < self.min_audio_length:
                self.logger.info("Insufficient audio data")
                return False
            
            # Prepare audio for model
            temp_file = self._prepare_audio_file(combined_audio)
            if not temp_file:
                return False
            
            try:
                # Run classification
                self.logger.info("Running model inference...")
                result = self.model_pipeline(temp_file)
                
                # Process result
                if result and len(result) > 0:
                    label = result[0]["label"]
                    confidence = result[0]["score"]

                    # Store in stats for reporting
                    self.stats['confidence'] = confidence
                    self.stats['last_label'] = label

                    self.logger.info(f"Model result: {label} (confidence: {confidence:.3f})")

                    return self._interpret_result(label, confidence)
                else:
                    self.logger.warning("No model result")
                    return False
                    
            finally:
                # Cleanup temp file
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Classification error: {e}")
            self.stats['errors'] += 1
            return False
    
    def _prepare_audio_file(self, audio_data: bytes) -> Optional[str]:
        """Prepare audio file for model input"""
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Upsample from 8kHz to 16kHz
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
    
    def _interpret_result(self, label: str, confidence: float) -> bool:
        """
        Interpret model result
        Returns True if voicemail detected with high confidence
        """
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            self.logger.info(f"Low confidence ({confidence:.3f} < {self.confidence_threshold})")
            return False
        
        # Check for voicemail keywords
        label_lower = label.lower()
        voicemail_keywords = [
            "voicemail", "machine", "answering", 
            "vm", "automated", "recording", "message"
        ]
        
        is_voicemail = any(keyword in label_lower for keyword in voicemail_keywords)
        
        if is_voicemail:
            self.logger.info(f"VOICEMAIL confirmed (label={label}, confidence={confidence:.3f})")
        else:
            self.logger.info(f"LIVE PERSON confirmed (label={label}, confidence={confidence:.3f})")
        
        return is_voicemail
    
    def _trigger_callback(self):
        """Trigger voicemail detection callback"""
        with self.callback_lock:
            if self.voicemail_callback:
                try:
                    self.logger.info("Triggering voicemail callback...")
                    self.voicemail_callback()
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")
                    self.stats['errors'] += 1
    
    def get_stats(self):
        """Get detector statistics"""
        with self.state_lock:
            return self.stats.copy()
    
    def is_detection_complete(self):
        """Check if detection is complete"""
        with self.state_lock:
            return self.detection_complete