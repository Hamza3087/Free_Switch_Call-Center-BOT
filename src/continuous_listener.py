#!/usr/bin/env python3
"""
Continuous Listening System for SIP Call Center Bot

Implements true continuous listening throughout the entire call, recording everything 
from start to finish. Monitors for DNC/NI intents in real-time even during audio playback.
Based on the proven ringing detector architecture with dual recording system.
"""

import threading
import time
import os
import logging
import numpy as np
import tempfile
from typing import Optional, List, Callable
import pjsua2 as pj
from src.sip_audio_manager import SIPAudioRecorder

# Audio processing constants (synchronized with main flow)
ENERGY_THRESHOLD = 0.045  # Energy threshold for voice detection (increased sensitivity)
MIN_SPEECH_DURATION = 0.08  # Minimum speech duration for intent detection (reduced for faster response)
MIN_AUDIO_LENGTH = 0.15  # Minimum audio length for processing (reduced for better detection) 
SILENCE_DURATION_THRESHOLD = 1  # Reduced from 0.5 to 0.2 for faster response to continuous speech
PLAYBACK_ENERGY_THRESHOLD =  0.045  # Higher threshold during playback to filter bot audio

class ContinuousListener:
    """Continuous listener with its own independent recording stream"""
    
    def __init__(self, logger, intent_detector, parakeet_model=None, energy_threshold=None, rnnt_confidence_threshold=None):
        self.logger = logger or logging.getLogger(__name__)
        self.intent_detector = intent_detector
        self.parakeet_model = parakeet_model

        # Use provided thresholds or fallback to global constants
        self.energy_threshold = energy_threshold if energy_threshold is not None else ENERGY_THRESHOLD
        self.playback_energy_threshold = self.energy_threshold  # Use same threshold for playback

        # Create Silero VAD instance using same threshold from database
        from src.silero_vad import SileroVAD
        self.silero_vad = SileroVAD(
            threshold=self.energy_threshold,  # Direct passthrough from database
            sample_rate=8000,
            logger=logger
        )

        # RNNT confidence threshold
        from src.config import RNNT_LOW_CONFIDENCE_THRESHOLD
        self.rnnt_confidence_threshold = rnnt_confidence_threshold if rnnt_confidence_threshold is not None else RNNT_LOW_CONFIDENCE_THRESHOLD
        
        # State management
        self.state_lock = threading.Lock()
        self.is_active = False
        self.detection_complete = False
        self.stop_event = threading.Event()
        
        # Detection thread
        self.detection_thread = None
        
        # Audio recording (independent from main flow)
        self.audio_recorder = SIPAudioRecorder(self.logger)
        self.audio_lock = threading.Lock()
        self.recorded_audio_data = []
        self.audio_timestamps = []
        
        # Playback tracking for response extraction
        self.playback_periods = []  # List of (start_time, end_time) tuples
        self.current_playback_start = None
        self.main_flow_listening = False  # Flag to prevent interference during response collection
        
        # Intent detection state
        self.dnc_triggered = False
        self.ni_triggered = False  
        self.obscenity_triggered = False
        self.hp_triggered = False  # Hold/Press keywords
        self.clbk_triggered = False  # Callback requests
        self.callback_lock = threading.Lock()
        
        # Audio processing buffer
        self.audio_buffer = bytearray()
        self.buffer_timestamps = []
        
        # Transcription storage for main flow access
        self.transcriptions = []  # List of (timestamp, text, confidence) tuples
        self.transcription_lock = threading.Lock()
        
        # Speech-silence pattern detection (same as sequential AudioBuffer)
        self.voice_detected = False
        self.speech_start = None
        self.silence_start = None
        self.speech_silence_lock = threading.Lock()
        
        # Intent detection configuration
        from src.config import (
            INTENT_DETECTION_INTERVAL, INTENT_WINDOW_SIZE,
            MIN_SPEECH_DURATION_FOR_INTENT
        )
        self.detection_interval = INTENT_DETECTION_INTERVAL
        self.window_size = INTENT_WINDOW_SIZE
        self.min_speech_duration = MIN_SPEECH_DURATION_FOR_INTENT
        
        # Statistics
        self.stats = {
            'detections_started': 0,
            'chunks_processed': 0,
            'intent_checks': 0,
            'dnc_detected': 0,
            'ni_detected': 0,
            'obscenity_detected': 0,
            'hp_detected': 0,
            'clbk_detected': 0
        }
        
        self.logger.info("ContinuousListener initialized with independent recording")

    def attach_to_running_recorder(self, running_recorder, recording_file_path):
        """
        Attach to an already-running SIPAudioRecorder (warm start mode)
        This eliminates initialization delay and prevents audio loss under load

        Args:
            running_recorder: Already-started SIPAudioRecorder instance
            recording_file_path: Path to the recording file that's already capturing audio
        """
        self.logger.info(f"🔥 WARM START: Attaching to pre-started recorder: {recording_file_path}")

        # Replace our audio_recorder with the running one
        self.audio_recorder = running_recorder

        # Mark as warm-started so _run_continuous_detection skips recorder initialization
        self._warm_start = True
        self._warm_start_file = recording_file_path

        self.logger.info("✅ Warm start configured - recorder already capturing audio (zero loss)")

    def _is_valid_speech(self, audio_bytes):
        """Check if audio contains valid speech using Silero VAD only"""
        if not audio_bytes or len(audio_bytes) < 1600:  # Less than 100ms at 16kHz
            return False

        try:
            # Check duration
            duration = len(audio_bytes) / (8000 * 2)  # 8kHz, 16-bit (SIP audio is 8kHz)
            min_duration = MIN_SPEECH_DURATION if self.current_playback_start else MIN_AUDIO_LENGTH

            if duration < min_duration:
                return False

            # Use Silero VAD to determine if speech (no energy threshold)
            if not self.silero_vad.is_speech(audio_bytes):
                self.logger.debug(f"Noise rejected by Silero")
                return False

            return True

        except Exception as e:
            self.logger.debug(f"Speech validation error: {e}")
            return False
    
    def set_intent_callback(self, callback: Callable):
        """Set callback for when intents are detected"""
        self.intent_callback = callback
    
    def start_detection(self, audio_media) -> bool:
        """
        Start continuous listening and intent detection
        Returns True if started successfully
        """
        with self.state_lock:
            if self.is_active:
                self.logger.warning("Continuous listening already active")
                return False
            
            self.logger.info("Starting continuous listening and intent detection...")
            
            # Reset state
            self.stop_event.clear()
            self.detection_complete = False
            self.dnc_triggered = False
            self.ni_triggered = False
            self.obscenity_triggered = False
            
            with self.audio_lock:
                self.recorded_audio_data.clear()
                self.audio_timestamps.clear()
                self.playback_periods.clear()
                self.current_playback_start = None
                self.audio_buffer.clear()
                self.buffer_timestamps.clear()
                
            with self.transcription_lock:
                self.transcriptions.clear()
                
            with self.speech_silence_lock:
                self.voice_detected = False
                self.speech_start = None
                self.silence_start = None
            
            # Start detection thread
            self.detection_thread = threading.Thread(
                target=self._detection_loop_safe,
                args=(audio_media,),
                name=f"continuous_listener_{id(self)}",
                daemon=True
            )
            self.detection_thread.start()
            
            self.is_active = True
            self.stats['detections_started'] += 1
            
            return True
    
    def stop_detection(self):
        """Stop continuous listening"""
        with self.state_lock:
            if not self.is_active:
                return
            
            self.logger.info("Stopping continuous listening...")
            self.is_active = False
            self.stop_event.set()
        
        # Wait for thread to finish
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)
        
        # Stop recording
        try:
            self.audio_recorder.stop_recording()
        except Exception as e:
            self.logger.error(f"Error stopping continuous recording: {e}")
        
        self.logger.info("Continuous listening stopped")
    
    def _detection_loop_safe(self, audio_media):
        """Safe wrapper for detection loop"""
        try:
            self._register_pjsip_thread()
            self._run_continuous_detection(audio_media)
        except Exception as e:
            self.logger.error(f"Continuous detection error: {e}", exc_info=True)
        finally:
            self.detection_complete = True
    
    def _register_pjsip_thread(self):
        """Register current thread with PJSIP"""
        try:
            pj.Endpoint.instance().libRegisterThread(f"continuous_listener_{id(self)}")
        except Exception as e:
            self.logger.warning(f"PJSIP thread registration failed: {e}")
    
    def _run_continuous_detection(self, audio_media):
        """Main continuous detection logic - runs for entire call"""
        recording_file = None

        try:
            # Check if we're in warm-start mode (recorder already running)
            if hasattr(self, '_warm_start') and self._warm_start:
                self.logger.info("🔥 WARM START MODE: Using pre-started recorder")
                recording_file = self._warm_start_file

                # Verify recorder is actually recording
                if not self.audio_recorder.is_recording:
                    self.logger.error("❌ Warm start failed - recorder not active, falling back to cold start")
                    # Fall through to cold start
                else:
                    self.logger.info(f"✅ Warm start verified - recorder active: {recording_file}")

                    # Get current file size to start reading from current position
                    if os.path.exists(recording_file):
                        initial_file_size = os.path.getsize(recording_file)
                        self.logger.info(f"Starting from file position: {initial_file_size} bytes (skipping WAV header)")
                    else:
                        self.logger.warning("Recording file doesn't exist yet, starting from 0")
                        initial_file_size = 0
            else:
                initial_file_size = None  # Will trigger cold start

            # Cold start: Start our own independent recording
            if recording_file is None or initial_file_size is None:
                self.logger.info("Cold start: Creating new recorder")
                recording_file = self.audio_recorder.start_recording(audio_media)
                if not recording_file:
                    self.logger.error("Failed to start continuous recording")
                    return

                self.logger.info(f"Continuous recording started: {recording_file}")
                initial_file_size = 0

            # Log file format info when it becomes available
            if os.path.exists(recording_file):
                file_size = os.path.getsize(recording_file)
                self.logger.info(f"Continuous recording file size: {file_size} bytes")
            
            # Start continuous detection loop immediately - no waiting
            # Use initial_file_size for warm start (skip already-written audio)
            last_file_size = initial_file_size if initial_file_size else 0

            self.logger.info(f"Starting detection loop from file position: {last_file_size} bytes")

            while not self.stop_event.is_set():
                # Get new audio data
                try:
                    new_audio = self.audio_recorder.get_new_audio_data(last_file_size)
                except Exception as e:
                    self.logger.warning(f"Failed to get audio data: {e}")
                    break
                
                if new_audio:
                    current_time = time.time()
                    
                    # Store audio with timestamp
                    with self.audio_lock:
                        self.recorded_audio_data.append(new_audio)
                        self.audio_timestamps.append(current_time)
                        self.audio_buffer.extend(new_audio)
                        self.buffer_timestamps.append(current_time)
                        
                        # Keep buffer manageable (last 120 seconds - increased for better transcription capture)
                        max_buffer_duration = 120.0
                        while (self.buffer_timestamps and 
                               current_time - self.buffer_timestamps[0] > max_buffer_duration):
                            # Remove old data
                            bytes_per_chunk = len(new_audio)  # Approximate
                            if len(self.audio_buffer) > bytes_per_chunk:
                                self.audio_buffer = self.audio_buffer[bytes_per_chunk:]
                                self.buffer_timestamps.pop(0)
                            else:
                                break
                    
                    # Update file size
                    if recording_file and os.path.exists(recording_file):
                        last_file_size = os.path.getsize(recording_file)
                    
                    self.stats['chunks_processed'] += 1
                    
                    # Only check for DNC/NI intents during playback
                    # The main flow will handle response processing
                    if self.current_playback_start:
                        # During playback, check for speech using Silero VAD (interruptions)
                        try:
                            if self.silero_vad.is_speech(new_audio):
                                # Process recent audio for DNC/NI detection only
                                with self.audio_lock:
                                    if len(self.audio_buffer) > 16000:  # At least 1 second
                                        recent_audio = bytes(self.audio_buffer[-32000:])  # Last 2 seconds
                                        self._check_for_intents_on_audio(recent_audio)
                        except:
                            pass
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.03)
                
        except Exception as e:
            self.logger.error(f"Continuous detection error: {e}")
        finally:
            # Stop recording
            try:
                self.audio_recorder.stop_recording()
            except Exception as e:
                self.logger.error(f"Failed to stop continuous recording: {e}")
    
    def _update_speech_silence_detection(self, audio_chunk):
        """Update speech/silence detection state using Silero VAD"""
        try:
            with self.speech_silence_lock:
                current_time = time.time()

                # DURING PLAYBACK: Process human speech immediately (no waiting for silence!)
                if self.current_playback_start:
                    if self.silero_vad.is_speech(audio_chunk):  # Human is speaking during playback
                        self.logger.debug(f"Human speech detected during playback (Silero VAD)")
                        return True  # Process immediately, don't wait for silence
                    return False

                # NOT PLAYING: Use normal VAD (speech + silence pattern)
                # Voice activity detection using Silero
                if self.silero_vad.is_speech(audio_chunk):
                    if not self.voice_detected:
                        self.speech_start = current_time
                        self.logger.debug(f"Speech started at {current_time}")
                    self.voice_detected = True
                    self.silence_start = None
                else:
                    # Not speech (silence or noise)
                    if self.voice_detected:
                        if self.silence_start is None:
                            self.silence_start = current_time
                            self.logger.debug(f"Silence started at {current_time}")
                
                # Check if we should process (same logic as sequential)
                if (self.voice_detected and self.silence_start and self.speech_start):
                    silence_duration = current_time - self.silence_start
                    speech_duration = self.silence_start - self.speech_start
                    
                    if (silence_duration >= SILENCE_DURATION_THRESHOLD and 
                        speech_duration >= MIN_SPEECH_DURATION):
                        # Reset for next detection
                        self.voice_detected = False
                        self.speech_start = None
                        self.silence_start = None
                        return True
                
                return False
                
        except Exception as e:
            self.logger.debug(f"Speech-silence detection error: {e}")
            return False
    
    def _process_accumulated_audio(self):
        """Process accumulated audio when speech-silence pattern is detected"""
        # Get recent audio for processing (last 3 seconds to include the speech)
        with self.audio_lock:
            if not self.audio_buffer or not self.buffer_timestamps:
                return
            
            current_time = time.time()
            window_start = current_time - 3.0  # 3 second window
            
            # Find audio in the window
            window_audio = bytearray()
            for i, timestamp in enumerate(self.buffer_timestamps):
                if timestamp >= window_start:
                    # Approximate audio chunk size
                    chunk_size = len(self.audio_buffer) // len(self.buffer_timestamps)
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, len(self.audio_buffer))
                    if start_idx < len(self.audio_buffer):
                        window_audio.extend(self.audio_buffer[start_idx:end_idx])
        
        # Apply final validation before processing
        if not self._is_valid_speech(bytes(window_audio)):
            return
        
        # Process the audio
        self._check_for_intents_on_audio(bytes(window_audio))
    
    def _check_for_intents_on_audio(self, audio_bytes):
        """Check audio for negative intents after speech-silence pattern detected"""
        if not self.intent_detector:
            return
        
        # Convert to text using Parakeet
        try:
            result = self._process_audio_chunk(audio_bytes)
            if result:
                # Extract text and confidence if tuple
                if isinstance(result, tuple):
                    text, confidence = result
                else:
                    text = result
                    confidence = 0.0
                
                if text and len(text.strip()) > 0:
                    # Transcription is already stored in _process_audio_chunk - no need to store again
                    
                    # Log that we're checking this transcription for intents
                    if self.current_playback_start:
                        self.logger.debug(f"🔍 INTENT CHECK (during playback): '{text}' (confidence: {confidence:.2f})")
                    else:
                        self.logger.debug(f"🔍 INTENT CHECK: '{text}' (confidence: {confidence:.2f})")
                    
                    # Check for intents
                    self.intent_detector.detect_intent(text)
                    self.stats['intent_checks'] += 1
                
                # Check if negative intents were detected
                if hasattr(self.intent_detector, 'negative_detected') and self.intent_detector.negative_detected:
                    intent_type = getattr(self.intent_detector, 'negative_intent', 'unknown')
                    self.logger.warning(f"🚫 NEGATIVE INTENT DETECTED: {intent_type} - '{text}'")
                    
                    # Set appropriate flags
                    if 'do_not_call' in intent_type.lower() or 'dnc' in intent_type.lower() or 'do not call' in text.lower():
                        self.dnc_triggered = True
                        self.stats['dnc_detected'] += 1
                        self.logger.warning(f"🚫 DNC TRIGGERED by continuous listening!")
                        # Set the intent detector's negative_detected flag to ensure main flow sees it
                        self.intent_detector.negative_detected = True
                        self.intent_detector.negative_intent = 'do_not_call'
                    elif 'obscenity' in intent_type.lower():
                        self.obscenity_triggered = True
                        self.stats['obscenity_detected'] += 1
                        self.logger.warning(f"🚫 OBSCENITY TRIGGERED by continuous listening!")
                        # Set the intent detector's negative_detected flag to ensure main flow sees it
                        self.intent_detector.negative_detected = True
                        self.intent_detector.negative_intent = 'do_not_call'  # Map obscenity to DNC
                    elif 'not_interested' in intent_type.lower() or 'not interested' in intent_type.lower() or 'ni' in intent_type.lower():
                        self.ni_triggered = True
                        self.stats['ni_detected'] += 1
                        self.logger.warning(f"🚫 NI TRIGGERED by continuous listening!")
                        # Set the intent detector's negative_detected flag to ensure main flow sees it
                        self.intent_detector.negative_detected = True
                        self.intent_detector.negative_intent = 'not_interested'
                    elif 'hold_press' in intent_type.lower():
                        self.hp_triggered = True
                        self.stats['hp_detected'] += 1
                        self.logger.warning(f"🚫 HP TRIGGERED by continuous listening!")
                        # Set the intent detector's negative_detected flag to ensure main flow sees it
                        self.intent_detector.negative_detected = True
                        self.intent_detector.negative_intent = 'hold_press'
                    elif 'callback' in intent_type.lower():
                        self.clbk_triggered = True
                        self.stats['clbk_detected'] += 1
                        self.logger.warning(f"🚫 CLBK TRIGGERED by continuous listening!")
                        # Set the intent detector's negative_detected flag to ensure main flow sees it
                        self.intent_detector.negative_detected = True
                        self.intent_detector.negative_intent = 'callback'
                    else:
                        # Fallback for unknown negative intents - treat as DNC
                        self.obscenity_triggered = True
                        self.stats['obscenity_detected'] += 1
                        self.logger.warning(f"🚫 UNKNOWN NEGATIVE INTENT TRIGGERED: {intent_type}")
                        # Set the intent detector's negative_detected flag to ensure main flow sees it
                        self.intent_detector.negative_detected = True
                        self.intent_detector.negative_intent = 'do_not_call'
                    
                    # Trigger callback if set
                    if hasattr(self, 'intent_callback'):
                        try:
                            self.intent_callback(intent_type, text)
                        except Exception as e:
                            self.logger.error(f"Intent callback error: {e}")
                
        except Exception as e:
            self.logger.debug(f"Intent checking error: {e}")
    
    def _process_audio_chunk(self, audio_bytes):
        """Process audio chunk for intent detection using Parakeet"""
        try:
            if not self.parakeet_model or not audio_bytes or len(audio_bytes) < 1600:
                return None
            
            # Use the same Parakeet processing as the main bot
            import torch
            import tempfile
            import soundfile as sf
            import numpy as np
            import scipy.signal as scipy_signal
            import torchaudio
            
            # Convert to numpy array - use float64 for processing (same as main flow)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float64)
            
            # Normalize to [-1, 1]  
            audio_array = audio_array / 32768.0
            
            # Apply pre-emphasis filter to boost high frequencies
            pre_emphasis = 0.97
            audio_array = np.append(audio_array[0], audio_array[1:] - pre_emphasis * audio_array[:-1])
            
            # Apply bandpass filter (300Hz - 3400Hz for speech)  
            nyquist = 4000  # Half of 8kHz sample rate
            low = 300 / nyquist
            high = 3400 / nyquist
            b, a = scipy_signal.butter(5, [low, high], btype='band')
            audio_array = scipy_signal.filtfilt(b, a, audio_array)
            
            # Normalize audio level
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array * (0.9 / max_val)
            
            # Convert to tensor with float64 dtype
            audio_tensor = torch.from_numpy(audio_array).to(torch.float64).unsqueeze(0)
            
            # Resample from 8kHz to 16kHz with better quality
            audio_tensor_32 = audio_tensor.to(torch.float32)
            resampler = torchaudio.transforms.Resample(
                orig_freq=8000,
                new_freq=16000,
                resampling_method='sinc_interp_hann'
            )
            audio_16k = resampler(audio_tensor_32)
            
            # Save to temporary file for Parakeet
            audio_16k_save = audio_16k.squeeze().numpy().astype(np.float32)
            
            # Normalize again
            max_val = np.max(np.abs(audio_16k_save))
            if max_val > 0:
                audio_16k_save = audio_16k_save * (0.95 / max_val)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(
                    tmp_file.name,
                    audio_16k_save,
                    16000,
                    subtype='PCM_16'
                )
                tmp_path = tmp_file.name
            
            try:
                # Run inference with Parakeet using singleton's transcribe_with_confidence
                with torch.no_grad():
                    # Try to get the singleton from parent bot
                    parakeet_singleton = None
                    if hasattr(self, 'parakeet_singleton'):
                        parakeet_singleton = self.parakeet_singleton
                    elif hasattr(self.parakeet_model, '_singleton'):  # Try to get singleton reference
                        parakeet_singleton = self.parakeet_model._singleton
                    
                    # Use new confidence method if available
                    if parakeet_singleton and hasattr(parakeet_singleton, 'transcribe_with_confidence'):
                        text, confidence = parakeet_singleton.transcribe_with_confidence(
                            tmp_path,
                            batch_size=1,
                            num_workers=0,
                            verbose=False
                        )
                        
                        # Log model type for debugging
                        model_type = parakeet_singleton.get_model_type() if hasattr(parakeet_singleton, 'get_model_type') else 'unknown'
                        if text:
                            self.logger.debug(f"Continuous {model_type}: '{text}' (confidence: {confidence:.3f})")
                        
                        # Apply proper confidence thresholds for RNNT
                        min_confidence = self.rnnt_confidence_threshold
                        
                        if confidence < min_confidence:
                            self.logger.debug(f"Continuous low confidence ({confidence:.3f}), rejecting: '{text}'")
                            return None
                        
                    else:
                        # Fallback to old method
                        transcriptions = self.parakeet_model.transcribe(
                            [tmp_path],
                            batch_size=1,
                            return_hypotheses=False,
                            num_workers=0,
                            verbose=False
                        )
                        
                        if transcriptions and len(transcriptions) > 0:
                            hypothesis = transcriptions[0]
                            
                            # Get confidence score (old method)
                            confidence = 0.0
                            if hasattr(hypothesis, 'score'):
                                confidence = hypothesis.score
                            elif hasattr(hypothesis, 'confidence'): 
                                confidence = hypothesis.confidence
                            
                            # Apply RNNT confidence threshold
                            if confidence < self.rnnt_confidence_threshold:
                                return None
                            
                            # Extract text
                            if hasattr(hypothesis, 'text'):
                                text = hypothesis.text
                            else:
                                text = str(hypothesis)
                        else:
                            return None
                    
                    # Clean up text if we have it
                    if text:
                        text = text.strip()
                        # Remove common ASR artifacts
                        import re
                        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
                        text = re.sub(r'\s+', ' ', text)    # Normalize whitespace
                        
                        # Apply same quality filters as sequential method
                        if len(text.strip()) < 2:  # Filter very short transcriptions
                            return None
                        
                        # CRITICAL FIX: Store transcription immediately when successfully transcribed
                        current_time = time.time()
                        with self.transcription_lock:
                            self.transcriptions.append((current_time, text, confidence))
                            # Keep only last 120 seconds of transcriptions (increased to preserve more speech)
                            cutoff_time = current_time - 120.0
                            self.transcriptions = [(t, txt, conf) for t, txt, conf in self.transcriptions if t > cutoff_time]
                        
                        # Log the stored transcription with context
                        if self.current_playback_start:
                            self.logger.info(f"🎤 STORED (during playback): '{text}' (confidence: {confidence:.2f})")
                        else:
                            self.logger.info(f"🎤 STORED: '{text}' (confidence: {confidence:.2f})")
                        
                        return text, confidence
                        
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Audio chunk processing error: {e}")
            return None
    
    def mark_playback_start(self):
        """Mark that audio playback has started"""
        self.current_playback_start = time.time()
        self.logger.debug(f"Playback started at {self.current_playback_start}")
    
    def mark_playback_end(self):
        """Mark that audio playback has ended"""
        if self.current_playback_start:
            end_time = time.time()
            with self.audio_lock:
                self.playback_periods.append((self.current_playback_start, end_time))
            self.logger.debug(f"Playback ended at {end_time} (duration: {end_time - self.current_playback_start:.2f}s)")
            self.current_playback_start = None
    
    def get_response_audio_after_playback(self, timeout=10.0):
        """
        Extract user response audio that occurred after the last playback ended
        Returns audio bytes or None if no response found
        """
        if not self.playback_periods:
            self.logger.debug("No playback periods recorded")
            return None
        
        last_playback_end = self.playback_periods[-1][1]
        response_start_time = last_playback_end + 1.0  # Wait 1.0s after playback (increased from 0.5s)
        
        # Collect audio from response_start_time to now
        with self.audio_lock:
            response_audio = bytearray()
            
            for i, timestamp in enumerate(self.audio_timestamps):
                if timestamp >= response_start_time:
                    # Add this audio chunk
                    if i < len(self.recorded_audio_data):
                        response_audio.extend(self.recorded_audio_data[i])
            
            # Apply energy validation before returning
            if len(response_audio) > 1600:  # At least 100ms of audio
                if self._is_valid_speech(bytes(response_audio)):
                    return bytes(response_audio)
                else:
                    self.logger.debug("Response audio failed energy validation")
        
        return None
    
    def get_transcriptions_since(self, start_time, min_confidence=0.3):
        """
        Get all transcriptions that occurred after start_time
        Only returns transcriptions with sufficient confidence
        Returns list of (timestamp, text, confidence) tuples
        """
        with self.transcription_lock:
            filtered_transcriptions = [(timestamp, text, confidence) 
                                     for timestamp, text, confidence in self.transcriptions 
                                     if timestamp >= start_time and confidence >= min_confidence]
            
            # Debug logging to understand what's being filtered
            total_count = len(self.transcriptions)
            time_filtered_count = len([t for t, _, _ in self.transcriptions if t >= start_time])
            confidence_filtered_count = len([c for _, _, c in self.transcriptions if c >= min_confidence])
            final_count = len(filtered_transcriptions)
            
            self.logger.debug(f"Transcription filter: total={total_count}, after_time={time_filtered_count}, "
                            f"conf>={min_confidence}: {confidence_filtered_count}, final={final_count}")
            
            return filtered_transcriptions
    
    def get_latest_transcription_after(self, start_time, min_confidence=0.3):
        """
        Get the most recent transcription that occurred after start_time
        Returns (timestamp, text, confidence) tuple or None
        """
        transcriptions = self.get_transcriptions_since(start_time, min_confidence)
        if transcriptions:
            return transcriptions[-1]  # Return most recent
        return None
    
    def is_intent_triggered(self):
        """Check if any negative intent was triggered"""
        return self.dnc_triggered or self.ni_triggered or self.obscenity_triggered or self.hp_triggered or self.clbk_triggered
    
    def get_triggered_intent(self):
        """Get the type of intent that was triggered"""
        if self.dnc_triggered:
            return "DNC"
        elif self.ni_triggered:
            return "NI"  
        elif self.obscenity_triggered:
            return "OBSCENITY"
        elif self.hp_triggered:
            return "HP"
        elif self.clbk_triggered:
            return "CLBK"
        return None
    
    def get_stats(self):
        """Get detection statistics"""
        return self.stats.copy()

    def get_recording_file_path(self) -> Optional[str]:
        """Get the current recording file path from the audio recorder"""
        try:
            if self.audio_recorder and hasattr(self.audio_recorder, 'recording_file_path'):
                return self.audio_recorder.recording_file_path
            return None
        except Exception as e:
            self.logger.error(f"Error getting recording file path: {e}")
            return None
    
    def get_audio_buffer_state(self):
        """
        Get current state of audio buffer for real-time processing.
        Returns: (has_audio, audio_bytes) tuple
        """
        with self.audio_lock:
            if not self.audio_buffer:
                return False, None
            
            # Return a copy of current buffer
            return True, bytes(self.audio_buffer)
    
    def clear_audio_buffer(self):
        """Clear the audio buffer for fresh recording"""
        with self.audio_lock:
            self.audio_buffer.clear()
            self.buffer_timestamps.clear()
        
        with self.speech_silence_lock:
            self.voice_detected = False
            self.speech_start = None
            self.silence_start = None
    
    def add_audio_chunk_for_processing(self, audio_chunk):
        """
        Add audio chunk and check if ready for processing using Silero VAD.
        Returns True if audio should be processed based on speech-silence pattern.
        """
        if not audio_chunk:
            return False

        try:
            with self.speech_silence_lock:
                current_time = time.time()

                # Voice activity detection using Silero VAD
                if self.silero_vad.is_speech(audio_chunk):
                    if not self.voice_detected:
                        self.speech_start = current_time
                    self.voice_detected = True
                    self.silence_start = None
                else:
                    # Not speech (silence or noise)
                    if self.voice_detected:
                        if self.silence_start is None:
                            self.silence_start = current_time

                # Check if we should process (same logic as sequential)
                if (self.voice_detected and self.silence_start and self.speech_start):
                    silence_duration = current_time - self.silence_start
                    speech_duration = self.silence_start - self.speech_start

                    if (silence_duration >= SILENCE_DURATION_THRESHOLD and
                        speech_duration >= MIN_SPEECH_DURATION):
                        # Reset for next detection
                        self.voice_detected = False
                        self.speech_start = None
                        self.silence_start = None
                        return True

                return False

        except Exception as e:
            self.logger.debug(f"Chunk processing error: {e}")
            return False
    
    def get_accumulated_audio(self):
        """Get all accumulated audio and clear buffer"""
        with self.audio_lock:
            if not self.audio_buffer:
                return None
            
            audio = bytes(self.audio_buffer)
            self.audio_buffer.clear()
            self.buffer_timestamps.clear()
            return audio
    
    def get_real_time_audio_for_processing(self, timeout=10.0):
        """
        Get real-time audio chunks for direct processing by main flow.
        Similar to the sequential approach but from continuous stream.
        Returns audio bytes when speech-silence pattern is detected.
        """
        start_time = time.time()
        
        while (time.time() - start_time) < timeout and not self.stop_event.is_set():
            # Check if we have audio ready for processing based on VAD
            with self.speech_silence_lock:
                if (self.voice_detected and self.silence_start and self.speech_start):
                    silence_duration = time.time() - self.silence_start
                    speech_duration = self.silence_start - self.speech_start
                    
                    if (silence_duration >= SILENCE_DURATION_THRESHOLD and 
                        speech_duration >= MIN_SPEECH_DURATION):
                        
                        # Get the audio from the recent buffer (last 5 seconds to include full speech)
                        with self.audio_lock:
                            if self.audio_buffer and self.buffer_timestamps:
                                current_time = time.time()
                                window_start = current_time - 5.0  # 5 second window
                                
                                # Find audio in the window (improved indexing)
                                window_audio = bytearray()
                                if self.buffer_timestamps:
                                    # Find the most recent audio (last 5 seconds worth)
                                    bytes_to_include = min(len(self.audio_buffer), 5 * 8000 * 2)  # 5 seconds at 8kHz, 16-bit
                                    if bytes_to_include > 0:
                                        window_audio.extend(self.audio_buffer[-bytes_to_include:])
                                
                                # Reset VAD state for next detection
                                self.voice_detected = False
                                self.speech_start = None
                                self.silence_start = None
                                
                                # Validate before returning
                                if self._is_valid_speech(bytes(window_audio)):
                                    return bytes(window_audio)
            
            # Check for immediate processing during playback (for interruptions)
            if self.current_playback_start:
                with self.audio_lock:
                    if self.audio_buffer and self.buffer_timestamps:
                        # Get recent audio (last 2 seconds during playback)
                        current_time = time.time()
                        window_start = current_time - 2.0
                        
                        window_audio = bytearray()
                        if self.buffer_timestamps:
                            # Get recent audio (last 2 seconds during playback)
                            bytes_to_include = min(len(self.audio_buffer), 2 * 8000 * 2)  # 2 seconds at 8kHz, 16-bit
                            if bytes_to_include > 0:
                                window_audio.extend(self.audio_buffer[-bytes_to_include:])
                        
                        # Check for speech during playback using Silero VAD
                        if len(window_audio) > 1600:  # At least 100ms
                            try:
                                if self.silero_vad.is_speech(bytes(window_audio)):
                                    return bytes(window_audio)
                            except:
                                pass
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.1)
        
        return None

    def flush_pending_transcription(self, reason="listener_switch"):
        """
        Force processing of any accumulated audio without waiting for silence.
        Used when switching from continuous to main listener to capture pending speech.
        """
        try:
            with self.audio_lock:
                if not self.audio_buffer or not self.buffer_timestamps:
                    self.logger.debug(f"No pending audio to flush ({reason})")
                    return []
                
                current_time = time.time()
                
                # Get all audio from last 2 seconds or all available if less
                window_start = current_time - 2.0
                window_audio = bytearray()
                
                for i, timestamp in enumerate(self.buffer_timestamps):
                    if timestamp >= window_start:
                        # Approximate audio chunk size
                        chunk_size = len(self.audio_buffer) // len(self.buffer_timestamps)
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, len(self.audio_buffer))
                        if start_idx < len(self.audio_buffer):
                            window_audio.extend(self.audio_buffer[start_idx:end_idx])
                
                # Check if we have enough audio to process
                if len(window_audio) < 1600:  # Minimum 0.2 seconds at 8kHz
                    self.logger.debug(f"Insufficient audio to flush ({len(window_audio)} bytes)")
                    return []
                
                # Apply validation
                if not self._is_valid_speech(bytes(window_audio)):
                    self.logger.debug(f"Flushed audio failed speech validation ({reason})")
                    return []
                
                # Process the audio chunk directly
                result = self._process_audio_chunk(bytes(window_audio))
                if result:
                    # Extract text and confidence if tuple
                    if isinstance(result, tuple):
                        text, confidence = result
                    else:
                        text = result
                        confidence = 0.0
                    
                    if text and len(text.strip()) > 0:
                        # Store transcription with timestamp
                        with self.transcription_lock:
                            self.transcriptions.append((current_time, text, confidence))
                            # Keep only last 120 seconds of transcriptions (increased to preserve more speech)
                            cutoff_time = current_time - 120.0
                            self.transcriptions = [(t, txt, conf) for t, txt, conf in self.transcriptions if t > cutoff_time]
                        
                        # Log the forced transcription
                        self.logger.info(f"🎤 FLUSH ({reason}): '{text}' (confidence: {confidence:.2f})")
                        
                        # Check for intents on the flushed text
                        if self.intent_detector:
                            try:
                                self.intent_detector.detect_intent(text)
                                self.stats['intent_checks'] += 1
                            except Exception as e:
                                self.logger.debug(f"Intent detection error on flush: {e}")
                        
                        return [(current_time, text, confidence)]
                
                self.logger.debug(f"No valid transcription from flushed audio ({reason})")
                return []
                
        except Exception as e:
            self.logger.error(f"Error flushing pending transcription ({reason}): {e}")
            return []
        finally:
            # Reset speech detection state after flush
            with self.speech_silence_lock:
                self.voice_detected = False
                self.speech_start = None
                self.silence_start = None
