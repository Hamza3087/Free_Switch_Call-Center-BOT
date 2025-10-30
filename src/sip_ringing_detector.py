"""
Ringing detection module for SIP bot using Goertzel algorithm
Detects ringback patterns continuously throughout call duration
"""

import numpy as np
import time
import threading
import logging
import os
import tempfile
import weakref
from typing import Optional, Callable
from pydub import AudioSegment
import pjsua2 as pj

# Error codes
PJSIP_ERROR_NOT_FOUND = 171140
PJSIP_ERROR_INVALID = 171141


class ModelSingleton:
    """Singleton pattern for ringing detector to share resources efficiently"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.reference_count = 0
            self.count_lock = threading.Lock()
            self._initialized = True
    
    def get_detector(self, logger=None):
        """Get ringing detector instance"""
        with self.count_lock:
            self.reference_count += 1
            if logger:
                logger.info(f"Ringing detector acquired (refs: {self.reference_count})")
            return True
    
    def release(self, logger=None):
        """Release ringing detector instance"""
        with self.count_lock:
            self.reference_count = max(0, self.reference_count - 1)
            if logger:
                logger.info(f"Ringing detector released (refs: {self.reference_count})")


# Global singleton instance
ringing_model_singleton = ModelSingleton()


class GoertzelDetector:
    """Goertzel algorithm implementation for frequency detection"""
    
    def __init__(self, sample_rate: int = 8000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
    def detect_frequency(self, samples: np.ndarray, target_freq: float) -> float:
        """Compute power at target frequency using Goertzel algorithm"""
        n = len(samples)
        if n == 0:
            return 0.0
            
        k = int(0.5 + ((n * target_freq) / self.sample_rate))
        omega = (2.0 * np.pi * k) / n
        coeff = 2.0 * np.cos(omega)
        
        q0 = q1 = q2 = 0.0
        for sample in samples:
            q0 = coeff * q1 - q2 + float(sample)
            q2, q1 = q1, q0
        
        return q1**2 + q2**2 - q1*q2*coeff


class RingbackPatternMatcher:
    """Pattern matcher for different regional ringback patterns"""
    
    def __init__(self):
        # Regional patterns (on_time, off_time in seconds)
        self.patterns = {
            'us_canada': {'on': 2.0, 'off': 4.0, 'frequencies': [440.0, 480.0]},
            'uk': {'on': 0.4, 'off': 0.2, 'frequencies': [400.0, 450.0]},  # Simplified
            'europe': {'on': 1.0, 'off': 4.0, 'frequencies': [425.0]},
            'australia': {'on': 0.4, 'off': 0.2, 'frequencies': [400.0, 425.0]}
        }
        
        # Use US/Canada pattern by default
        self.current_pattern = self.patterns['us_canada']
        
    def is_ringing_pattern(self, detections: list, timestamps: list) -> bool:
        """Check if detection pattern matches ringback timing"""
        if len(detections) < 10:  # Need minimum data
            return False
            
        # Find on/off transitions
        transitions = []
        current_state = detections[0]
        
        for i, detection in enumerate(detections[1:], 1):
            if detection != current_state:
                transitions.append((timestamps[i], detection))
                current_state = detection
        
        if len(transitions) < 4:  # Need at least 2 complete cycles
            return False
        
        # Check timing patterns
        on_times = []
        off_times = []
        
        for i in range(0, len(transitions) - 1, 2):
            if i + 1 < len(transitions):
                if transitions[i][1] == 1:  # Start of ON period
                    on_duration = transitions[i + 1][0] - transitions[i][0]
                    on_times.append(on_duration)
                if i + 2 < len(transitions):
                    off_duration = transitions[i + 2][0] - transitions[i + 1][0]
                    off_times.append(off_duration)
        
        # Check if timing matches expected pattern (with tolerance)
        expected_on = self.current_pattern['on']
        expected_off = self.current_pattern['off']
        tolerance = 0.8  # 800ms tolerance for more flexibility
        
        valid_on = sum(1 for t in on_times if abs(t - expected_on) < tolerance)
        valid_off = sum(1 for t in off_times if abs(t - expected_off) < tolerance)
        
        # At least 60% of timings should match (more permissive for real-world conditions)
        on_match_rate = valid_on / len(on_times) if on_times else 0
        off_match_rate = valid_off / len(off_times) if off_times else 0
        
        # Need at least 2 complete cycles and reasonable match rates
        has_enough_cycles = len(on_times) >= 2 and len(off_times) >= 1
        pattern_matches = on_match_rate >= 0.6 and off_match_rate >= 0.6
        
        return has_enough_cycles and pattern_matches


class LocalRingingDetector:
    """
    Production-grade continuous ringing detector with robust error handling
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        from src.config import (
            RINGING_MIN_CYCLES, RINGING_CONFIDENCE_THRESHOLD, 
            RINGING_CHUNK_SIZE, RINGING_RELATIVE_THRESHOLD
        )
        self.min_cycles = RINGING_MIN_CYCLES
        self.confidence_threshold = RINGING_CONFIDENCE_THRESHOLD
        self.chunk_size = RINGING_CHUNK_SIZE
        self.relative_threshold = RINGING_RELATIVE_THRESHOLD
        
        # State management
        self.state_lock = threading.RLock()
        self.is_active = False
        self.detection_complete = False
        
        # Detection thread
        self.detection_thread = None
        self.stop_event = threading.Event()
        
        # Callbacks
        self.ringing_callback = None
        self.callback_lock = threading.Lock()
        
        # Audio recording
        self.audio_recorder = None
        self.recorded_audio_data = []
        self.audio_lock = threading.Lock()
        
        # Detection components
        self.goertzel_detector = GoertzelDetector(8000, self.chunk_size)
        self.pattern_matcher = RingbackPatternMatcher()
        
        # Detection state
        self.detections = []
        self.timestamps = []
        self.detection_start_time = None
        self.audio_buffer = b''  # Accumulate audio for better analysis
        self.min_analysis_size = self.chunk_size * 2  # Need at least 2 chunks for reliable analysis
        
        # Ring counting state - require 2 complete rings before triggering
        self.ring_cycles = []  # Track ring on/off cycles
        self.current_ring_state = False  # Track if currently in ringing phase
        self.last_state_change = None  # Track when state last changed
        self.completed_rings = 0  # Count of completed ring cycles
        self.required_rings = 2  # Require 2 rings before triggering
        
        # Enhanced detection validation
        self.consecutive_detections = 0  # Track consecutive positive detections
        self.required_consecutive = 2    # Reduced for real-time audio compatibility
        self.max_strength_threshold = 100.0  # Filter out extremely high impulse noise
        self.frequency_balance_ratio = 6.5  # Relaxed for real-time audio (was 5.0)
        
        # Pattern validation for sustained signals
        self.high_strength_buffer = []   # Track recent high-strength detections
        self.high_strength_window = 5    # Look at last 5 chunks for pattern
        
        # Audio activity detection flag
        self.audio_activity_detected = False
        self.audio_activity_lock = threading.Lock()
        
        # Stats
        self.stats = {
            'detections_started': 0,
            'detections_completed': 0,
            'ringing_detected': 0,
            'errors': 0,
            'chunks_analyzed': 0,
            'audio_bytes_processed': 0,
            'completed_rings': 0,
            'required_rings': self.required_rings,
            'audio_activity_detected': False
        }
        
        self.logger.info(f"Ringing detector initialized (threshold={self.relative_threshold}, required_rings={self.required_rings})")
    
    def set_ringing_callback(self, callback: Callable):
        """Set callback for ringing detection"""
        with self.callback_lock:
            self.ringing_callback = callback
    
    def set_audio_recorder(self, audio_recorder):
        """Set audio recorder instance"""
        self.audio_recorder = audio_recorder
    
    def set_required_rings(self, required_rings: int):
        """Set the number of rings required before triggering callback"""
        if required_rings < 1:
            required_rings = 1
        elif required_rings > 10:
            required_rings = 10
        
        with self.state_lock:
            self.required_rings = required_rings
            self.stats['required_rings'] = required_rings
    
    def has_audio_activity(self) -> bool:
        """Check if any audio activity has been detected during the call"""
        with self.audio_activity_lock:
            return self.audio_activity_detected
    
    def start_detection(self, audio_media) -> bool:
        """
        Start continuous ringing detection
        Returns True if started successfully
        """
        with self.state_lock:
            if self.is_active:
                self.logger.warning("Ringing detection already active")
                return False
            
            if not self.audio_recorder:
                self.logger.error("No audio recorder set for ringing detection")
                return False
            
            self.logger.info("Starting continuous ringing detection...")
            
            # Reset state
            self.stop_event.clear()
            self.detection_complete = False
            self.detection_start_time = time.time()
            
            with self.audio_lock:
                self.recorded_audio_data.clear()
                self.detections.clear()
                self.timestamps.clear()
                
                # Reset ring counting state
                self.ring_cycles.clear()
                self.current_ring_state = False
                self.last_state_change = None
                self.completed_rings = 0
                
                # Reset enhanced detection state
                self.consecutive_detections = 0
                self.high_strength_buffer.clear()
            
            # Start detection thread
            self.detection_thread = threading.Thread(
                target=self._detection_loop_safe,
                args=(audio_media,),
                name=f"ringing_{id(self)}",
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
            
            self.logger.info("Stopping ringing detection...")
            
            # Signal stop
            self.stop_event.set()
            self.is_active = False
        
        # Wait for thread outside lock
        if self.detection_thread and self.detection_thread.is_alive():
            if threading.current_thread() != self.detection_thread:
                self.detection_thread.join(timeout=2.0)
        
        # Release model reference
        ringing_model_singleton.release(self.logger)
        
        self.logger.info("Ringing detection stopped")
    
    def _detection_loop_safe(self, audio_media):
        """Detection loop with comprehensive error handling"""
        try:
            # Register thread with PJSIP
            self._register_pjsip_thread()
            
            # Run continuous detection
            self._run_continuous_detection(audio_media)
            
        except Exception as e:
            self.logger.error(f"Ringing detection error: {e}", exc_info=True)
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
                pj.Endpoint.instance().libRegisterThread("ringing_thread")
                threading.current_thread()._pj_registered = True
                
        except Exception as e:
            self.logger.warning(f"PJSIP registration warning: {e}")
    
    def _run_continuous_detection(self, audio_media):
        """Main continuous detection logic - runs for entire call"""
        recording_file = None
        
        try:
            # Start recording
            recording_file = self.audio_recorder.start_recording(audio_media)
            if not recording_file:
                self.logger.error("Failed to start recording for ringing detection")
                return
            
            self.logger.info(f"Ringing detection recording started (continuous): {recording_file}")
            
            # Wait for initial data and verify audio format
            initial_wait_count = 0
            while initial_wait_count < 20 and not self.stop_event.is_set():  # Wait up to 1 second
                if os.path.exists(recording_file) and os.path.getsize(recording_file) > 44:  # WAV header is 44 bytes
                    break
                time.sleep(0.05)
                initial_wait_count += 1
            
            # Log file info and verify format
            if os.path.exists(recording_file):
                file_size = os.path.getsize(recording_file)
                self.logger.info(f"Recording file created: {file_size} bytes")
                
                # Try to read and verify format
                try:
                    with open(recording_file, 'rb') as f:
                        header = f.read(44)
                        if len(header) >= 44 and header.startswith(b'RIFF'):
                            # Parse basic WAV info
                            sample_rate = int.from_bytes(header[24:28], 'little')
                            channels = int.from_bytes(header[22:24], 'little')
                            bits_per_sample = int.from_bytes(header[34:36], 'little')
                            self.logger.info(f"Audio format: {sample_rate}Hz, {channels}ch, {bits_per_sample}bit")
                            
                            if sample_rate != 8000:
                                self.logger.warning(f"Expected 8000Hz, got {sample_rate}Hz - detection may be less accurate")
                        else:
                            self.logger.warning("Audio file doesn't have valid WAV header")
                except Exception as e:
                    self.logger.error(f"Failed to verify audio format: {e}")
            else:
                self.logger.error(f"Recording file not found: {recording_file}")
            
            # Continuous detection loop - no timeout, runs until call ends
            last_file_size = 0
            warning_count = 0       # For logging warnings every 5 seconds
            total_silence_count = 0 # For tracking total silence duration
            
            # Log initial recording info
            detection_count = 0
            last_stats_time = time.time()
            
            while not self.stop_event.is_set():
                # Check if audio recorder is still valid
                if not self.audio_recorder:
                    self.logger.info("Audio recorder no longer available - stopping ringing detection")
                    break
                
                # Get new audio data
                try:
                    new_audio = self.audio_recorder.get_new_audio_data(last_file_size)
                except Exception as e:
                    self.logger.warning(f"Failed to get audio data: {e} - stopping ringing detection")
                    break
                if new_audio:
                    warning_count = 0
                    total_silence_count = 0
                    detection_count += 1
                    self.logger.debug(f"Got {len(new_audio)} bytes of audio data (file size: {last_file_size})")
                    
                    # Log statistics every 10 seconds
                    current_time = time.time()
                    if current_time - last_stats_time > 10.0:
                        total_detections = len(self.detections)
                        positive_detections = sum(self.detections) if self.detections else 0
                        detection_rate = positive_detections / total_detections if total_detections > 0 else 0
                        buffer_size = len(self.audio_buffer) if hasattr(self, 'audio_buffer') else 0
                        self.logger.info(f"Ringing stats: chunks={detection_count}, detections={total_detections}, positive={positive_detections} ({detection_rate:.1%}), buffer={buffer_size} bytes")
                        last_stats_time = current_time
                    
                    with self.audio_lock:
                        self.recorded_audio_data.append(new_audio)
                        self.audio_buffer += new_audio
                        
                        # Keep buffer manageable (last 10 seconds worth)
                        max_buffer_size = 8000 * 2 * 10  # 10 seconds of 8kHz 16-bit audio
                        if len(self.audio_buffer) > max_buffer_size:
                            self.audio_buffer = self.audio_buffer[-max_buffer_size:]
                    
                    # Update file size
                    if recording_file and os.path.exists(recording_file):
                        last_file_size = os.path.getsize(recording_file)
                    else:
                        # Recording file no longer exists - call likely disconnected
                        self.logger.info("Recording file no longer exists - stopping ringing detection")
                        break
                    
                    # Analyze accumulated audio buffer for ringing
                    if len(self.audio_buffer) >= self.min_analysis_size:
                        if self._analyze_audio_chunk(self.audio_buffer[-self.min_analysis_size:]):
                            self.logger.info("🔔 Ringing detected during call")
                            self.stats['ringing_detected'] += 1
                            self._trigger_callback()
                            break
                else:
                    warning_count += 1
                    total_silence_count += 1
                    
                    # Log warning every 5 seconds
                    if warning_count > 100:  # 5 seconds of no data (100 * 0.05s)
                        self.logger.warning(f"No audio data received for {total_silence_count * 0.05:.1f}s")
                        warning_count = 0  # Reset warning counter for next warning
                    
                    # Stop after 10 seconds of total silence
                    if total_silence_count >= 200:  # 10 seconds of no data (200 * 0.05s)
                        self.logger.info("Extended silence detected - stopping ringing detection")
                        break
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.05)  # 50ms intervals for responsive detection
            
        except Exception as e:
            self.logger.error(f"Detection recording error: {e}")
            
        finally:
            # Stop recording with proper error handling
            try:
                self.audio_recorder.stop_recording()
            except Exception as e:
                self.logger.error(f"Failed to stop ringing detection recording: {e}")
    
    def _analyze_audio_chunk(self, audio_data: bytes) -> bool:
        """
        Enhanced audio chunk analysis with false positive filtering
        Returns True if ringing detected
        """
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            self.stats['chunks_analyzed'] += 1
            self.stats['audio_bytes_processed'] += len(audio_data)
            
            if len(audio_array) < self.chunk_size // 4:  # Too small
                return False
            
            # Check for general audio activity (any frequency content)
            self._detect_audio_activity(audio_array)
            
            # Get target frequencies (US/Canada pattern)
            target_freqs = [440.0, 480.0]
            
            # Calculate energy at target frequencies
            target_energies = []
            for freq in target_freqs:
                energy = self.goertzel_detector.detect_frequency(audio_array, freq)
                target_energies.append(energy)
            
            # Calculate background energy (nearby frequencies)
            bg_freqs = [300, 350, 400, 500, 550, 600]
            bg_energies = []
            for freq in bg_freqs:
                energy = self.goertzel_detector.detect_frequency(audio_array, freq)
                bg_energies.append(energy)
            
            avg_bg = np.mean(bg_energies) + 1e-9  # Avoid division by zero
            min_target = min(target_energies)
            relative_strength = min_target / avg_bg
            
            # Enhanced validation checks
            is_valid_detection = self._validate_detection(
                target_energies, relative_strength, audio_array
            )
            
            # Record detection with current time
            current_time = time.time() - self.detection_start_time
            with self.audio_lock:
                self.detections.append(1 if is_valid_detection else 0)
                self.timestamps.append(current_time)
                
                # Keep only recent data (last 30 seconds)
                if len(self.detections) > 600:  # 30s * 20fps
                    self.detections = self.detections[-600:]
                    self.timestamps = self.timestamps[-600:]
            
            # Use ring cycle tracking (simplified for real-time compatibility)
            if self._track_ring_cycle(is_valid_detection, current_time):
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Audio chunk analysis error: {e}")
            return False
    
    def _detect_audio_activity(self, audio_array: np.ndarray):
        """
        Detect any audio activity (frequency content) to differentiate 
        silent calls from calls with audio but no speech
        """
        try:
            # Skip if already detected audio activity
            with self.audio_activity_lock:
                if self.audio_activity_detected:
                    return
            
            # Check for any significant frequency content across a broader range
            # Use multiple frequency bands to detect any kind of audio signal
            test_frequencies = [200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 2000]
            
            total_energy = 0.0
            max_single_freq_energy = 0.0
            
            for freq in test_frequencies:
                energy = self.goertzel_detector.detect_frequency(audio_array, freq)
                total_energy += energy
                max_single_freq_energy = max(max_single_freq_energy, energy)
            
            # Calculate average energy for background noise estimation
            avg_energy = total_energy / len(test_frequencies)
            
            # Set thresholds for audio activity detection (more sensitive than ringing detection)
            # Use lower thresholds to detect any audio activity, not just strong signals
            activity_threshold = 1e4  # Much lower than ringing detection threshold (1e5)
            max_energy_threshold = 5e4  # Lower threshold for maximum single frequency
            
            # Check if we have significant audio activity
            has_activity = (
                avg_energy > activity_threshold or 
                max_single_freq_energy > max_energy_threshold
            )
            
            if has_activity:
                with self.audio_activity_lock:
                    if not self.audio_activity_detected:
                        self.audio_activity_detected = True
                        self.stats['audio_activity_detected'] = True
                        self.logger.info(f"🎵 Audio activity detected: avg_energy={avg_energy:.2e}, max_energy={max_single_freq_energy:.2e}")
            
        except Exception as e:
            self.logger.error(f"Audio activity detection error: {e}")
    
    def _validate_detection(self, target_energies: list, relative_strength: float, audio_array: np.ndarray) -> bool:
        """
        Enhanced validation to filter false positives
        """
        try:
            energy_440 = target_energies[0]
            energy_480 = target_energies[1]
            
            # Check 1: Basic threshold
            if relative_strength <= self.relative_threshold:
                self.consecutive_detections = 0
                self.logger.debug(f"Detection rejected: strength {relative_strength:.2f} <= threshold {self.relative_threshold}")
                return False
            
            # Check 2: Pattern-based validation for extremely high strength
            # Instead of flat rejection, check if high strength is sustained (real ringing)
            # vs isolated spikes (impulse noise)
            if relative_strength > self.max_strength_threshold:
                # Track high-strength detections in buffer
                self.high_strength_buffer.append(relative_strength)
                if len(self.high_strength_buffer) > self.high_strength_window:
                    self.high_strength_buffer.pop(0)
                
                # If we have sustained high strength (3+ out of last 5 chunks), it's likely real ringing
                high_count = sum(1 for s in self.high_strength_buffer if s > self.max_strength_threshold)
                if high_count < 3 and len(self.high_strength_buffer) >= 3:
                    self.consecutive_detections = 0
                    self.logger.debug(f"Detection rejected: isolated impulse noise {relative_strength:.2f}, sustained={high_count}/5")
                    return False
                else:
                    self.logger.debug(f"High strength accepted: sustained pattern {high_count}/5 chunks > {self.max_strength_threshold}")
            else:
                # Normal strength, clear high strength buffer
                self.high_strength_buffer.clear()
            
            # Check 3: Frequency balance - relaxed for real-time audio
            freq_ratio = max(energy_440, energy_480) / (min(energy_440, energy_480) + 1e-9)
            if freq_ratio > self.frequency_balance_ratio:
                self.consecutive_detections = 0
                self.logger.debug(f"Detection rejected: unbalanced frequencies ratio {freq_ratio:.2f} > {self.frequency_balance_ratio}")
                return False
            
            # Check 4: Minimum energy threshold (relaxed for real-time audio)
            min_energy = min(energy_440, energy_480)
            if min_energy < 1e5:  # Reduced threshold for SIP audio (was 1e6)
                self.consecutive_detections = 0
                self.logger.debug(f"Detection rejected: energy too low {min_energy:.2e}")
                return False
            
            # Check 5: Consecutive detection requirement
            self.consecutive_detections += 1
            if self.consecutive_detections < self.required_consecutive:
                self.logger.debug(f"Detection pending: consecutive {self.consecutive_detections}/{self.required_consecutive}")
                return False
            
            # All checks passed
            self.logger.debug(f"Detection valid: strength={relative_strength:.2f}, ratio={freq_ratio:.2f}, consecutive={self.consecutive_detections}")
            return True
            
        except Exception as e:
            self.logger.error(f"Detection validation error: {e}")
            self.consecutive_detections = 0
            return False
    
    def _track_ring_cycle(self, is_ringing: bool, current_time: float) -> bool:
        """
        Track ring cycles and return True if required number of rings completed
        A complete ring cycle = ON phase + OFF phase
        """
        state_changed = False
        
        # Check for state change
        if is_ringing != self.current_ring_state:
            if self.last_state_change is not None:
                # Calculate duration of previous state
                duration = current_time - self.last_state_change
                
                # Record the cycle transition
                cycle_info = {
                    'from_state': 'ringing' if self.current_ring_state else 'silence',
                    'to_state': 'ringing' if is_ringing else 'silence',
                    'duration': duration,
                    'timestamp': current_time
                }
                self.ring_cycles.append(cycle_info)
                
                # Check if we completed a ring (ringing -> silence transition)
                if self.current_ring_state and not is_ringing:
                    self.completed_rings += 1
                    
                    # Check if we have enough rings
                    if self.completed_rings >= self.required_rings:
                        return True
                
                state_changed = True
            
            # Update state
            self.current_ring_state = is_ringing
            self.last_state_change = current_time
            
        
        return False
    
    def _trigger_callback(self):
        """Trigger ringing detection callback"""
        with self.callback_lock:
            if self.ringing_callback:
                try:
                    self.logger.info("Triggering ringing callback...")
                    self.ringing_callback()
                except Exception as e:
                    self.logger.error(f"Ringing callback error: {e}")
                    self.stats['errors'] += 1
    
    def get_stats(self):
        """Get detector statistics"""
        with self.state_lock:
            stats = self.stats.copy()
            stats['completed_rings'] = self.completed_rings
            with self.audio_activity_lock:
                stats['audio_activity_detected'] = self.audio_activity_detected
            return stats
    
    def is_detection_complete(self):
        """Check if detection is complete"""
        with self.state_lock:
            return self.detection_complete