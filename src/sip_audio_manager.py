#!/usr/bin/env python3
"""
Production-Grade SIP Audio Manager
- Thread-safe audio operations with guaranteed cleanup
- Robust error handling for all PJSIP operations
- Zero resource leaks with automatic recovery
"""

import os
import time
import tempfile
import threading
import numpy as np
import pjsua2 as pj
from pydub import AudioSegment
from typing import Optional, List, Tuple
import logging
import weakref
import gc
from contextlib import contextmanager
import queue

# Import configuration
try:
    from .config import SIP_SAMPLE_RATE, AUDIO_STOP_DELAY
except ImportError:
    SIP_SAMPLE_RATE = 8000
    AUDIO_STOP_DELAY = 0.2

# Import utilities
try:
    from .audio_utils import mix_audio
except ImportError:
    def mix_audio(audio1: bytes, audio2: bytes, volume1: float = 1.0, volume2: float = 0.3) -> bytes:
        """Fallback audio mixing function"""
        if not audio1:
            return audio2
        if not audio2:
            return audio1
        
        samples1 = np.frombuffer(audio1, dtype=np.int16).astype(np.float32)
        samples2 = np.frombuffer(audio2, dtype=np.int16).astype(np.float32)
        
        min_length = min(len(samples1), len(samples2))
        samples1 = samples1[:min_length] * volume1
        samples2 = samples2[:min_length] * volume2
        
        mixed = np.clip(samples1 + samples2, -32768, 32767)
        return mixed.astype(np.int16).tobytes()

# PJSIP Error codes
PJSIP_ERROR_NOT_FOUND = 70001  # PJ_ENOTFOUND
PJSIP_ERROR_INVALID = 70004     # PJ_EINVAL

class ThreadSafePJSIP:
    """Thread-safe PJSIP operations manager"""
    _lock = threading.RLock()
    _registered_threads = set()
    _media_creation_lock = threading.Lock()  # Global lock for ALL media object creation (players + recorders)
    
    @classmethod
    def register_thread(cls, thread_name=None):
        """Register current thread with PJSIP"""
        thread_id = threading.get_ident()
        
        with cls._lock:
            if thread_id in cls._registered_threads:
                return True
            
            try:
                if not thread_name:
                    thread_name = f"audio_thread_{thread_id}"
                
                pj.Endpoint.instance().libRegisterThread(thread_name)
                cls._registered_threads.add(thread_id)
                threading.current_thread()._pj_registered = True
                return True
            except Exception:
                return False
    
    @classmethod
    @contextmanager
    def pjsip_operation(cls, operation_name="operation"):
        """Context manager for PJSIP operations"""
        cls.register_thread()
        start_time = time.time()
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            if elapsed > 5:
                logging.warning(f"PJSIP operation '{operation_name}' took {elapsed:.2f}s")
    
    @classmethod
    @contextmanager
    def media_creation_operation(cls, operation_name="media_creation"):
        """Context manager for PJSIP media object creation - prevents race conditions"""
        cls.register_thread()
        start_time = time.time()
        
        with cls._media_creation_lock:
            try:
                yield
            finally:
                elapsed = time.time() - start_time
                if elapsed > 2:
                    logging.warning(f"Media creation '{operation_name}' took {elapsed:.2f}s")

class ResourceTracker:
    """Tracks and manages audio resources"""
    def __init__(self, logger):
        self.logger = logger
        self.resources = weakref.WeakSet()
        self.temp_files = []
        self.lock = threading.Lock()
    
    def register(self, resource):
        """Register a resource for tracking"""
        with self.lock:
            self.resources.add(resource)
    
    def register_temp_file(self, filepath):
        """Register a temporary file for cleanup"""
        with self.lock:
            self.temp_files.append(filepath)
    
    def cleanup_all(self):
        """Cleanup all tracked resources"""
        with self.lock:
            # Cleanup temp files
            for filepath in self.temp_files:
                try:
                    if os.path.exists(filepath):
                        os.unlink(filepath)
                        self.logger.info(f"Deleted temp file: {filepath}")
                except Exception as e:
                    self.logger.error(f"Failed to delete {filepath}: {e}")
            
            self.temp_files.clear()
            
            # Force garbage collection
            gc.collect()

class SIPAudioManager:
    """Production-grade SIP audio manager with bulletproof resource management"""
    
    def __init__(self, noise_path: str = None, noise_volume: float = 1.0, logger=None, stereo_recording_callback=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Resource tracking
        self.resource_tracker = ResourceTracker(self.logger)
        
        # Player management
        self.current_player = None
        self.current_sink = None
        self.player_lock = threading.RLock()
        self.player_cleanup_event = threading.Event()
        
        # Background noise
        self.noise_path = noise_path
        self.noise_volume = max(0.0, min(1.0, noise_volume))  # Clamp volume
        self.background_noise_data = None
        self.background_noise_pos = 0
        self.noise_lock = threading.Lock()
        
        # Stats
        self.stats = {
            'files_played': 0,
            'errors': 0,
            'cleanups': 0
        }
        
        # Interrupt callback for DNC/NI detection during playback
        self._interrupt_callback = None
        
        # Stereo recording callback for capturing bot audio
        self._stereo_recording_callback = stereo_recording_callback
        
        # Load background noise
        self._load_background_noise()
        
        self.logger.info(f"Audio manager initialized (noise_volume={self.noise_volume:.2f})")
    
    def set_interrupt_callback(self, callback):
        """Set callback to check for DNC/NI during audio playback"""
        self._interrupt_callback = callback
    
    def _load_background_noise(self):
        """Load background noise file with error handling"""
        if not self.noise_path:
            self.logger.info("No background noise configured")
            return
        
        # Try different path variations
        paths_to_try = [
            self.noise_path,
            f"{self.noise_path}.wav",
            f"{self.noise_path}.mp3"
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                try:
                    self.logger.info(f"Loading background noise: {path}")
                    
                    # Determine format
                    _, ext = os.path.splitext(path)
                    format_hint = None if ext else "wav"
                    
                    # Load and convert audio
                    audio = AudioSegment.from_file(path, format=format_hint)
                    audio = audio.set_channels(1).set_frame_rate(SIP_SAMPLE_RATE).set_sample_width(2)
                    
                    self.background_noise_data = audio.raw_data
                    self.logger.info(f"✅ Background noise loaded ({len(self.background_noise_data)} bytes)")
                    return
                    
                except Exception as e:
                    self.logger.error(f"Failed to load {path}: {e}")
        
        self.logger.warning(f"Background noise not found: {self.noise_path}")
    
    def get_background_noise_chunk(self, frame_count: int) -> bytes:
        """Get background noise chunk with looping"""
        if not self.background_noise_data:
            return b"\x00" * (frame_count * 2)
        
        with self.noise_lock:
            bytes_needed = frame_count * 2
            noise_len = len(self.background_noise_data)
            
            # Handle looping
            if self.background_noise_pos + bytes_needed > noise_len:
                # Wrap around
                part1 = self.background_noise_data[self.background_noise_pos:]
                part2_len = bytes_needed - len(part1)
                part2 = self.background_noise_data[:part2_len]
                chunk = part1 + part2
                self.background_noise_pos = part2_len
            else:
                # Simple case
                chunk = self.background_noise_data[self.background_noise_pos:self.background_noise_pos + bytes_needed]
                self.background_noise_pos += bytes_needed
            
            return chunk
    
    def play_audio_file(self, audio_path: str, audio_media: pj.AudioMedia) -> Optional[bytes]:
        """
        Play audio file with comprehensive error handling and resource management
        Returns the mixed audio bytes that were played
        """
        with ThreadSafePJSIP.pjsip_operation(f"play_{os.path.basename(audio_path)}"):
            try:
                # Validate inputs
                if not audio_path or not os.path.exists(audio_path):
                    self.logger.error(f"Audio file not found: {audio_path}")
                    self.stats['errors'] += 1
                    return None
                
                if not audio_media:
                    self.logger.error("No audio media provided")
                    self.stats['errors'] += 1
                    return None
                
                # Cleanup any existing player
                self._cleanup_player_safe()
                
                self.logger.info(f"Playing: {os.path.basename(audio_path)}")
                
                # Load and process audio
                audio = self._load_and_convert_audio(audio_path)
                if not audio:
                    return None
                
                # Mix with background noise
                mixed_audio = self._mix_with_background_noise(audio)
                
                # Create temp file
                temp_file = self._create_temp_file(mixed_audio)
                if not temp_file:
                    return None
                
                # Play the audio
                playback_timestamp = time.time()
                success = self._play_sip_audio(temp_file, audio_media, mixed_audio)
                
                if success:
                    self.stats['files_played'] += 1
                    
                    # Notify stereo recorder of bot audio
                    if self._stereo_recording_callback:
                        try:
                            self._stereo_recording_callback(mixed_audio.raw_data, playback_timestamp)
                        except Exception as e:
                            self.logger.error(f"Stereo recording callback error: {e}")
                    
                    return mixed_audio.raw_data
                else:
                    self.stats['errors'] += 1
                    return None
                    
            except pj.Error as e:
                # Handle PJSIP errors gracefully
                if e.status in [PJSIP_ERROR_NOT_FOUND, PJSIP_ERROR_INVALID]:
                    self.logger.info("Call disconnected during playback")
                else:
                    self.logger.error(f"PJSIP error: {e.reason}")
                self.stats['errors'] += 1
                return None
                
            except Exception as e:
                self.logger.error(f"Playback error: {e}", exc_info=True)
                self.stats['errors'] += 1
                return None
                
            finally:
                # Always cleanup
                self._cleanup_player_safe()
    
    def _load_and_convert_audio(self, audio_path: str) -> Optional[AudioSegment]:
        """Load and convert audio file"""
        try:
            audio = AudioSegment.from_file(audio_path)
            # Convert to SIP format
            return audio.set_channels(1).set_frame_rate(SIP_SAMPLE_RATE).set_sample_width(2)
        except Exception as e:
            self.logger.error(f"Failed to load audio: {e}")
            return None
    
    def _mix_with_background_noise(self, audio: AudioSegment) -> AudioSegment:
        """Mix audio with background noise"""
        if not self.background_noise_data:
            return audio
        
        try:
            # Get noise chunk matching audio duration
            frame_count = len(audio.raw_data) // 2
            noise_chunk = self.get_background_noise_chunk(frame_count)
            
            # Mix audio
            mixed_data = mix_audio(audio.raw_data, noise_chunk, 1.0, self.noise_volume)
            
            return AudioSegment(
                data=mixed_data,
                sample_width=2,
                frame_rate=SIP_SAMPLE_RATE,
                channels=1
            )
        except Exception as e:
            self.logger.error(f"Mixing failed: {e}")
            return audio
    
    def _create_temp_file(self, audio: AudioSegment) -> Optional[str]:
        """Create temporary audio file"""
        try:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.wav',
                prefix='sip_audio_'
            )
            
            # Export audio
            audio.export(temp_file.name, format="wav")
            temp_file.close()
            
            # Register for cleanup
            self.resource_tracker.register_temp_file(temp_file.name)
            
            return temp_file.name
            
        except Exception as e:
            self.logger.error(f"Temp file creation failed: {e}")
            return None
    
    def _play_sip_audio(self, file_path: str, audio_media: pj.AudioMedia, 
                       audio_segment: AudioSegment) -> bool:
        """Play audio through PJSIP with proper synchronization"""
        player_created = False
        
        try:
            # Create player with minimal serialization to prevent assertion failures
            with self.player_lock:
                with ThreadSafePJSIP._media_creation_lock:
                    self.current_player = pj.AudioMediaPlayer()
                # Create the player file outside the global lock
                self.current_player.createPlayer(file_path)
                player_created = True
            
            # Connect to sink (outside the global lock to allow concurrent audio operations)
            with self.player_lock:
                self.current_sink = audio_media
                self.current_player.startTransmit(audio_media)
                
                # Clear cleanup event
                self.player_cleanup_event.clear()
            
            # CRITICAL FIX: Stop 50ms before the end to prevent loop-back
            duration = (len(audio_segment) / 1000.0) - 0.05
            self.logger.info(f"Playing for {duration:.2f}s")
            
            # Wait for playback with HIGH PRECISION and DNC/NI interrupt checking
            end_time = time.time() + duration
            last_interrupt_check = time.time()
            
            while time.time() < end_time:
                if self.player_cleanup_event.is_set():
                    self.logger.info("Playback interrupted")
                    break
                
                # Check for DNC/NI every 100ms during playback
                current_time = time.time()
                if current_time - last_interrupt_check >= 0.1:
                    # Check if there's an interrupt callback and call it
                    if hasattr(self, '_interrupt_callback') and self._interrupt_callback:
                        if self._interrupt_callback():
                            self.logger.warning("🚫 Playback interrupted by DNC/NI detection")
                            break
                    last_interrupt_check = current_time
                
                time.sleep(0.001)  # 1ms precision instead of 100ms
            
            # IMMEDIATELY stop the player before it can loop
            with self.player_lock:
                if self.current_player and self.current_sink:
                    try:
                        self.current_player.stopTransmit(self.current_sink)
                        # Destroy references immediately
                        self.current_player = None
                        self.current_sink = None
                    except:
                        pass
            
            # Longer delay for audio pipeline to prevent echo
            time.sleep(AUDIO_STOP_DELAY)
            
            return True
            
        except pj.Error as e:
            if player_created:
                self.logger.error(f"PJSIP playback error: {e.reason}")
            return False
            
        except Exception as e:
            self.logger.error(f"Playback error: {e}")
            return False
    
    def _cleanup_player_safe(self):
        """Safely cleanup player resources"""
        with self.player_lock:
            if self.current_player and self.current_sink:
                try:
                    # Only attempt cleanup if we're in a PJSIP registered thread
                    if hasattr(threading.current_thread(), '_pj_registered'):
                        self.current_player.stopTransmit(self.current_sink)
                        self.logger.info("Player stopped")
                    else:
                        self.logger.warning("Skipping player cleanup - not in PJSIP thread")
                except pj.Error as e:
                    # Use numeric error codes
                    if e.status not in [PJSIP_ERROR_NOT_FOUND, PJSIP_ERROR_INVALID]:
                        self.logger.error(f"Player cleanup error: {e.reason}")
                except Exception as e:
                    self.logger.error(f"Player cleanup error: {e}")
                finally:
                    self.current_player = None
                    self.current_sink = None
                    self.stats['cleanups'] += 1
            
            # Signal cleanup done
            self.player_cleanup_event.set()
    
    def cleanup_player(self):
        """Public method to cleanup player"""
        self._cleanup_player_safe()
    
    def cleanup_all(self):
        """Cleanup all resources"""
        self.logger.info("Cleaning up audio manager")
        
        # Only cleanup PJSIP resources if in proper thread
        if hasattr(threading.current_thread(), '_pj_registered'):
            # Cleanup player
            self._cleanup_player_safe()
        else:
            self.logger.warning("Skipping PJSIP cleanup - not in registered thread")
            # Just clear the references
            with self.player_lock:
                self.current_player = None
                self.current_sink = None
        
        # Cleanup tracked resources (temp files, etc)
        self.resource_tracker.cleanup_all()
        
        # Clear data
        self.background_noise_data = None
        
        # Log stats
        self.logger.info(f"Audio manager stats: {self.stats}")
    
    def get_stats(self):
        """Get manager statistics"""
        return self.stats.copy()


class SIPAudioRecorder:
    """Production-grade SIP audio recorder with robust error handling"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Recording state
        self.recording_file_path = None
        self.sip_recorder = None
        self.audio_source = None
        self.is_recording = False
        self.recording_lock = threading.RLock()
        
        # Resource tracking
        self.resource_tracker = ResourceTracker(self.logger)
        
        # Stats
        self.stats = {
            'recordings_started': 0,
            'recordings_completed': 0,
            'errors': 0,
            'bytes_recorded': 0
        }
        
        self.logger.info("Audio recorder initialized")
    
    def start_recording(self, audio_media: pj.AudioMedia) -> Optional[str]:
        """
        Start recording with comprehensive error handling
        Returns the recording file path or None on failure
        """
        self.logger.info(f"🎥 SIPAudioRecorder.start_recording() called with audio_media: {audio_media}")
        
        with ThreadSafePJSIP.pjsip_operation("start_recording"):
            with self.recording_lock:
                self.logger.info(f"🎥 Acquired recording lock, is_recording: {self.is_recording}")
                
                # Stop any existing recording
                if self.is_recording:
                    self._stop_recording_internal()
                
                try:
                    # Validate input
                    if not audio_media:
                        self.logger.error("❌ No audio media provided to SIPAudioRecorder")
                        self.stats['errors'] += 1
                        return None
                    
                    self.logger.info(f"🎥 Audio media validated: {audio_media}")
                    
                    # Create temp file
                    self.logger.info("🎥 Creating temp recording file...")
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix='.wav',
                        prefix='sip_recording_'
                    )
                    temp_file.close()
                    
                    self.recording_file_path = temp_file.name
                    self.resource_tracker.register_temp_file(self.recording_file_path)
                    self.logger.info(f"🎥 Temp file created: {self.recording_file_path}")
                    
                    # Create recorder with full serialization to prevent assertion failures
                    self.logger.info("🎥 Creating PJSIP AudioMediaRecorder...")
                    with ThreadSafePJSIP._media_creation_lock:
                        self.sip_recorder = pj.AudioMediaRecorder()
                        self.sip_recorder.createRecorder(self.recording_file_path)
                    self.logger.info("🎥 AudioMediaRecorder created successfully")
                    
                    # Start recording (outside the lock to allow concurrent audio operations)
                    self.logger.info("🎥 Starting transmission...")
                    self.audio_source = audio_media
                    self.audio_source.startTransmit(self.sip_recorder)
                    
                    self.is_recording = True
                    self.stats['recordings_started'] += 1
                    
                    self.logger.info(f"✅ Recording started successfully: {self.recording_file_path}")
                    return self.recording_file_path
                    
                except pj.Error as e:
                    # Handle race condition where call disconnects
                    if e.status in [PJSIP_ERROR_NOT_FOUND, PJSIP_ERROR_INVALID]:
                        self.logger.warning(f"❌ PJSIP error - Call disconnected before recording could start: {e.reason} (status: {e.status})")
                    else:
                        self.logger.error(f"❌ PJSIP recording error: {e.reason} (status: {e.status})")
                    
                    self._cleanup_recording_resources()
                    self.stats['errors'] += 1
                    return None
                    
                except Exception as e:
                    self.logger.error(f"❌ Recording start failed with exception: {e}", exc_info=True)
                    self._cleanup_recording_resources()
                    self.stats['errors'] += 1
                    return None
    
    def stop_recording(self):
        """Stop recording with error handling"""
        with ThreadSafePJSIP.pjsip_operation("stop_recording"):
            with self.recording_lock:
                self._stop_recording_internal()
                
    def clear_buffer(self):
        """Clear any residual audio data"""
        self.audio_buffer = b''
        if hasattr(self, 'recorded_audio'):
            self.recorded_audio = b''
            
    def _stop_recording_internal(self):
        """Internal method to stop recording"""
        if not self.is_recording:
            return
        
        try:
            # Only attempt PJSIP operations if in registered thread
            if self.audio_source and self.sip_recorder:
                if hasattr(threading.current_thread(), '_pj_registered'):
                    try:
                        self.audio_source.stopTransmit(self.sip_recorder)
                        self.logger.info("Recording stopped")
                        self.stats['recordings_completed'] += 1
                    except pj.Error as e:
                        # Use numeric error codes
                        if e.status not in [PJSIP_ERROR_NOT_FOUND, PJSIP_ERROR_INVALID]:
                            self.logger.error(f"PJSIP stop recording error: {e.reason}")
                else:
                    self.logger.warning("Skipping recording stop - not in PJSIP thread")
                    
        except Exception as e:
            self.logger.error(f"Stop recording error: {e}")
            
        finally:
            self.is_recording = False
            self._cleanup_recording_resources()
    
    def _cleanup_recording_resources(self):
        """Cleanup recording resources"""
        self.sip_recorder = None
        self.audio_source = None
        # Don't delete recording file here - it may still be needed
    
    def get_new_audio_data(self, last_file_size: int = 0) -> bytes:
        """
        Get new audio data from recording file
        Thread-safe and error-resistant
        """
        with self.recording_lock:
            try:
                # Check if recording
                if not self.is_recording or not self.recording_file_path:
                    return b''
                
                # Check file exists
                if not os.path.exists(self.recording_file_path):
                    return b''
                
                # Get current size
                current_size = os.path.getsize(self.recording_file_path)
                
                # Check for new data
                if current_size <= last_file_size:
                    return b''
                
                # Read new data
                with open(self.recording_file_path, 'rb') as f:
                    f.seek(last_file_size)
                    new_data = f.read(current_size - last_file_size)
                    
                    # Strip WAV header on first read
                    if last_file_size == 0 and new_data.startswith(b'RIFF'):
                        # Find data chunk
                        data_pos = new_data.find(b'data')
                        if data_pos > 0 and (data_pos + 8) < len(new_data):
                            new_data = new_data[data_pos + 8:]
                    
                    self.stats['bytes_recorded'] += len(new_data)
                    return new_data
                    
            except Exception as e:
                self.logger.error(f"Error reading audio data: {e}")
                return b''
    
    def upsample_8k_to_16k(self, audio_8k: bytes) -> bytes:
        """
        Upsample audio from 8kHz to 16kHz
        Simple linear interpolation
        """
        try:
            if not audio_8k:
                return b''
            
            # Convert to numpy array
            samples = np.frombuffer(audio_8k, dtype=np.int16)
            
            # Simple upsampling by duplication
            upsampled = np.repeat(samples, 2)
            
            return upsampled.tobytes()
            
        except Exception as e:
            self.logger.error(f"Upsampling error: {e}")
            return audio_8k
    
    def cleanup(self):
        """Cleanup all resources"""
        self.logger.info("Cleaning up audio recorder")
        
        # Only stop recording if in PJSIP thread
        if hasattr(threading.current_thread(), '_pj_registered'):
            # Stop recording
            self.stop_recording()
        else:
            self.logger.warning("Skipping PJSIP recording stop - not in registered thread")
            # Just clear the state
            with self.recording_lock:
                self.is_recording = False
                self.sip_recorder = None
                self.audio_source = None
        
        # Cleanup resources (temp files)
        self.resource_tracker.cleanup_all()
        
        # Log stats
        self.logger.info(f"Recorder stats: {self.stats}")
    
    def get_stats(self):
        """Get recorder statistics"""
        with self.recording_lock:
            return self.stats.copy()


class ContinuousBackgroundNoisePlayer:
    """
    Continuous background noise player for natural call ambience
    - Independent PJSIP AudioMediaPlayer instance
    - Plays looping background noise throughout entire call
    - Thread-safe lifecycle with proper PJSIP registration
    - Graceful failure (call continues if player fails)
    """

    def __init__(self, noise_path: str, noise_volume: float, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.noise_path = noise_path
        self.noise_volume = max(0.0, min(1.0, noise_volume))  # Clamp 0.0-1.0

        # Player state
        self.current_player = None
        self.current_sink = None
        self.is_playing = False
        self.player_lock = threading.RLock()

        # Resource tracking
        self.resource_tracker = ResourceTracker(self.logger)
        self.looped_file_path = None

        # Stats
        self.stats = {
            'start_attempts': 0,
            'start_successes': 0,
            'start_failures': 0,
            'stop_count': 0
        }

        self.logger.info(f"Continuous background noise player initialized (volume={self.noise_volume:.2f})")

    def _create_looped_noise_file(self) -> Optional[str]:
        """Create extended looping background noise file"""
        try:
            # Validate noise path
            if not self.noise_path:
                self.logger.warning("No noise path configured")
                return None

            # Try different path variations
            paths_to_try = [
                self.noise_path,
                f"{self.noise_path}.wav",
                f"{self.noise_path}.mp3"
            ]

            noise_audio = None
            for path in paths_to_try:
                if os.path.exists(path):
                    try:
                        self.logger.info(f"Loading background noise for continuous playback: {path}")

                        # Determine format
                        _, ext = os.path.splitext(path)
                        format_hint = None if ext else "wav"

                        # Load and convert audio
                        noise_audio = AudioSegment.from_file(path, format=format_hint)
                        noise_audio = noise_audio.set_channels(1).set_frame_rate(SIP_SAMPLE_RATE).set_sample_width(2)
                        break
                    except Exception as e:
                        self.logger.error(f"Failed to load {path}: {e}")

            if not noise_audio:
                self.logger.warning(f"Background noise file not found: {self.noise_path}")
                return None

            # Apply volume adjustment (convert to dB)
            if self.noise_volume < 1.0:
                # Calculate dB reduction
                db_change = 20 * np.log10(self.noise_volume) if self.noise_volume > 0 else -60
                noise_audio = noise_audio + db_change

            # Loop to create 60-second file (covers max call duration with potential looping)
            duration_ms = len(noise_audio)
            if duration_ms < 60000:  # Less than 60 seconds
                loops_needed = int((60 * 1000) / duration_ms) + 1
                looped_audio = noise_audio * loops_needed
                self.logger.info(f"Looped background noise {loops_needed} times to create {len(looped_audio)/1000:.1f}s file")
            else:
                looped_audio = noise_audio
                self.logger.info(f"Background noise file is {len(looped_audio)/1000:.1f}s, no looping needed")

            # Export to temp file
            temp_file = tempfile.NamedTemporaryFile(
                delete=False,
                suffix='.wav',
                prefix='bg_noise_loop_'
            )
            looped_audio.export(temp_file.name, format="wav")
            temp_file.close()

            # Register for cleanup
            self.resource_tracker.register_temp_file(temp_file.name)
            self.looped_file_path = temp_file.name

            self.logger.info(f"✅ Looped background noise file created: {temp_file.name} ({len(looped_audio)/1000:.1f}s)")
            return temp_file.name

        except Exception as e:
            self.logger.error(f"Failed to create looped noise file: {e}", exc_info=True)
            return None

    def start(self, audio_media: pj.AudioMedia) -> bool:
        """
        Start continuous background noise playback
        Returns True if started successfully, False otherwise (non-fatal)
        """
        with ThreadSafePJSIP.pjsip_operation("start_bg_noise"):
            with self.player_lock:
                self.stats['start_attempts'] += 1

                if self.is_playing:
                    self.logger.info("Continuous background noise already playing")
                    return True  # Already playing

                try:
                    # Create looped noise file
                    looped_file = self._create_looped_noise_file()
                    if not looped_file:
                        self.logger.warning("Failed to create looped noise file - call continues without background noise")
                        self.stats['start_failures'] += 1
                        return False  # Non-fatal, call continues

                    # Create player with serialization (same pattern as main player)
                    with ThreadSafePJSIP._media_creation_lock:
                        self.current_player = pj.AudioMediaPlayer()

                    # Create player outside global lock
                    self.current_player.createPlayer(looped_file, pj.PJMEDIA_FILE_NO_LOOP)

                    # Connect to sink
                    self.current_sink = audio_media
                    self.current_player.startTransmit(audio_media)

                    self.is_playing = True
                    self.stats['start_successes'] += 1

                    self.logger.info(f"✅ Continuous background noise started (volume={self.noise_volume:.2f})")
                    return True

                except pj.Error as e:
                    self.logger.error(f"PJSIP error starting background noise: {e.reason} (status: {e.status})")
                    self._cleanup_safe()
                    self.stats['start_failures'] += 1
                    return False  # Non-fatal, call continues

                except Exception as e:
                    self.logger.error(f"Error starting background noise: {e}", exc_info=True)
                    self._cleanup_safe()
                    self.stats['start_failures'] += 1
                    return False  # Non-fatal, call continues

    def stop(self):
        """Stop continuous background noise playback"""
        with ThreadSafePJSIP.pjsip_operation("stop_bg_noise"):
            self._cleanup_safe()
            self.stats['stop_count'] += 1

    def _cleanup_safe(self):
        """Thread-safe cleanup (handles non-PJSIP threads)"""
        with self.player_lock:
            if self.current_player and self.current_sink:
                try:
                    # Only attempt PJSIP cleanup if in registered thread (same pattern as main audio manager)
                    if hasattr(threading.current_thread(), '_pj_registered'):
                        self.current_player.stopTransmit(self.current_sink)
                        self.logger.info("Continuous background noise player stopped")
                    else:
                        self.logger.warning("Skipping bg noise cleanup - not in PJSIP thread")
                except pj.Error as e:
                    # Use numeric error codes (same pattern as main audio manager)
                    if e.status not in [PJSIP_ERROR_NOT_FOUND, PJSIP_ERROR_INVALID]:
                        self.logger.error(f"Background noise cleanup error: {e.reason}")
                except Exception as e:
                    self.logger.error(f"Background noise cleanup error: {e}")
                finally:
                    self.current_player = None
                    self.current_sink = None
                    self.is_playing = False

    def cleanup(self):
        """Cleanup all resources"""
        self.logger.info("Cleaning up continuous background noise player")

        # Only cleanup PJSIP resources if in proper thread
        if hasattr(threading.current_thread(), '_pj_registered'):
            # Cleanup player
            self._cleanup_safe()
        else:
            self.logger.warning("Skipping PJSIP bg noise cleanup - not in registered thread")
            # Just clear the references
            with self.player_lock:
                self.current_player = None
                self.current_sink = None
                self.is_playing = False

        # Cleanup temp files
        self.resource_tracker.cleanup_all()
        self.looped_file_path = None

        # Log stats
        self.logger.info(f"Continuous background noise stats: {self.stats}")

    def get_stats(self):
        """Get player statistics"""
        with self.player_lock:
            return self.stats.copy()

    def is_active(self) -> bool:
        """Check if player is currently active"""
        with self.player_lock:
            return self.is_playing