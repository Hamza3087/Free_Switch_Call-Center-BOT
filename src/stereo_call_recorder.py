#!/usr/bin/env python3
"""
Stereo Call Recorder for SIP Bot
Creates stereo recordings with user audio in left channel and bot audio in right channel
"""

import os
import time
import tempfile
import threading
import numpy as np
import wave
import logging
from typing import Optional, List, Tuple
from pydub import AudioSegment

# Import existing audio recorder
from .sip_audio_manager import SIPAudioRecorder
import pjsua2 as pj

class StereoCallRecorder:
    """
    Records stereo call audio with user in left channel and bot in right channel
    """
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Recording state
        self.is_recording = False
        self.call_start_time = None
        self.recording_lock = threading.RLock()
        
        # User audio recording (shared from external source)
        # Note: No internal SIPAudioRecorder to prevent PJSIP conflicts
        self.user_recording_file = None
        
        # Bot audio storage (timestamped chunks)
        self.bot_audio_chunks = []  # List of (timestamp, audio_data) tuples
        self.bot_audio_lock = threading.Lock()
        
        # Output configuration
        self.sample_rate = 8000  # SIP audio is 8kHz
        self.channels = 2  # Stereo
        self.sample_width = 2  # 16-bit
        
        # Stats
        self.stats = {
            'recordings_started': 0,
            'user_bytes_recorded': 0,
            'bot_chunks_recorded': 0,
            'stereo_files_created': 0,
            'errors': 0
        }
        
        self.logger.info("Stereo call recorder initialized")
    
    def start_recording(self, audio_media: pj.AudioMedia) -> bool:
        """
        DISABLED: This method is disabled to prevent PJSIP conflicts.
        Use use_existing_recording() instead to share recordings with other components.
        """
        self.logger.warning("🎙️ StereoCallRecorder.start_recording() is DISABLED to prevent PJSIP deadlock")
        self.logger.info("Use use_existing_recording() method instead to share recordings")
        return False

    def use_existing_recording(self, recording_file_path: str) -> bool:
        """
        Use an existing recording file instead of starting our own recording
        This is useful when sharing a recording file with another component
        """
        self.logger.info(f"🎙️ StereoCallRecorder.use_existing_recording() called with: {recording_file_path}")
        
        with self.recording_lock:
            if self.is_recording:
                self.logger.warning("Stereo recording already active")
                return False
                
            try:
                # Validate the existing recording file
                if not recording_file_path or not os.path.exists(recording_file_path):
                    self.logger.error(f"❌ Recording file does not exist: {recording_file_path}")
                    return False
                    
                # Record call start time
                self.call_start_time = time.time()
                self.logger.info(f"🎙️ Call start time recorded: {self.call_start_time}")
                
                # Use the existing recording file
                self.user_recording_file = recording_file_path
                self.logger.info(f"✅ Using existing recording file: {self.user_recording_file}")
                
                # Clear bot audio storage
                with self.bot_audio_lock:
                    self.bot_audio_chunks.clear()
                    self.logger.info("🎙️ Bot audio storage cleared")
                
                # Mark as recording (even though we're using existing file)
                self.is_recording = True
                self.stats['recordings_started'] += 1
                
                self.logger.info(f"✅ Stereo recording configured successfully - Using file: {self.user_recording_file}")
                return True
                
            except Exception as e:
                self.logger.error(f"❌ Exception in use_existing_recording(): {e}", exc_info=True)
                self.stats['errors'] += 1
                return False
    
    def add_bot_audio(self, audio_data: bytes, timestamp: Optional[float] = None) -> bool:
        """
        Add bot audio chunk with timestamp
        audio_data: raw audio bytes (8kHz, 16-bit, mono)
        timestamp: when this audio was played (defaults to current time)
        """
        if not self.is_recording:
            self.logger.debug("🎵 Bot audio received but stereo recording not active")
            return False
        
        try:
            if timestamp is None:
                timestamp = time.time()
            
            # Calculate relative timestamp from call start
            if self.call_start_time:
                relative_timestamp = timestamp - self.call_start_time
            else:
                relative_timestamp = 0.0
            
            with self.bot_audio_lock:
                self.bot_audio_chunks.append((relative_timestamp, audio_data))
                self.stats['bot_chunks_recorded'] += 1
            
            self.logger.info(f"🎵 Bot audio chunk added: {len(audio_data)} bytes at {relative_timestamp:.2f}s (total chunks: {self.stats['bot_chunks_recorded']})")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to add bot audio: {e}")
            self.stats['errors'] += 1
            return False
    
    def stop_recording(self):
        """Stop stereo recording (no-op since we don't manage recordings directly)"""
        with self.recording_lock:
            if not self.is_recording:
                return
            
            # No need to stop external recordings - that's managed by the owner
            self.is_recording = False
            self.logger.info("🎙️ Stereo recording marked as stopped")
    
    def save_stereo_recording(self, output_path: str) -> bool:
        """
        Save stereo recording to file
        Left channel: User audio
        Right channel: Bot audio
        """
        try:
            if self.is_recording:
                self.stop_recording()
            
            # Ensure we have user recording
            if not self.user_recording_file:
                self.logger.error("No user recording file path set")
                self.logger.info("This usually means start_recording() or use_existing_recording() was never called")
                return False
            
            if not os.path.exists(self.user_recording_file):
                self.logger.error(f"User recording file does not exist: {self.user_recording_file}")
                self.logger.info("The recording file may have been moved or deleted")
                return False
            
            # Check if file is empty or too small
            file_size = os.path.getsize(self.user_recording_file)
            if file_size < 1024:  # Less than 1KB suggests no real audio
                self.logger.warning(f"User recording file is very small ({file_size} bytes), may be empty")
                # Continue anyway, but warn
            
            self.logger.info(f"Creating stereo recording: {output_path}")
            
            # Load user audio (left channel)
            user_audio = self._load_user_audio()
            if user_audio is None:
                self.logger.error("Failed to load user audio")
                return False
            
            # Create bot audio track (right channel)
            bot_audio = self._create_bot_audio_track(len(user_audio))
            
            # Ensure both channels have same length
            min_length = min(len(user_audio), len(bot_audio))
            user_audio = user_audio[:min_length]
            bot_audio = bot_audio[:min_length]
            
            # Create stereo audio
            stereo_audio = self._combine_to_stereo(user_audio, bot_audio)
            
            # Save stereo WAV file
            success = self._save_stereo_wav(stereo_audio, output_path)
            
            if success:
                self.stats['stereo_files_created'] += 1
                self.logger.info(f"✅ Stereo recording saved: {output_path} ({len(stereo_audio)} samples)")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to save stereo recording: {e}", exc_info=True)
            self.stats['errors'] += 1
            return False
    
    def _load_user_audio(self) -> Optional[np.ndarray]:
        """Load user audio from recording file"""
        try:
            # Load the recorded WAV file
            with wave.open(self.user_recording_file, 'rb') as wav_file:
                frames = wav_file.readframes(wav_file.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16)
            
            self.stats['user_bytes_recorded'] = len(frames)
            self.logger.debug(f"Loaded user audio: {len(audio_data)} samples")
            return audio_data
            
        except Exception as e:
            self.logger.error(f"Failed to load user audio: {e}")
            return None
    
    def _create_bot_audio_track(self, target_length: int) -> np.ndarray:
        """Create bot audio track with same length as user audio"""
        try:
            # Start with silence
            bot_track = np.zeros(target_length, dtype=np.int16)
            
            with self.bot_audio_lock:
                # Insert bot audio chunks at correct positions
                for timestamp, audio_data in self.bot_audio_chunks:
                    try:
                        # Convert timestamp to sample position
                        sample_position = int(timestamp * self.sample_rate)
                        
                        # Convert audio data to samples
                        audio_samples = np.frombuffer(audio_data, dtype=np.int16)
                        
                        # Calculate end position
                        end_position = sample_position + len(audio_samples)
                        
                        # Ensure we don't exceed track length
                        if sample_position < target_length:
                            actual_end = min(end_position, target_length)
                            actual_samples = actual_end - sample_position
                            
                            # Insert audio samples
                            bot_track[sample_position:actual_end] = audio_samples[:actual_samples]
                            
                            self.logger.debug(f"Inserted bot audio: {actual_samples} samples at {timestamp:.2f}s")
                    
                    except Exception as e:
                        self.logger.warning(f"Failed to insert bot audio chunk: {e}")
                        continue
            
            self.logger.debug(f"Created bot audio track: {len(bot_track)} samples")
            return bot_track
            
        except Exception as e:
            self.logger.error(f"Failed to create bot audio track: {e}")
            return np.zeros(target_length, dtype=np.int16)
    
    def _combine_to_stereo(self, left_channel: np.ndarray, right_channel: np.ndarray) -> np.ndarray:
        """Combine mono channels into stereo"""
        try:
            # Ensure same length
            length = min(len(left_channel), len(right_channel))
            left = left_channel[:length]
            right = right_channel[:length]
            
            # Interleave samples (LRLRLR...)
            stereo = np.empty((length * 2,), dtype=np.int16)
            stereo[0::2] = left   # Left channel (user)
            stereo[1::2] = right  # Right channel (bot)
            
            return stereo
            
        except Exception as e:
            self.logger.error(f"Failed to combine to stereo: {e}")
            return np.array([], dtype=np.int16)
    
    def _save_stereo_wav(self, stereo_audio: np.ndarray, output_path: str) -> bool:
        """Save stereo audio as WAV file"""
        try:
            with wave.open(output_path, 'wb') as wav_file:
                wav_file.setnchannels(2)  # Stereo
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)  # 8kHz
                wav_file.writeframes(stereo_audio.tobytes())
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save stereo WAV: {e}")
            return False
    
    def cleanup(self):
        """Cleanup all resources"""
        self.logger.info("Cleaning up stereo call recorder")
        
        # Stop recording if active
        if self.is_recording:
            self.stop_recording()
        
        # Clear bot audio chunks
        with self.bot_audio_lock:
            self.bot_audio_chunks.clear()
        
        # Clear user recording reference
        self.user_recording_file = None
        
        # Log stats
        self.logger.info(f"Stereo recorder stats: {self.stats}")
    
    def get_stats(self) -> dict:
        """Get recorder statistics"""
        with self.recording_lock:
            return self.stats.copy()