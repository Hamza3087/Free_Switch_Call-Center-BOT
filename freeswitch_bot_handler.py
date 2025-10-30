#!/usr/bin/env python3
"""
FreeSWITCH Bot Handler
- Core call handling logic (replaces SuiteCRMBotInstance)
- Plain Python class (no pjsua2 inheritance)
- FreeSWITCH handles SIP/RTP/media
- Python handles business logic only
"""

import os
import time
import tempfile
import logging
import threading
import uuid
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import ESL
except ImportError:
    print("ERROR: ESL module not installed")
    raise

# Import business logic components (all reused from existing codebase)
from src.call_flow import parse_call_flow_from_string, get_audio_path_for_agent
from src.intent_detector import IntentDetector
from src.suitecrm_integration import SuiteCRMAgentConfig, SuiteCRMLogger
from src.call_outcome_handler import CallOutcomeHandler, CallOutcome
from src.parakeet_rnnt import ParakeetRNNTModel

# Import continuous transcription handler
from src.continuous_transcription_handler import ContinuousTranscriptionHandler

# Import Silero VAD for speech detection in regular transcription
from src.silero_vad import SileroVAD

# Ringing detection components (framework-agnostic)
from src.ringing_detector_core import (
    GoertzelDetector,
    RingCycleTracker,
    DetectionValidator
)
import numpy as np

# Import model singletons (standalone modules)
from src.parakeet_singleton import ParakeetModelSingleton
try:
    from src.qwen_singleton import QwenModelSingleton
except ImportError:
    QwenModelSingleton = None

# Configuration
SUITECRM_UPLOAD_DIR = "/var/www/recordings"


class FreeSWITCHBotHandler:
    """
    Core call handler for FreeSWITCH ESL
    Replaces the complex pjsua2 SuiteCRMBotInstance with clean ESL-based logic
    """

    def __init__(self, conn: ESL.ESLconnection, agent_config: SuiteCRMAgentConfig, info: ESL.ESLevent):
        """
        Initialize bot handler

        Args:
            conn: ESL connection object
            agent_config: Agent configuration from database
            info: Call information from FreeSWITCH
        """
        self.conn = conn
        self.agent_config = agent_config
        self.info = info

        # Extract call information
        self.uuid = info.getHeader("Unique-ID")
        self.phone_number = info.getHeader("Caller-Caller-ID-Number") or "UNKNOWN"
        self.destination = info.getHeader("Caller-Destination-Number")
        self.caller_state = None  # Can be extracted from area code if needed

        # Setup logging
        self.logger = logging.getLogger(f"Call-{self.uuid[:8]}")
        self.logger.setLevel(logging.DEBUG)  # Ensure all log levels are captured

        # Call state
        self.is_active = True
        self.call_start_time = time.time()
        self.current_step = None
        self.call_flow = None

        # Tracking
        self.consecutive_silence_count = 0
        self.clarification_count = 0
        self.conversation_log = []

        # Configuration from agent
        self.max_consecutive_silences = agent_config.max_silence_retries
        self.max_clarifications = agent_config.max_clarification_retries

        # Initialize call data
        self.call_data = self._initialize_call_data()

        # Initialize components
        self.intent_detector = None
        self.parakeet_model = None
        self.crm_logger = None
        self.outcome_handler = None
        self.continuous_transcription = None  # Continuous listening handler
        self.silero_vad = None  # VAD for regular response transcription

        # Ringing detection (uuid_record-based)
        self.ringing_detected = False
        self.recording_file = None
        self.detection_thread = None
        self.detection_stop_event = threading.Event()

        # Goertzel detection components
        self.goertzel = None
        self.cycle_tracker = None
        self.validator = None

        # Voicemail detection (VMD) components
        self.vmd_enabled = False
        self.vmd_classifier = None
        self.vmd_start_time = None
        self.vmd_complete = False
        self.voicemail_detected = False
        self.vmd_recording_buffer = bytearray()
        self.vmd_confidence = 0.0
        self.vmd_detection_duration = 7.0

        # Playback state tracking for continuous transcription
        self.is_playing_audio = False
        self.playback_start_time = None
        self.playback_periods = []  # Track (start_time, end_time) tuples for race condition fix

        # Main response listening flag (continuous transcription runs when this is False)
        self.main_response_listening = False

        self.logger.info(f"Handler created for {self.phone_number}")

    def _initialize_call_data(self) -> Dict[str, Any]:
        """Initialize call data structure"""
        return {
            'id': None,
            'phone_number': self.phone_number,
            'caller_state': self.caller_state,
            'start_time': self.call_start_time,
            'end_time': None,
            'disposition': 'INITIATED',
            'transcript': '',
            'is_voicemail': False,
            'intent_detected': None,
            'uniqueid': f"voicebot_{self.uuid}_{int(self.call_start_time)}",
            'duration': 0,
            'call_result': 'UNKNOWN',
            'originating_agent': self.agent_config.agent_id,
            'e_agent_id': self.agent_config.agent_id,
            'campaign_id': self.agent_config.campaign_id,
            'vici_lead_id': None,
            'vici_list_id': None,
            'filename': None,
            'file_mime_type': None,
            'call_drop_step': None,
            'error': None,
            'transfer_target': None,
            'transfer_status': None,
            'transfer_response_code': None,
            'transfer_timestamp': None,
            'transfer_reason': None
        }

    def _start_audio_detection(self):
        """Start unified audio detection (ringing + voicemail) using uuid_record"""
        from src.config import VMD_ENABLED, VMD_DETECTION_DURATION, VMD_CONFIDENCE_THRESHOLD, VMD_MODEL_PATH

        try:
            # Create recording file path (shared by both detections)
            self.recording_file = f"/usr/local/freeswitch/recordings/detect_{self.uuid}_{int(time.time())}.wav"

            # Enable stereo recording (LEFT=caller, RIGHT=bot)
            self.conn.execute("set", "RECORD_STEREO=true")
            self.logger.info("Enabled stereo recording mode")

            # Start FreeSwitch recording (will now be stereo)
            cmd = f"uuid_record {self.uuid} start {self.recording_file}"
            result = self.conn.api(cmd)

            if result:
                response = result.getBody()
                self.logger.info(f"Started STEREO recording: {response}")
                self.logger.info("  LEFT channel = Caller (inbound)")
                self.logger.info("  RIGHT channel = Bot (outbound)")
            else:
                self.logger.error("Failed to start recording")
                return

            # Initialize ringing detection components
            self.goertzel = GoertzelDetector(sample_rate=8000, chunk_size=1024)
            self.cycle_tracker = RingCycleTracker(required_rings=2)
            self.validator = DetectionValidator(
                relative_threshold=3.0,         # Lowered from 7.0 to 3.0
                max_strength_threshold=200.0,   # Raised from 100.0 to 200.0
                frequency_balance_ratio=8.0,    # Raised from 6.5 to 8.0
                min_energy=3e4,                 # Lowered from 1e5 to 3e4 (30,000)
                required_consecutive=2
            )

            # Initialize voicemail detection components (NEW)
            if VMD_ENABLED:
                from src.voicemail_detector_core import VoicemailClassifier
                self.vmd_enabled = True
                self.vmd_classifier = VoicemailClassifier(
                    model_path=VMD_MODEL_PATH,
                    confidence_threshold=VMD_CONFIDENCE_THRESHOLD,
                    logger=self.logger
                )
                self.vmd_start_time = time.time()
                self.vmd_detection_duration = VMD_DETECTION_DURATION
                self.vmd_recording_buffer = bytearray()
                self.logger.info(
                    f"VMD enabled (duration={VMD_DETECTION_DURATION}s, "
                    f"threshold={VMD_CONFIDENCE_THRESHOLD})"
                )

            # Start unified detection thread
            self.detection_thread = threading.Thread(
                target=self._unified_detection_loop,
                daemon=True
            )
            self.detection_thread.start()

            detection_types = []
            if True:  # Ringing always enabled
                detection_types.append("ringing")
            if self.vmd_enabled:
                detection_types.append("voicemail")

            self.logger.info(f"✅ Audio detection started: {', '.join(detection_types)}")

        except Exception as e:
            self.logger.error(f"Failed to start audio detection: {e}", exc_info=True)

    def _unified_detection_loop(self):
        """
        Unified detection loop - monitors for BOTH ringback tones AND voicemail
        Runs in background thread, analyzes same recording file in parallel
        """
        last_position = 44  # Skip WAV header
        detection_start_time = time.time()
        chunks_analyzed = 0

        try:
            # Wait for file to be created
            wait_count = 0
            while not os.path.exists(self.recording_file) and wait_count < 40:
                time.sleep(0.05)
                wait_count += 1

            if not os.path.exists(self.recording_file):
                self.logger.error("Recording file not created")
                return

            self.logger.info(f"Analyzing recording: {self.recording_file}")

            while self.is_active and not self.detection_stop_event.is_set():
                try:
                    current_size = os.path.getsize(self.recording_file)

                    if current_size > last_position:
                        # Read new audio data (stereo from uuid_record)
                        with open(self.recording_file, 'rb') as f:
                            f.seek(last_position)
                            new_audio_stereo = f.read(current_size - last_position)

                        if len(new_audio_stereo) > 0:
                            # Extract LEFT channel only (caller audio, not bot audio)
                            new_audio = self._extract_left_channel_from_stereo(new_audio_stereo)

                            # === PARALLEL DETECTION 1: RINGING (Left channel only) ===
                            self._detect_ringing(new_audio, chunks_analyzed)
                            if self.ringing_detected:
                                return  # Exit if ringing found

                            # === PARALLEL DETECTION 2: VOICEMAIL (Left channel only) ===
                            if self.vmd_enabled and not self.vmd_complete:
                                self._detect_voicemail(new_audio)
                                if self.voicemail_detected:
                                    return  # Exit if voicemail found

                            # === PARALLEL DETECTION 3: CONTINUOUS TRANSCRIPTION (always-on except during main response) ===
                            # Continuous transcription runs 95% of call time (only paused during main response recording)
                            if self.continuous_transcription:
                                # Check if main response listener is active
                                is_continuous_active = not self.main_response_listening

                                # Diagnostic logging (only log every 20 chunks to avoid spam)
                                if chunks_analyzed % 20 == 0:
                                    if is_continuous_active:
                                        self.logger.debug(f"[CONTINUOUS TRANSCRIPTION] Active (main_response_listening={self.main_response_listening}) - processing audio")
                                    else:
                                        self.logger.debug(f"[CONTINUOUS TRANSCRIPTION] Paused during main response recording")

                                if is_continuous_active:
                                    # Add audio chunk to transcription buffer (also updates VAD state)
                                    self.continuous_transcription.add_audio_chunk(new_audio)

                                    # Check for speech-silence patterns and transcribe when appropriate
                                    # The handler now uses Silero VAD and speech-silence detection
                                    # to decide when to transcribe (not fixed intervals)
                                    self.continuous_transcription.transcribe_and_check_intents()

                            chunks_analyzed += 1
                            last_position = current_size

                    # Sleep briefly to avoid spinning
                    time.sleep(0.03)  # 30ms intervals (matches pjsua2 for faster detection)

                except Exception as e:
                    self.logger.error(f"Error in detection loop: {e}", exc_info=True)
                    time.sleep(0.1)

        except Exception as e:
            self.logger.error(f"Fatal error in unified detection: {e}", exc_info=True)
        finally:
            detection_summary = []
            if self.ringing_detected:
                detection_summary.append("ringing=YES")
            if self.voicemail_detected:
                detection_summary.append(f"voicemail=YES (conf={self.vmd_confidence:.2f})")
            if not self.ringing_detected and not self.voicemail_detected:
                detection_summary.append("no detection")

            self.logger.info(
                f"Unified detection stopped. Analyzed {chunks_analyzed} chunks. "
                f"Results: {', '.join(detection_summary)}"
            )

    def _detect_ringing(self, audio_bytes: bytes, chunk_num: int):
        """
        Detect ringback tones (440Hz/480Hz) - continuous analysis
        Extracted from original ringing detection loop for parallel execution
        """
        try:
            # Convert to numpy array
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)

            # Analyze in 1024-sample chunks
            chunk_size = 1024
            for i in range(0, len(audio_array), chunk_size):
                chunk = audio_array[i:i+chunk_size]

                if len(chunk) < chunk_size // 2:
                    continue

                # Detect frequencies (440Hz, 480Hz for US ringback)
                energy_440 = self.goertzel.detect_frequency(chunk, 440.0)
                energy_480 = self.goertzel.detect_frequency(chunk, 480.0)

                # Calculate background energy
                bg_energies = [
                    self.goertzel.detect_frequency(chunk, freq)
                    for freq in [300, 350, 400, 500, 550, 600]
                ]
                avg_bg = np.mean(bg_energies) + 1e-9

                # Relative strength
                min_target = min(energy_440, energy_480)
                relative_strength = min_target / avg_bg

                # Validate detection
                is_ringing = self.validator.validate(
                    energy_440, energy_480, relative_strength, chunk
                )

                # Debug logging every 20 chunks (avoid spam)
                if chunk_num % 20 == 0:
                    self.logger.info(
                        f"[RINGING] Chunk {chunk_num}: "
                        f"440Hz={energy_440:.0f}, 480Hz={energy_480:.0f}, "
                        f"BG={avg_bg:.0f}, Ratio={relative_strength:.2f}, "
                        f"Detected={is_ringing}"
                    )

                # Track ring cycles
                current_time = time.time()
                if self.cycle_tracker.update(is_ringing, current_time):
                    # Ringing detected!
                    self._handle_ringback_detected()
                    return

        except Exception as e:
            self.logger.error(f"Error in ringing detection: {e}")

    def _detect_voicemail(self, audio_bytes: bytes):
        """
        Detect voicemail - accumulate audio for duration, then classify with ML
        Only runs during the detection period (first 7 seconds)
        """
        from src.config import VMD_END_CALL_ON_DETECTION

        try:
            # === DIAGNOSTIC: Track audio chunk size ===
            chunk_size = len(audio_bytes)
            buffer_before = len(self.vmd_recording_buffer)

            # Accumulate audio for VMD analysis
            self.vmd_recording_buffer.extend(audio_bytes)

            buffer_after = len(self.vmd_recording_buffer)

            # === DIAGNOSTIC: Verify buffer growth ===
            if buffer_after != buffer_before + chunk_size:
                self.logger.warning(
                    f"[VMD DIAGNOSTIC] Buffer growth mismatch: "
                    f"expected {buffer_before + chunk_size}, got {buffer_after}"
                )

            # Check if detection period is complete
            elapsed = time.time() - self.vmd_start_time

            # === DIAGNOSTIC: Calculate expected buffer size for 7s at 8kHz 16-bit ===
            expected_buffer_size = int(elapsed * 8000 * 2)  # 8kHz, 16-bit (2 bytes)
            buffer_ratio = buffer_after / expected_buffer_size if expected_buffer_size > 0 else 0

            if elapsed >= self.vmd_detection_duration:
                if not self.vmd_complete:
                    self.vmd_complete = True

                    # === DIAGNOSTIC: Detailed completion logging ===
                    expected_7s_buffer_mono = int(7.0 * 8000 * 2)  # 112,000 bytes (mono)
                    expected_7s_buffer_stereo = expected_7s_buffer_mono * 2  # 224,000 bytes (stereo)

                    # Get stereo file size for comparison
                    try:
                        stereo_file_size = os.path.getsize(self.recording_file) if os.path.exists(self.recording_file) else 0
                    except:
                        stereo_file_size = 0

                    self.logger.info(
                        f"[VMD DIAGNOSTIC] Detection period complete:\n"
                        f"  Elapsed time: {elapsed:.3f}s (target: {self.vmd_detection_duration}s)\n"
                        f"  Stereo file size: {stereo_file_size:,} bytes\n"
                        f"  Mono buffer size (LEFT channel): {buffer_after:,} bytes\n"
                        f"  Expected mono: {expected_7s_buffer_mono:,} bytes (7s at 8kHz 16-bit)\n"
                        f"  Buffer ratio: {buffer_ratio:.2f}x\n"
                        f"  Audio duration: {buffer_after / (8000 * 2):.2f}s\n"
                        f"  Note: Analyzing LEFT channel only (caller audio, not bot)"
                    )

                    # === DIAGNOSTIC: Warn if buffer size is unexpected ===
                    if buffer_ratio < 0.8:
                        self.logger.warning(
                            f"[VMD DIAGNOSTIC] ⚠️  Buffer undersized! "
                            f"Only {buffer_ratio:.1%} of expected audio collected. "
                            f"This may cause false positives!"
                        )
                    elif buffer_ratio > 1.2:
                        self.logger.warning(
                            f"[VMD DIAGNOSTIC] ⚠️  Buffer oversized! "
                            f"{buffer_ratio:.1%} of expected audio collected."
                        )

                    # Run ML classification
                    is_voicemail, confidence = self._classify_voicemail()

                    if is_voicemail:
                        self.voicemail_detected = True
                        self.vmd_confidence = confidence
                        self.logger.warning(
                            f"🤖 VOICEMAIL DETECTED (confidence: {confidence:.2f})"
                        )

                        # Handle detection
                        if VMD_END_CALL_ON_DETECTION:
                            self._handle_voicemail_detected()
                    else:
                        self.logger.info(
                            f"👤 LIVE PERSON detected (confidence: {confidence:.2f})"
                        )
            else:
                # === DIAGNOSTIC: Enhanced progress logging (every 1 second) ===
                if int(elapsed * 10) % 10 == 0 and int(elapsed * 10) != int((elapsed - 0.05) * 10):
                    remaining = self.vmd_detection_duration - elapsed
                    audio_duration = buffer_after / (8000 * 2)
                    self.logger.info(
                        f"[VMD] Recording: {elapsed:.1f}s/{self.vmd_detection_duration}s "
                        f"| Buffer: {buffer_after:,} bytes ({audio_duration:.1f}s audio) "
                        f"| Ratio: {buffer_ratio:.2f}x"
                    )

        except Exception as e:
            self.logger.error(f"[VMD DIAGNOSTIC] Error in VMD detection: {e}", exc_info=True)

    def _classify_voicemail(self) -> tuple:
        """
        Classify accumulated audio as voicemail or live person
        Returns: (is_voicemail: bool, confidence: float)
        """
        from src.config import VMD_MIN_AUDIO_LENGTH, VMD_CONFIDENCE_THRESHOLD
        import numpy as np

        try:
            # === DIAGNOSTIC: Pre-classification validation ===
            buffer_size = len(self.vmd_recording_buffer)
            audio_duration = buffer_size / (8000 * 2)  # 8kHz, 16-bit

            self.logger.info(
                f"[VMD DIAGNOSTIC] Pre-classification validation:\n"
                f"  Buffer size: {buffer_size:,} bytes\n"
                f"  Audio duration: {audio_duration:.2f}s\n"
                f"  Minimum required: {VMD_MIN_AUDIO_LENGTH:,} bytes\n"
                f"  Confidence threshold: {VMD_CONFIDENCE_THRESHOLD}"
            )

            # Check minimum audio length
            if buffer_size < VMD_MIN_AUDIO_LENGTH:
                self.logger.warning(
                    f"[VMD DIAGNOSTIC] ❌ Insufficient audio for VMD "
                    f"({buffer_size:,} < {VMD_MIN_AUDIO_LENGTH:,} bytes)"
                )
                return False, 0.0

            # === DIAGNOSTIC: Audio quality validation ===
            try:
                audio_array = np.frombuffer(self.vmd_recording_buffer, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_array.astype(float)**2))
                peak = np.max(np.abs(audio_array))

                self.logger.info(
                    f"[VMD DIAGNOSTIC] Audio quality metrics:\n"
                    f"  Samples: {len(audio_array):,}\n"
                    f"  RMS energy: {rms:.1f}\n"
                    f"  Peak amplitude: {peak:.0f}\n"
                    f"  Duration check: {len(audio_array) / 8000:.2f}s"
                )

                # Warn about audio quality issues
                if rms < 100:
                    self.logger.warning(
                        f"[VMD DIAGNOSTIC] ⚠️  Very low audio energy (RMS={rms:.1f}). "
                        f"Audio may be silent or nearly silent."
                    )
                if peak < 500:
                    self.logger.warning(
                        f"[VMD DIAGNOSTIC] ⚠️  Very low peak amplitude (peak={peak:.0f}). "
                        f"Audio may be too quiet for reliable classification."
                    )

            except Exception as e:
                self.logger.warning(f"[VMD DIAGNOSTIC] Could not analyze audio quality: {e}")

            # === DIAGNOSTIC: Timing verification ===
            elapsed_since_start = time.time() - self.vmd_start_time
            self.logger.info(
                f"[VMD DIAGNOSTIC] Classification timing:\n"
                f"  Time since VMD start: {elapsed_since_start:.3f}s\n"
                f"  Target duration: {self.vmd_detection_duration}s\n"
                f"  Classification triggered: {'ON TIME' if elapsed_since_start >= self.vmd_detection_duration else 'EARLY!'}"
            )

            if elapsed_since_start < self.vmd_detection_duration:
                self.logger.error(
                    f"[VMD DIAGNOSTIC] 🚨 CRITICAL: Classification triggered EARLY! "
                    f"({elapsed_since_start:.2f}s < {self.vmd_detection_duration}s). "
                    f"This WILL cause false positives!"
                )

            # Convert to bytes
            audio_bytes = bytes(self.vmd_recording_buffer)

            self.logger.info(f"[VMD] Running ML classification on {len(audio_bytes):,} bytes...")

            # Use classifier (time the inference)
            classification_start = time.time()
            is_voicemail, confidence = self.vmd_classifier.classify_audio(audio_bytes)
            classification_time = time.time() - classification_start

            # === DIAGNOSTIC: Classification results ===
            self.logger.info(
                f"[VMD DIAGNOSTIC] Classification complete:\n"
                f"  Result: {'VOICEMAIL' if is_voicemail else 'LIVE PERSON'}\n"
                f"  Confidence: {confidence:.3f}\n"
                f"  Threshold: {VMD_CONFIDENCE_THRESHOLD}\n"
                f"  Decision: {'ABOVE threshold' if confidence >= VMD_CONFIDENCE_THRESHOLD else 'BELOW threshold'}\n"
                f"  Inference time: {classification_time:.2f}s"
            )

            return is_voicemail, confidence

        except Exception as e:
            self.logger.error(f"[VMD DIAGNOSTIC] VMD classification error: {e}", exc_info=True)
            return False, 0.0

    def _handle_voicemail_detected(self):
        """Called when voicemail is detected - hangup immediately"""
        self.logger.warning(
            f"🤖 VOICEMAIL DETECTED (confidence: {self.vmd_confidence:.2f}) - "
            f"Ending call immediately"
        )

        # Set disposition to VM (Voicemail)
        self.call_data['disposition'] = 'VM'

        # Mark detection flag
        self.voicemail_detected = True

        # Stop detection
        self.detection_stop_event.set()

        # Signal main thread to stop
        self.is_active = False

        # Direct hangup from detection thread
        try:
            self.conn.execute("hangup", "NORMAL_CLEARING")
        except Exception as e:
            self.logger.error(f"Error hanging up: {e}")

    def _handle_ringback_detected(self):
        """Called when ringback is detected - hangup immediately"""
        self.logger.warning("🔔 RINGBACK DETECTED - Ending call immediately")

        # Set disposition to RI (Ring, no answer)
        self.call_data['disposition'] = 'RI'
        self.ringing_detected = True

        # Stop detection
        self.detection_stop_event.set()

        # Signal main thread to stop
        self.is_active = False

        # Direct hangup from detection thread
        try:
            self.conn.execute("hangup", "NORMAL_CLEARING")
        except Exception as e:
            self.logger.error(f"Error hanging up: {e}")

    def _extract_left_channel_from_stereo(self, stereo_bytes: bytes) -> bytes:
        """
        Extract left channel (caller audio) from stereo PCM bytes

        Stereo audio is interleaved: L R L R L R ... (left, right, left, right, ...)
        We extract only the left channel: L L L ... (caller's voice only)

        Args:
            stereo_bytes: Interleaved stereo PCM bytes (16-bit samples)

        Returns:
            Mono PCM bytes containing only left channel (caller audio)
        """
        try:
            import numpy as np

            # Convert to numpy array (16-bit signed integers)
            stereo_array = np.frombuffer(stereo_bytes, dtype=np.int16)

            # Check if we actually have stereo data (even number of samples)
            if len(stereo_array) % 2 != 0:
                self.logger.warning(f"Odd number of samples ({len(stereo_array)}), audio may not be stereo")
                # Return as-is, might be mono already
                return stereo_bytes

            # Extract left channel (every other sample starting at index 0)
            # Stereo: [L0, R0, L1, R1, L2, R2, ...]
            # Left:   [L0, L1, L2, ...]
            left_channel = stereo_array[0::2]

            return left_channel.tobytes()

        except Exception as e:
            self.logger.error(f"Error extracting left channel: {e}")
            # Fallback: return original bytes (better than crashing)
            return stereo_bytes

    def _stop_audio_detection(self):
        """Stop unified audio detection (ringing + voicemail) and cleanup"""
        try:
            # Signal stop
            self.detection_stop_event.set()

            # Wait for thread
            if self.detection_thread and self.detection_thread.is_alive():
                self.detection_thread.join(timeout=1.0)

            # Stop recording
            if self.recording_file:
                cmd = f"uuid_record {self.uuid} stop {self.recording_file}"
                try:
                    self.conn.api(cmd)
                except:
                    pass

                # Keep recording file for debug analysis (don't delete)
                if os.path.exists(self.recording_file):
                    file_size = os.path.getsize(self.recording_file)
                    self.logger.info(f"Recording saved: {self.recording_file} ({file_size} bytes)")

            # Cleanup VMD classifier (release model reference)
            if self.vmd_classifier:
                try:
                    self.vmd_classifier.cleanup()
                    self.vmd_classifier = None
                except Exception as e:
                    self.logger.error(f"Error cleaning up VMD classifier: {e}")

            detection_summary = []
            if self.ringing_detected:
                detection_summary.append("ringing detected")
            if self.voicemail_detected:
                detection_summary.append(f"voicemail detected (conf={self.vmd_confidence:.2f})")
            if not detection_summary:
                detection_summary.append("no detections")

            self.logger.info(f"Audio detection stopped: {', '.join(detection_summary)}")

        except Exception as e:
            self.logger.error(f"Error stopping audio detection: {e}")

    def run(self):
        """
        Main call lifecycle
        This is the entry point - blocks until call completes
        """
        try:
            self.logger.info("="*60)
            self.logger.info(f"📞 Call Started: {self.phone_number}")
            self.logger.info("="*60)

            # Subscribe to events for this call
            self.conn.sendRecv("myevents")
            self.conn.sendRecv("linger")  # Keep connection alive after hangup

            # Initialize components
            self._initialize_components()

            # Answer the call
            self._answer_call()

            # Start unified audio detection (ringing + voicemail, uuid_record-based)
            self._start_audio_detection()

            # Load call flow
            self.call_flow = parse_call_flow_from_string(
                self.agent_config.script_content
            )

            if not self.call_flow:
                raise ValueError("Failed to load call flow")

            self.logger.info(f"Call flow loaded: {self.call_flow.get('name', 'Unknown')}")

            # Execute conversation
            self._execute_call_flow()

            # Finalize
            self._finalize_call()

        except Exception as e:
            self.logger.error(f"Call error: {e}", exc_info=True)
            self.call_data['error'] = str(e)
            self.call_data['disposition'] = 'NP'

        finally:
            self._cleanup()

        self.logger.info(f"✅ Call completed: {self.uuid[:8]}")

    def _initialize_components(self):
        """Initialize business logic components"""
        try:
            # Intent detector
            self.intent_detector = IntentDetector(
                hp_phrases=self.agent_config.honey_pot_sentences
            )

            # Parakeet model (singleton - accesses same instance preloaded by bot_server)
            parakeet = ParakeetModelSingleton()
            self.parakeet_model = parakeet.get_model(self.logger)
            if not self.parakeet_model:
                self.logger.warning("Parakeet model not available")

            # Silero VAD for regular response transcription filtering
            self.silero_vad = SileroVAD(
                threshold=0.5,
                sample_rate=8000,
                logger=self.logger
            )
            self.logger.info("Silero VAD initialized for response transcription")

            # Qwen intent detector (singleton - accesses same instance preloaded by bot_server)
            if QwenModelSingleton:
                qwen = QwenModelSingleton.get_instance()
                self.qwen_detector = qwen.get_detector(self.logger)
                self.logger.info("Qwen intent detector initialized")
            else:
                self.qwen_detector = None
                self.logger.warning("Qwen singleton not available")

            # CRM logger
            self.crm_logger = SuiteCRMLogger(
                self.agent_config,
                self.logger,
                self
            )

            # Outcome handler
            self.outcome_handler = CallOutcomeHandler(
                vicidial_integration=None,  # Simplified - no ViciDial integration
                logger=self.logger
            )

            # Continuous transcription handler
            if self.parakeet_model and self.intent_detector:
                self.continuous_transcription = ContinuousTranscriptionHandler(
                    parakeet_model=self.parakeet_model,
                    intent_detector=self.intent_detector,
                    logger=self.logger,
                    rnnt_confidence_threshold=self.agent_config.rnnt_confidence_threshold,
                    energy_threshold=self.agent_config.energy_threshold  # From database
                )
                self.logger.info(f"Continuous transcription handler initialized (energy_threshold={self.agent_config.energy_threshold})")
            else:
                self.logger.warning("Continuous transcription disabled - missing parakeet_model or intent_detector")

            # Log call start to CRM
            call_id = self.crm_logger.log_call_start(self.call_data)
            if call_id:
                self.call_data['id'] = call_id
                self.logger.info(f"Call logged to CRM: ID={call_id}")

            self.logger.info("Components initialized")

        except Exception as e:
            self.logger.error(f"Component initialization failed: {e}")
            raise

    def _answer_call(self):
        """Answer the call"""
        self.logger.info("Answering call...")
        self.conn.execute("answer", "")
        time.sleep(0.2)  # Brief pause after answer
        self.logger.info("Call answered")

    def _play_audio(self, audio_file: str, step: Optional[Dict] = None) -> bool:
        """
        Play audio file using simple blocking playback

        Args:
            audio_file: Audio file name
            step: Optional step dictionary with greetings/us_states flags

        Returns:
            True if playback succeeded, False if interrupted or failed
        """
        try:
            # Get full path to audio file
            audio_path = get_audio_path_for_agent(
                audio_file,
                self.agent_config.voice_location,
                greetings=False,  # Simplified - no time-based variants
                us_states=False   # Simplified - no state-based variants
            )

            if not audio_path or not os.path.exists(audio_path):
                self.logger.error(f"Audio file not found: {audio_path}")
                return False

            # Check if call is still active before playing
            if not self.is_active:
                self.logger.warning("Call not active - skipping playback")
                return False

            self.logger.info(f"Playing: {os.path.basename(audio_path)}")
            self.conversation_log.append(f"Bot: {os.path.basename(audio_path)}")

            # Mark playback start for continuous transcription
            self.is_playing_audio = True
            self.playback_start_time = time.time()
            if self.continuous_transcription:
                self.continuous_transcription.mark_playback_start()

            # Simple blocking playback
            self.conn.execute("playback", audio_path)

            # Mark playback end
            self.is_playing_audio = False
            if self.continuous_transcription:
                self.continuous_transcription.mark_playback_end()

            self.logger.debug(f"Playback completed")

            # Check for DNC/NI/HP AFTER playback completes
            if self.continuous_transcription:
                detected, intent_type = self.continuous_transcription.has_dnc_ni_detection()
                if detected:
                    self.logger.warning(f"🚫 {intent_type} detected after playback - ending call")

                    # Map intent type to disposition
                    intent_map = {
                        "DNC": ("DNC", "do_not_call", "dnc_during_playback"),
                        "NI": ("NI", "not_interested", "ni_during_playback"),
                        "HP": ("HP", "hold_press", "hp_during_playback")
                    }

                    if intent_type in intent_map:
                        disposition, intent_detected, call_result = intent_map[intent_type]
                        self.call_data['disposition'] = disposition
                        self.call_data['intent_detected'] = intent_detected
                        self.call_data['call_result'] = call_result
                        self.is_active = False
                        return False

            return True

        except Exception as e:
            self.logger.error(f"Playback error: {e}", exc_info=True)
            # Make sure to clean up playback state
            self.is_playing_audio = False
            if self.continuous_transcription:
                self.continuous_transcription.mark_playback_end()
            return False

    def _listen_for_response(self, timeout: int = 5) -> Optional[str]:
        """
        Record and transcribe user response using simple blocking recording

        Args:
            timeout: Maximum recording time in seconds

        Returns:
            Transcribed text or None
        """
        temp_path = None

        try:
            # Set main response listening flag (pauses continuous transcription)
            self.main_response_listening = True
            self.logger.debug("[MAIN RESPONSE] Pausing continuous transcription during response recording")

            # Generate unique filename
            temp_path = f"/tmp/fs_recording_{uuid.uuid4().hex[:10]}.wav"

            self.logger.info(f"Recording to: {temp_path}")

            # Check if call is still active
            if not self.is_active:
                self.logger.warning("Call not active - skipping recording")
                return None

            # Simple blocking recording
            # Format: record <file> <time_limit_secs> <silence_threshold_ms>
            record_args = f"{temp_path} {timeout} 2000"
            self.conn.execute("record", record_args)

            # Check if file has audio content
            if not os.path.exists(temp_path):
                self.logger.warning("Recording file not created")
                return None

            file_size = os.path.getsize(temp_path)
            if file_size < 1000:  # Less than 1KB
                self.logger.info("Silence detected (file too small)")
                return None

            self.logger.info(f"Recording completed: {file_size} bytes")

            # Transcribe using Parakeet
            text, confidence = self._transcribe_audio(temp_path)

            if text:
                self.logger.info(f"Transcribed: '{text}' (confidence: {confidence:.2f})")
                self.conversation_log.append(f"User: {text}")
                return text
            else:
                self.logger.info("No speech detected or low confidence")
                return None

        except Exception as e:
            self.logger.error(f"Recording error: {e}", exc_info=True)
            return None

        finally:
            # Clear main response listening flag (resumes continuous transcription)
            self.main_response_listening = False
            self.logger.debug("[MAIN RESPONSE] Resuming continuous transcription after response recording")

            # Cleanup temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

    def _transcribe_audio(self, audio_path: str) -> tuple:
        """
        Transcribe audio using Parakeet with Silero VAD pre-filtering

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (text, confidence)
        """
        try:
            if not self.parakeet_model:
                self.logger.error("Parakeet model not available")
                return None, 0.0

            # Pre-filter with Silero VAD to avoid transcribing silence/noise
            if self.silero_vad:
                try:
                    # Read audio file and check for speech
                    import soundfile as sf
                    audio_data, sample_rate = sf.read(audio_path)

                    # Convert to bytes for VAD (8kHz, 16-bit)
                    import numpy as np
                    if sample_rate != 8000:
                        # Resample if needed
                        import scipy.signal as scipy_signal
                        audio_data = scipy_signal.resample(
                            audio_data,
                            int(len(audio_data) * 8000 / sample_rate)
                        )

                    # Convert to 16-bit PCM bytes
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    audio_bytes = audio_int16.tobytes()

                    # Check if audio contains speech
                    if not self.silero_vad.is_speech(audio_bytes):
                        self.logger.info("Silero VAD: No speech detected in response audio")
                        return None, 0.0

                    self.logger.debug("Silero VAD: Speech detected, proceeding with transcription")

                except Exception as vad_error:
                    self.logger.warning(f"Silero VAD check failed: {vad_error}, proceeding with transcription anyway")

            # Use Parakeet to transcribe
            text, confidence = self.parakeet_model.transcribe_with_confidence(
                audio_path
            )

            # Check confidence threshold
            if confidence < self.agent_config.rnnt_confidence_threshold:
                self.logger.warning(
                    f"Low confidence: {confidence:.2f} < {self.agent_config.rnnt_confidence_threshold}"
                )
                return None, confidence

            return text, confidence

        except Exception as e:
            self.logger.error(f"Transcription error: {e}")
            return None, 0.0

    def _transfer_call(self, did: str) -> bool:
        """
        Transfer call to ViciDial using FreeSWITCH bridge

        Args:
            did: DID number to transfer to

        Returns:
            True if transfer succeeded
        """
        try:
            # Build SIP URI for transfer
            # Note: This depends on your FreeSWITCH gateway configuration
            sip_uri = f"sofia/gateway/vicidial_trunk/{did}"

            self.logger.info(f"Transferring to: {sip_uri}")

            # Update call data
            self.call_data['transfer_target'] = did
            self.call_data['transfer_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Execute bridge (blocks until transfer completes or fails)
            result = self.conn.execute("bridge", sip_uri)

            # Check result
            # Note: You may need to parse the result to determine success/failure
            self.logger.info(f"Transfer result: {result}")

            self.call_data['transfer_status'] = 'SUCCESS'
            self.call_data['disposition'] = 'QUALIFIED'

            return True

        except Exception as e:
            self.logger.error(f"Transfer error: {e}")
            self.call_data['transfer_status'] = 'FAILED'
            self.call_data['transfer_reason'] = str(e)
            return False

    def _hangup_call(self):
        """Hangup the call"""
        try:
            self.logger.info("Hanging up call")
            self.conn.execute("hangup", "")
            self.is_active = False
        except Exception as e:
            self.logger.error(f"Hangup error: {e}")

    def _perform_hangup_immediate(self, disposition: str, intent_type: str, call_result: str):
        """
        Flag for immediate hangup with disposition update
        Main thread handles actual hangup to avoid blocking

        Args:
            disposition: Call disposition (DNC, NI, HP)
            intent_type: Intent type for logging (do_not_call, not_interested, hold_press)
            call_result: Call result string for logging (e.g., 'dnc_during_playback')
        """
        self.logger.warning(f"🚫 Flagging immediate hangup: {disposition} detected")

        # Signal main thread to stop (no blocking execute() call!)
        self.is_active = False

        # Update call data
        self.call_data['disposition'] = disposition
        self.call_data['intent_detected'] = intent_type
        self.call_data['call_result'] = call_result

        # NOTE: Main thread will handle actual hangup in cleanup
        # Transcriptions collected in _finalize_call() (finally block)

    def _execute_call_flow(self):
        """
        Execute conversation flow (SIMPLIFIED FROM OLD CODE)
        Main conversation loop
        """
        self.logger.info(f"Executing flow: {self.call_flow.get('name', 'Unknown')}")

        # Get initial step
        self.current_step = self._get_initial_step()

        # Main conversation loop
        while self.current_step != 'exit' and self.is_active:
            # Get current step
            step = self.call_flow['steps'].get(self.current_step)
            if not step:
                self.logger.error(f"Step '{self.current_step}' not found")
                break

            self.logger.info(f"Step: {self.current_step}")

            # Process step
            try:
                if not self._process_step(step):
                    break
            except Exception as e:
                self.logger.error(f"Step processing error: {e}")
                self.call_data['disposition'] = 'NP'
                break

            # Small delay between steps
            time.sleep(0.2)

        # Handle flow completion
        if self.is_active:
            if self.call_data['disposition'] == 'INITIATED':
                self.call_data['disposition'] = 'NP'

    def _get_initial_step(self) -> str:
        """Determine the initial step"""
        if 'steps' not in self.call_flow:
            return 'start'

        # Look for common entry points
        for potential_start in ['hello', 'introduction', 'start']:
            if potential_start in self.call_flow['steps']:
                return potential_start

        # Use first available step
        if self.call_flow['steps']:
            return list(self.call_flow['steps'].keys())[0]

        return 'start'

    def _process_step(self, step: Dict[str, Any]) -> bool:
        """
        Process a single call flow step (SIMPLIFIED FROM OLD CODE)

        Args:
            step: Step dictionary from call flow

        Returns:
            True to continue, False to end flow
        """
        # Set disposition if specified
        if 'disposition' in step:
            self.call_data['disposition'] = step['disposition']
            self.logger.info(f"Disposition: {step['disposition']}")

        # Play audio if specified
        if 'audio_file' in step:
            if not self._play_audio(step['audio_file'], step):
                if self.call_data['disposition'] == 'INITIATED':
                    self.call_data['disposition'] = 'NP'
                return False

        # Handle action
        if 'action' in step:
            action = step['action']
            self.logger.info(f"Action: {action}")

            if action == 'transfer':
                self._handle_transfer()
                return False
            elif action == 'hangup':
                self._hangup_call()
                return False

        # Handle pause
        if 'pause_duration' in step:
            pause_seconds = step['pause_duration']
            if pause_seconds > 0:
                time.sleep(pause_seconds)

        # Handle response waiting
        if step.get('wait_for_response', False):
            return self._handle_user_response(step)
        else:
            self.current_step = step.get('next', 'exit')
            return True

    def _extract_question_from_step(self, step: Dict[str, Any]) -> str:
        """
        Extract a meaningful question from the step for Qwen context.

        Args:
            step: Current step dictionary

        Returns:
            A question string to provide context for Qwen
        """
        # Try to get question from step's audio_file name
        audio_file = step.get('audio_file', '')

        if audio_file:
            # Convert audio file name to a readable question
            # Example: "medicare_main_qualification.wav" → "Do you have Medicare?"
            # Example: "medicare_age_check.wav" → "Are you over 65?"

            question_base = audio_file.replace('.wav', '').replace('_', ' ')

            # Common question patterns based on Medicare flow
            if 'qualification' in audio_file.lower() or 'medicare' in audio_file.lower():
                return "Do you have Medicare Part A and Part B?"
            elif 'age' in audio_file.lower():
                return "Are you over the age of 65?"
            elif 'interest' in audio_file.lower():
                return "Are you interested in learning more about this?"
            else:
                # Generic question based on file name
                return f"Regarding {question_base}, how do you respond?"

        # Fallback: use step name or generic question
        step_name = self.current_step if self.current_step else 'this question'
        return f"Are you interested in continuing with {step_name}?"

    def _handle_user_response(self, step: Dict[str, Any]) -> bool:
        """
        Handle user response (SIMPLIFIED FROM OLD CODE)

        Args:
            step: Current step dictionary

        Returns:
            True to continue, False to end flow
        """
        # Stop continuous transcription before response recording to prevent bleed
        # The grace period would otherwise process user response audio as delayed playback chunks
        self.playback_periods.clear()  # Stop grace period from matching
        if self.continuous_transcription:
            self.continuous_transcription.clear_buffer()  # Clear stale audio

        timeout = step.get('timeout', 10)
        user_response = self._listen_for_response(timeout)

        if not user_response:
            # Silence
            self.conversation_log.append("User: <silence>")
            self.consecutive_silence_count += 1

            if self.consecutive_silence_count >= self.max_consecutive_silences:
                self.logger.warning("Max silences exceeded")
                self.call_data['disposition'] = 'NP'
                return False

            # Go to silence step
            self.current_step = step.get('silence_step', step.get('next', 'exit'))
            return True

        # Reset silence counter
        self.consecutive_silence_count = 0

        # Log current call state for debugging
        self.logger.info(f"🔍 Starting intent detection for: '{user_response}'")
        self.logger.info(f"📊 Current call state: step={self.current_step}, silence_count={self.consecutive_silence_count}")

        # Detect intent using Qwen (with fallback to keyword detector)
        intent_result = None

        # Try Qwen first if available
        if hasattr(self, 'qwen_detector') and self.qwen_detector:
            try:
                # Construct question context from step
                # Try to get a meaningful question from audio_file name or step name
                question = self._extract_question_from_step(step)

                self.logger.info(f"🤖 Attempting Qwen classification - Q: '{question}' A: '{user_response}'")

                # Get Qwen classification (returns: positive, negative, clarifying, neutral, or None)
                qwen_classification = self.qwen_detector.detect_intent(question, user_response, timeout=1.5)

                if qwen_classification:
                    self.logger.info(f"✅ Qwen result: {qwen_classification}")

                    # Map Qwen classifications to next step using conditions array
                    if qwen_classification == "negative":
                        # User is declining - evaluate conditions to find next step
                        self.logger.info("Qwen: Negative response detected → Evaluating conditions")
                        next_step = self._map_qwen_intent_to_step(step, "negative")
                        if next_step:
                            self.logger.info(f"Qwen: Routing negative response to step: {next_step}")
                            self.current_step = next_step
                            # Check if the next step is a terminal "not_interested" step
                            if next_step == "not_interested":
                                self._handle_not_interested()
                                return False
                            return True
                        else:
                            # No condition matched - treat as not interested
                            self._handle_not_interested()
                            return False
                    elif qwen_classification == "positive":
                        # User is agreeing/interested - evaluate conditions to find next step
                        self.logger.info("Qwen: Positive response detected → Evaluating conditions")
                        next_step = self._map_qwen_intent_to_step(step, "positive")
                        if next_step:
                            self.logger.info(f"Qwen: Routing positive response to step: {next_step}")
                            self.current_step = next_step
                            return True
                        else:
                            # No condition matched - default to exit
                            self.logger.warning("Qwen: No condition matched for positive response - exiting")
                            self.current_step = 'exit'
                            return False
                    elif qwen_classification == "clarifying":
                        # User needs clarification - evaluate conditions to find next step
                        self.logger.info("Qwen: Clarifying response detected → Evaluating conditions")
                        next_step = self._map_qwen_intent_to_step(step, "clarifying")
                        if next_step:
                            self.logger.info(f"Qwen: Routing clarifying response to step: {next_step}")
                            self.current_step = next_step
                            return True
                        else:
                            # No explicit clarifying route - use no_match_next for clarification
                            self.logger.info("Qwen: No clarifying route - using no_match_next")
                            self.current_step = step.get('no_match_next', self.current_step)
                            return True
                    elif qwen_classification == "neutral":
                        # Random/nonsensical response - evaluate conditions
                        self.logger.info("Qwen: Neutral response detected → Evaluating conditions")
                        next_step = self._map_qwen_intent_to_step(step, "neutral")
                        if next_step:
                            self.logger.info(f"Qwen: Routing neutral response to step: {next_step}")
                            self.current_step = next_step
                            return True
                        else:
                            # Treat as silence/unclear - increment silence counter
                            self.logger.info("Qwen: Neutral response - treating as silence")
                            self.consecutive_silence_count += 1
                            if self.consecutive_silence_count >= self.max_consecutive_silences:
                                self.call_data['disposition'] = 'NP'
                                return False
                            self.current_step = step.get('silence_step', step.get('next', 'exit'))
                            return True
                else:
                    # Qwen returned None (timeout or error) - fall back to keyword detector
                    self.logger.warning(f"⚠️ Qwen returned None (likely timeout) - falling back to keyword detector")
                    # Log Qwen metrics for debugging
                    try:
                        metrics = self.qwen_detector.get_performance_metrics()
                        self.logger.info(f"📈 Qwen metrics: {metrics}")
                    except:
                        pass

            except Exception as e:
                self.logger.warning(f"⚠️ Qwen detection exception: {e} - falling back to keyword detector")

        # Fallback to keyword-based intent detector
        self.logger.info("📝 Using keyword-based intent detector as fallback")
        result = self.intent_detector.detect_intent(user_response)
        if result:
            intent, confidence = result
            self.logger.info(f"✅ Keyword detector SUCCEEDED: {intent} (confidence: {confidence:.2f})")

            # Handle intents
            if intent == "do_not_call":
                self._handle_dnc()
                return False
            elif intent == "not_interested":
                self._handle_not_interested()
                return False
            elif intent == "qualified":
                self._handle_qualified()
                return False
        else:
            self.logger.info(f"❌ Keyword detector found NO MATCH for: '{user_response}'")

        # No specific intent detected - move to next step
        self.logger.info("No specific intent detected, continuing to next step")
        self.current_step = step.get('next', 'exit')
        return True

    def _handle_dnc(self):
        """Handle Do Not Call request"""
        self.logger.warning("🚫 DNC request detected - ending call immediately")
        self._perform_hangup_immediate('DNC', 'do_not_call', 'dnc_from_response')

    # OBSOLETE METHODS REMOVED (replaced by _check_all_detections() unified approach):
    # - _handle_dnc_detected_during_playback()
    # - _handle_ni_detected_during_playback()
    # - _handle_hp_detected_during_playback()
    # All playback interruptions now handled by unified detection checker in _play_audio() loop

    def _map_qwen_intent_to_step(self, step, qwen_intent):
        """Map Qwen intent (positive/negative/clarifying) to next step based on conditions array"""
        conditions = step.get('conditions', [])

        if not conditions:
            return step.get('next', 'exit')

        # First priority: Check for intent-type conditions (new format)
        for condition in conditions:
            if condition.get('type') == 'intent' and condition.get('intent') == qwen_intent:
                next_step = condition.get('next')
                self.logger.debug(f"Matched intent condition: {qwen_intent} -> {next_step}")
                return next_step

        # Handle clarifying intent specifically
        if qwen_intent == "clarifying":
            # User is asking for more information
            # Look for a condition with intent='clarifying' in the step
            for condition in conditions:
                if condition.get('type') == 'intent' and condition.get('intent') == 'clarifying':
                    next_step = condition.get('next')
                    self.logger.debug(f"Qwen classified as clarifying - routing to: {next_step}")
                    return next_step

            # If no explicit clarifying condition, treat as neutral (trigger clarification)
            self.logger.debug(f"Qwen classified as clarifying but no route - triggering clarification")
            return None

        # Fallback: Handle legacy keyword-based conditions
        positive_keywords = ['yes', 'yeah', 'i do', 'i have', 'sure', 'okay', 'correct']
        negative_keywords = ['no', 'nope', "don't", "not", "none", "neither"]

        if qwen_intent == "positive":
            # Find condition with positive keywords
            for condition in conditions:
                if condition.get('type') == 'contains':
                    keywords = [kw.lower() for kw in condition.get('keywords', [])]
                    if any(kw in positive_keywords for kw in keywords):
                        return condition.get('next')

            # If no explicit positive condition, take first condition
            if conditions:
                return conditions[0].get('next')

        elif qwen_intent == "negative":
            # Find condition with negative keywords
            for condition in conditions:
                if condition.get('type') == 'contains':
                    keywords = [kw.lower() for kw in condition.get('keywords', [])]
                    if any(kw in negative_keywords for kw in keywords):
                        return condition.get('next')

            # If no explicit negative condition, take second condition if exists
            if len(conditions) > 1:
                return conditions[1].get('next')

        elif qwen_intent == "neutral":
            # Neutral responses should trigger clarification
            # Return None to use no_match_next path (clarification)
            self.logger.debug(f"Qwen classified response as neutral - triggering clarification")
            return None

        # Default fallback
        return step.get('next', 'exit')

    def _handle_not_interested(self):
        """Handle Not Interested response"""
        self.logger.warning("⚠️ Not interested detected - ending call immediately")
        self._perform_hangup_immediate('NI', 'not_interested', 'ni_from_response')

    def _handle_qualified(self):
        """Handle qualified lead"""
        self.logger.info("✅ Qualified lead detected")
        self.call_data['disposition'] = 'SALE'
        self.call_data['intent_detected'] = 'qualified'
        # Transfer will be handled by step action

    def _handle_transfer(self):
        """Handle transfer action"""
        did = self.agent_config.did_transfer_qualified
        if not did:
            self.logger.error("No transfer DID configured")
            return

        self.logger.info(f"Transferring qualified call to {did}")

        if self._transfer_call(did):
            self.logger.info("✅ Transfer successful")
        else:
            self.logger.error("❌ Transfer failed")
            self.call_data['disposition'] = 'NP'

    def _collect_continuous_transcriptions(self):
        """
        Collect transcriptions from continuous listener and add to conversation log
        Similar to old pjsua2 implementation
        """
        if not self.continuous_transcription:
            return

        try:
            # Get all transcriptions since call start
            since_time = self.call_data['start_time']
            transcriptions = self.continuous_transcription.get_transcriptions_since(
                since_time,
                min_confidence=0.3
            )

            if transcriptions:
                self.logger.info(f"Collected {len(transcriptions)} continuous transcriptions for logging")

                # Add to conversation log with timestamps
                for timestamp, text, confidence in transcriptions:
                    # Calculate time offset from call start
                    offset = timestamp - self.call_data['start_time']
                    time_str = time.strftime('%H:%M:%S', time.localtime(timestamp))

                    # Format similar to old implementation
                    formatted_entry = f"User (continuous @ {time_str}): {text}"
                    self.conversation_log.append(formatted_entry)

                    self.logger.debug(f"Added continuous transcription: {formatted_entry}")

        except Exception as e:
            self.logger.error(f"Error collecting continuous transcriptions: {e}", exc_info=True)

    def _finalize_call(self):
        """Finalize call and save to CRM"""
        try:
            # Collect any remaining continuous transcriptions
            self._collect_continuous_transcriptions()

            # Calculate duration
            self.call_data['end_time'] = time.time()
            self.call_data['duration'] = int(
                self.call_data['end_time'] - self.call_data['start_time']
            )

            # Build transcript
            self.call_data['transcript'] = "\n".join(self.conversation_log)

            # Log continuous transcription stats
            if self.continuous_transcription:
                stats = self.continuous_transcription.get_stats()
                self.logger.info(
                    f"Continuous transcription stats: "
                    f"chunks={stats['chunks_processed']}, "
                    f"transcriptions={stats['transcriptions_successful']}/{stats['transcriptions_attempted']}, "
                    f"dnc={stats['dnc_detections']}, ni={stats['ni_detections']}, hp={stats['hp_detections']}"
                )

            # Save to CRM
            if self.crm_logger:
                success = self.crm_logger.log_call_end(self.call_data)
                if success:
                    self.logger.info("✅ Call saved to CRM")
                else:
                    self.logger.error("❌ Failed to save call to CRM")

            self.logger.info(f"Call finalized: Disposition={self.call_data['disposition']}, "
                           f"Duration={self.call_data['duration']}s")

        except Exception as e:
            self.logger.error(f"Finalization error: {e}")

    def _cleanup(self):
        """Cleanup resources"""
        try:
            self.is_active = False

            # Stop unified audio detection (ringing + voicemail)
            self._stop_audio_detection()

            self.logger.info("Cleanup complete")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
