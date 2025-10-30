#!/usr/bin/env python3
"""
Production-Grade SuiteCRM Bot Instance
- Bulletproof resource management with automatic cleanup
- Thread-safe operations with timeout protection
- Crash-resistant with comprehensive error handling
- Zero resource leaks guaranteed
- Using NVIDIA Parakeet TDT for speech recognition
"""

import pjsua2 as pj
import time
import os
import threading
import json
import re
import logging
import copy
import gc
import weakref
import queue
import traceback
import numpy as np
import torch
import torchaudio
import soundfile as sf
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from scipy import signal as scipy_signal

# Project imports
from src.config import MIN_INTENT_CONFIDENCE, USE_GPU
from src.call_flow import parse_call_flow_from_string, get_audio_path_for_agent
from src.intent_detector import IntentDetector
from src.sip_audio_manager import SIPAudioManager, SIPAudioRecorder
from src.stereo_call_recorder import StereoCallRecorder
from src.call_outcome_handler import CallOutcomeHandler, CallOutcome
from src.suitecrm_integration import SuiteCRMAgentConfig, SuiteCRMLogger
from src.sip_transfer_manager import ViciDialApiTransfer
from src.sip_voicemail_detector import LocalVoicemailDetector
from src.sip_ringing_detector import LocalRingingDetector
from src.vicidial_integration import ViciDialAPI, VoicebotViciDialIntegration

# Configuration
LOG_DIR = "/var/log/sip-bot"
SUITECRM_UPLOAD_DIR = "/var/www/recordings"
OPERATION_TIMEOUT = 30  # Default timeout for operations
CLEANUP_TIMEOUT = 10  # Timeout for cleanup operations
MAX_RETRIES = 3  # Maximum retries for critical operations

# PJSIP Error codes
PJSIP_ERROR_NOT_FOUND = 70001  # PJ_ENOTFOUND
PJSIP_ERROR_INVALID = 70004     # PJ_EINVAL


class AudioBuffer:
    """Manages audio buffering and voice activity detection"""
    def __init__(self, sample_rate=8000, logger=None, energy_threshold=0.045):
        self.sample_rate = sample_rate
        self.logger = logger or logging.getLogger(__name__)
        self.energy_threshold = energy_threshold

        # Create Silero VAD instance using same threshold from database
        from src.silero_vad import SileroVAD
        self.silero_vad = SileroVAD(
            threshold=energy_threshold,  # Direct passthrough from database
            sample_rate=sample_rate,
            logger=logger
        )

        self.buffer = []
        self.silence_start = None
        self.voice_detected = False
        self.speech_start = None
        self.total_samples = 0
        
    def add_chunk(self, audio_chunk):
        """Add audio chunk and detect voice activity using Silero VAD only"""
        # Convert to numpy array
        audio_array = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32)
        audio_array = audio_array / 32768.0  # Normalize to [-1, 1]

        # Voice activity detection with Silero VAD (no energy threshold)
        if self.silero_vad.is_speech(audio_chunk):
            # Silero confirmed speech - buffer regardless of volume
            if not self.voice_detected:
                self.speech_start = time.time()
            self.voice_detected = True
            self.silence_start = None
            self.buffer.append(audio_chunk)
            self.total_samples += len(audio_array)
        else:
            # Not speech (silence or noise)
            # Buffer trailing non-speech after voice detected (for speech boundary detection)
            if self.voice_detected:
                if self.silence_start is None:
                    self.silence_start = time.time()
                self.buffer.append(audio_chunk)
                self.total_samples += len(audio_array)
            # else: non-speech before voice started - discard
        
        # Check if we should process
        should_process = False
        
        # Process if we have enough silence after voice AND minimum speech duration
        if self.voice_detected and self.silence_start and self.speech_start:
            silence_duration = time.time() - self.silence_start
            speech_duration = self.silence_start - self.speech_start
            if silence_duration >= SILENCE_DURATION_THRESHOLD and speech_duration >= MIN_SPEECH_DURATION:
                should_process = True
        
        # Process if buffer is getting too large
        buffer_duration = self.total_samples / self.sample_rate
        if buffer_duration >= MAX_AUDIO_BUFFER_SECONDS:
            should_process = True
        
        return should_process
    
    def get_audio(self):
        """Get buffered audio and reset"""
        if not self.buffer:
            return None
        
        # Check minimum length
        buffer_duration = self.total_samples / self.sample_rate
        if buffer_duration < MIN_AUDIO_LENGTH_SECONDS:
            return None
        
        combined = b''.join(self.buffer)
        self.reset()
        return combined
    
    def reset(self):
        """Reset buffer"""
        self.buffer = []
        self.silence_start = None
        self.speech_start = None
        self.voice_detected = False
        self.total_samples = 0

class CallLogCollector:
    """Thread-safe collector for detailed call logs to be stored in database"""
    def __init__(self, logger_adapter, call_id):
        self._lock = threading.RLock()
        self._logs = []
        self._db_metrics = []  # Track database performance metrics
        self._start_time = time.time()
        self._agent_id = getattr(logger_adapter, 'extra', {}).get('agent_id', 'unknown')
        self._call_id = call_id
        
    def _strip_emojis(self, text: str) -> str:
        """Remove emoji characters that cause MySQL encoding issues"""
        # Remove common emoji patterns and replace with simple text
        emoji_replacements = {
            '🎙️': '[MIC]',
            '🎧': '[HEADPHONE]',
            '✅': '[OK]',
            '❌': '[ERROR]',
            '⚠️': '[WARNING]',
            '🚀': '[ROCKET]',
            '🔌': '[PLUG]',
            '🛑': '[STOP]',
            '💿': '[DISC]',
            '📻': '[RADIO]',
            '🔔': '[BELL]',
            '🚫': '[BLOCKED]',
            '👤': '[PERSON]',
            '🎵': '[MUSIC]',
            '🔍': '[SEARCH]',
            '🌐': '[GLOBE]',
        }
        
        clean_text = text
        for emoji, replacement in emoji_replacements.items():
            clean_text = clean_text.replace(emoji, replacement)
        
        # Remove any remaining emoji characters using regex
        import re
        # Unicode ranges for emojis and symbols
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002500-\U00002BEF"  # chinese char
            u"\U00002702-\U000027B0"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U00010000-\U0010ffff"
            u"\u2640-\u2642" 
            u"\u2600-\u2B55"
            u"\u200d"
            u"\u23cf"
            u"\u23e9"
            u"\u231a"
            u"\ufe0f"  # dingbats
            u"\u3030"
            "]+", flags=re.UNICODE)
        
        return emoji_pattern.sub('', clean_text)
        
    def add_log(self, level: str, message: str):
        """Add a log entry with timestamp, thread-safe"""
        with self._lock:
            # Strip emojis from the message to prevent database encoding issues
            clean_message = self._strip_emojis(message)
            timestamp = time.time() - self._start_time
            formatted_time = time.strftime('%H:%M:%S', time.localtime(self._start_time + timestamp))
            log_entry = f"[{formatted_time}] [{level}] [{self._agent_id}] [Call-{self._call_id}] {clean_message}"
            self._logs.append(log_entry)
            
            # Track database operations for metrics
            if any(keyword in clean_message for keyword in ["Query:", "SLOW QUERY:", "Query error", "Query failed"]):
                self.add_db_metric(clean_message)
    
    def add_db_metric(self, metric_message: str):
        """Add database performance metric with thread safety"""
        with self._lock:
            timestamp = time.time() - self._start_time
            formatted_time = time.strftime('%H:%M:%S', time.localtime(self._start_time + timestamp))
            db_entry = f"[{formatted_time}] {metric_message}"
            self._db_metrics.append(db_entry)
    
    def get_logs(self) -> str:
        """Get all collected logs as formatted text with database metrics"""
        with self._lock:
            combined = '\n'.join(self._logs)
            
            # Add database metrics section if we have any
            if self._db_metrics:
                combined += "\n\n=== DATABASE PERFORMANCE METRICS ===\n"
                combined += "\n".join(self._db_metrics)
                
                # Add summary statistics
                query_count = len([m for m in self._db_metrics if "Query:" in m])
                slow_query_count = len([m for m in self._db_metrics if "SLOW QUERY:" in m])
                error_count = len([m for m in self._db_metrics if "Query error" in m or "Query failed" in m])
                
                combined += f"\n\n=== DATABASE METRICS SUMMARY ===\n"
                combined += f"Total Queries: {query_count}\n"
                combined += f"Slow Queries (>1s): {slow_query_count}\n"
                combined += f"Query Errors: {error_count}"
            
            return combined
    
    def clear(self):
        """Clear all collected logs and database metrics"""
        with self._lock:
            self._logs.clear()
            self._db_metrics.clear()
    
    def get_log_count(self) -> int:
        """Get the number of collected logs"""
        with self._lock:
            return len(self._logs)

class ThreadSafeState:
    """Thread-safe state management with atomic operations"""
    def __init__(self):
        self._lock = threading.RLock()
        self._state = {
            'is_active': True,
            'call_transferred': False,
            'cleanup_done': False,
            'media_active': False,
            'conversation_started': False,
            'vmd_triggered': False,
            'ringing_triggered': False,
            'emergency_cleanup': False,
            'disposition_set': False,  # Track if disposition was explicitly set
            'crm_logged': False,  # Track if call was logged to CRM
            'crm_updated': False  # Track if CRM was updated
        }
        self._timestamps = {
            'created': time.time(),
            'last_activity': time.time()
        }
    
    def get(self, key, default=None):
        with self._lock:
            return self._state.get(key, default)
    
    def set(self, key, value):
        with self._lock:
            self._state[key] = value
            self._timestamps['last_activity'] = time.time()
    
    def update_multiple(self, updates):
        with self._lock:
            self._state.update(updates)
            self._timestamps['last_activity'] = time.time()
    
    def get_age(self):
        with self._lock:
            return time.time() - self._timestamps['created']
    
    def get_idle_time(self):
        with self._lock:
            return time.time() - self._timestamps['last_activity']

class ResourceManager:
    """Manages all resources for a call with guaranteed cleanup"""
    def __init__(self, logger):
        self.logger = logger
        self.resources = {}
        self.cleanup_queue = queue.Queue()
        self.lock = threading.RLock()
        self._cleanup_done = False
    
    def register(self, name, resource, cleanup_func=None, is_pjsip=False):
        """Register a resource with optional cleanup function"""
        with self.lock:
            if self._cleanup_done:
                return False
            self.resources[name] = {
                'resource': resource,
                'cleanup': cleanup_func,
                'is_pjsip': is_pjsip,
                'registered': time.time()
            }
            return True
    
    def get(self, name):
        """Get a registered resource"""
        with self.lock:
            entry = self.resources.get(name)
            return entry['resource'] if entry else None
    
    def cleanup_resource(self, name):
        """Cleanup a specific resource"""
        with self.lock:
            entry = self.resources.pop(name, None)
            if not entry:
                return
                
        # Cleanup outside lock
        if entry.get('cleanup'):
            try:
                # Check if it's a PJSIP resource and we're in wrong thread
                if entry.get('is_pjsip') and not hasattr(threading.current_thread(), '_pj_registered'):
                    self.logger.warning(f"Skipping PJSIP cleanup for {name} - wrong thread")
                    return
                    
                entry['cleanup']()
                self.logger.info(f"Cleaned up resource: {name}")
            except Exception as e:
                self.logger.error(f"Failed to cleanup {name}: {e}")
    
    def cleanup_all(self):
        """Cleanup all resources in reverse order"""
        with self.lock:
            if self._cleanup_done:
                return
            self._cleanup_done = True
            resources_to_clean = list(self.resources.items())
        
        # Separate PJSIP and non-PJSIP resources
        pjsip_resources = []
        other_resources = []
        
        for name, entry in resources_to_clean:
            if entry.get('is_pjsip'):
                pjsip_resources.append((name, entry))
            else:
                other_resources.append((name, entry))
        
        # Clean non-PJSIP resources first
        for name, entry in reversed(other_resources):
            if entry.get('cleanup'):
                try:
                    entry['cleanup']()
                    self.logger.info(f"Cleaned up: {name}")
                except Exception as e:
                    self.logger.error(f"Cleanup failed for {name}: {e}")
        
        # Only clean PJSIP resources if we're in a registered thread
        if hasattr(threading.current_thread(), '_pj_registered'):
            for name, entry in reversed(pjsip_resources):
                if entry.get('cleanup'):
                    try:
                        entry['cleanup']()
                        self.logger.info(f"Cleaned up PJSIP resource: {name}")
                    except Exception as e:
                        self.logger.error(f"PJSIP cleanup failed for {name}: {e}")
        else:
            if pjsip_resources:
                self.logger.warning(f"Skipping {len(pjsip_resources)} PJSIP resources - not in PJSIP thread")
        
        with self.lock:
            self.resources.clear()

class SuiteCRMBotInstance(pj.Call):
    """Production-grade bot instance with bulletproof resource management"""
    
    def __init__(self, account, call_id, agent_config, source_ip=None, parakeet_singleton=None, qwen_singleton=None, resource_registry=None, 
                 global_energy_threshold=0.045, global_rnnt_confidence_threshold=0.5):
        super().__init__(account, call_id)
        
        # Essential attributes
        self.account = account
        self.agent_config = agent_config
        self.source_ip = source_ip  # ViciDial server IP that originated this call
        self.parakeet_singleton = parakeet_singleton
        self.qwen_singleton = qwen_singleton
        self.resource_registry = resource_registry
        
        # Global thresholds from database
        self.global_energy_threshold = global_energy_threshold
        self.global_rnnt_confidence_threshold = global_rnnt_confidence_threshold
        
        # Thread-safe state
        self.state = ThreadSafeState()
        self.resource_manager = ResourceManager(self._get_logger())
        
        # Initialize logger
        agent_id_short = self.agent_config.agent_id.split('-')[0]
        self.logger = logging.LoggerAdapter(
            logging.getLogger(f"Call-{self.getId()}"),
            {'agent_id': agent_id_short}
        )
        
        # Initialize call log collector
        self.call_log_collector = CallLogCollector(self.logger, self.getId())
        
        # Core attributes
        self.call_start_time = time.time()
        self.phone_number = "UNKNOWN"
        self.caller_state = None  # US state detected from area code
        self.current_step = None
        self.call_flow = None

        # Transfer tracking (most reliable indicator)
        self.sip_refer_sent = False  # Set to True when onCallTransferStatus fires (confirms SIP REFER was sent)
        
        # Audio race condition prevention
        self.audio_transition_lock = threading.Lock()
        self.last_playback_end = 0
        
        # Thread management
        self.conversation_thread = None
        self.operation_executor = ThreadPoolExecutor(max_workers=5)
        # Mark executor as non-PJSIP resource
        self.resource_manager.register('executor', self.operation_executor, 
                                      lambda: self.operation_executor.shutdown(wait=False),
                                      is_pjsip=False)
        
        # Media resources
        self.audio_media = None
        self.audio_manager = None
        self.audio_recorder = None
        
        # Initialize stereo call recorder early (now safe - no PJSIP resources)
        try:
            from src.stereo_call_recorder import StereoCallRecorder
            self.stereo_call_recorder = StereoCallRecorder(self.logger)
            # Register for cleanup
            self.resource_manager.register('stereo_call_recorder', self.stereo_call_recorder,
                                          lambda: self._safe_cleanup(self.stereo_call_recorder, 'cleanup'),
                                          is_pjsip=False)  # Not a PJSIP resource
            self._log_and_collect('info', "Stereo call recorder initialized early (safe - no PJSIP)")
        except Exception as e:
            self._log_and_collect('error', f"Failed to initialize stereo recorder in constructor: {e}")
            self.stereo_call_recorder = None
        
        self.vmd_audio_recorder = None
        self.voicemail_detector = None
        self.ringing_audio_recorder = None
        self.ringing_detector = None
        
        # Speech recognition
        self.parakeet_model = None
        self.device = None
        self.audio_buffer = None
        
        # Integration components
        self.vicidial_integration = None
        self.intent_detector = None
        self.qwen_detector = None  # Qwen intent detector
        self.crm_logger = None  # Initialize early
        self.transfer_manager = None
        self.outcome_handler = None
        
        # Question tracking for Qwen
        self.current_question_text = None
        self.current_step_id = None
        
        # Qwen metrics tracking
        self.qwen_metrics = {
            'total_calls': 0,
            'avg_latency': 0,
            'fallback_count': 0,
            'successful_decisions': 0,
            'failed_decisions': 0
        }
        
        # Call data
        self.call_data = self._initialize_call_data()
        
        # Conversation tracking
        self.consecutive_silence_count = 0
        self.clarification_count = 0
        self.all_audio_chunks = []
        self.conversation_log = []
        
        # Configuration
        self.max_consecutive_silences = self.agent_config.max_silence_retries
        self.max_clarifications = self.agent_config.max_clarification_retries
        
        # File handler for logging
        self.file_handler = None
        
        # CRITICAL: Initialize CRM logger immediately
        try:
            self.crm_logger = SuiteCRMLogger(self.agent_config, self.logger, self)
            self._log_and_collect('info', "CRM logger initialized early")
        except Exception as e:
            self._log_and_collect('error', f"Failed to initialize CRM logger: {e}")
        
        # Initialize VMD if enabled
        if self.agent_config.enable_voicemail_detection == 'Yes':
            self._initialize_vmd()
        
        # Initialize ringing detection if enabled
        from src.config import RINGING_DETECTION_ENABLED
        if RINGING_DETECTION_ENABLED:
            self._initialize_ringing_detection()
        
        # Initialize continuous listening
        self.continuous_listener = None
        from src.config import CONTINUOUS_LISTENING_ENABLED
        if CONTINUOUS_LISTENING_ENABLED:
            self._initialize_continuous_listening()
        
        self._log_and_collect('info', f"Bot instance created (Silences: {self.max_consecutive_silences}, "
                        f"Clarifications: {self.max_clarifications})")
    
    def _get_logger(self):
        """Get logger instance"""
        return getattr(self, 'logger', logging.getLogger(__name__))
    
    def _log_and_collect(self, level: str, message: str):
        """Log message to both standard logger and call log collector"""
        # Log to standard logger
        if level.lower() == 'info':
            self.logger.info(message)
        elif level.lower() == 'warning':
            self.logger.warning(message)
        elif level.lower() == 'error':
            self.logger.error(message)
        elif level.lower() == 'debug':
            self.logger.debug(message)
        
        # Also collect in call log collector for database storage
        if hasattr(self, 'call_log_collector'):
            self.call_log_collector.add_log(level.upper(), message)

    
    def _collect_continuous_transcriptions(self, since_time=None, final_collection=False):
        """
        Collect transcriptions from continuous listener and add to conversation log
        
        Args:
            since_time: Only collect transcriptions after this timestamp. If None, collect all recent.
            final_collection: If True, this is the final collection at call end
        
        Returns:
            List of collected transcriptions
        """
        if not hasattr(self, 'continuous_listener') or not self.continuous_listener:
            return []
        
        try:
            # Default to collecting from last 120 seconds if no since_time specified
            # For final collection, collect from entire call duration
            if since_time is None:
                if final_collection:
                    # Collect from entire call - use call start time if available
                    since_time = getattr(self, 'call_start_time', time.time() - 300.0)  # Fallback to 5 minutes
                else:
                    since_time = time.time() - 120.0
            
            # Get transcriptions from continuous listener
            transcriptions = self.continuous_listener.get_transcriptions_since(since_time, min_confidence=0.3)
            
            if not transcriptions:
                if final_collection:
                    # Enhanced debug logging to understand why no transcriptions
                    total_transcriptions = len(self.continuous_listener.transcriptions) if hasattr(self.continuous_listener, 'transcriptions') else 0
                    self._log_and_collect('debug', f"No continuous transcriptions to collect at call end. Total stored: {total_transcriptions}, since_time: {since_time}")
                    
                    # Log a few recent transcriptions for debugging
                    if total_transcriptions > 0:
                        recent_transcriptions = self.continuous_listener.transcriptions[-3:] if hasattr(self.continuous_listener, 'transcriptions') else []
                        self._log_and_collect('debug', f"Recent transcriptions in buffer: {[(t, txt[:50], conf) for t, txt, conf in recent_transcriptions]}")
                return []
            
            collected_count = 0
            new_transcriptions = []
            
            for timestamp, text, confidence in transcriptions:
                # Format timestamp for logging
                time_str = datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')
                
                # Create formatted entry for conversation log
                formatted_entry = f"User (continuous @ {time_str}): {text}"
                
                # Only add if not already in conversation log (avoid duplicates)
                if formatted_entry not in self.conversation_log:
                    self.conversation_log.append(formatted_entry)
                    new_transcriptions.append((timestamp, text, confidence))
                    collected_count += 1
                    
                    # Log each continuous transcription with context
                    self._log_and_collect('info', f"📝 Continuous transcription added: '{text}' (conf: {confidence:.2f})")
            
            if collected_count > 0:
                context = "final collection" if final_collection else "periodic collection"
                self._log_and_collect('info', f"Collected {collected_count} continuous transcriptions ({context})")
            elif final_collection:
                self._log_and_collect('debug', f"All {len(transcriptions)} continuous transcriptions were already in conversation log")
            
            return new_transcriptions
            
        except Exception as e:
            self._log_and_collect('error', f"Error collecting continuous transcriptions: {e}")
            return []
    
    def _initialize_call_data(self):
        """Initialize call data structure"""
        return {
            'id': None,
            'phone_number': self.phone_number,
            'caller_state': self.caller_state,
            'start_time': self.call_start_time,
            'end_time': None,
            'disposition': 'INITIATED',  # Start with INITIATED
            'transcript': '',
            'is_voicemail': False,
            'intent_detected': None,
            'uniqueid': f"voicebot_{self.getId()}_{int(self.call_start_time)}",
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
    
    def _initialize_vmd(self):
        """Initialize voicemail detection components"""
        try:
            self.vmd_audio_recorder = SIPAudioRecorder(self.logger)
            self.voicemail_detector = LocalVoicemailDetector(logger=self.logger)
            self.voicemail_detector.set_audio_recorder(self.vmd_audio_recorder)
            self.voicemail_detector.set_voicemail_callback(self._handle_voicemail_detection)
            
            # Mark PJSIP resources
            self.resource_manager.register('vmd_recorder', self.vmd_audio_recorder,
                                          lambda: self._safe_cleanup(self.vmd_audio_recorder, 'cleanup'),
                                          is_pjsip=True)
            self.resource_manager.register('vmd_detector', self.voicemail_detector,
                                          lambda: self._safe_cleanup(self.voicemail_detector, 'stop_detection'),
                                          is_pjsip=False)
            
            self._log_and_collect('info', "VMD components initialized")
        except Exception as e:
            self._log_and_collect('error', f"Failed to initialize VMD: {e}")
    
    def _initialize_ringing_detection(self):
        """Initialize ringing detection components"""
        try:
            self.ringing_audio_recorder = SIPAudioRecorder(self.logger)
            self.ringing_detector = LocalRingingDetector(logger=self.logger)
            self.ringing_detector.set_audio_recorder(self.ringing_audio_recorder)
            self.ringing_detector.set_ringing_callback(self._handle_ringing_detection)
            
            # Mark PJSIP resources
            self.resource_manager.register('ringing_recorder', self.ringing_audio_recorder,
                                          lambda: self._safe_cleanup(self.ringing_audio_recorder, 'cleanup'),
                                          is_pjsip=True)
            self.resource_manager.register('ringing_detector', self.ringing_detector,
                                          lambda: self._safe_cleanup(self.ringing_detector, 'stop_detection'),
                                          is_pjsip=False)
            
            self._log_and_collect('info', "Ringing detection components initialized")
        except Exception as e:
            self._log_and_collect('error', f"Failed to initialize ringing detection: {e}")

    def _initialize_continuous_listening(self):
        """Initialize continuous listening components"""
        try:
            # Import here to avoid circular dependency
            from src.continuous_listener import ContinuousListener
            
            # Create continuous listener but don't start it yet
            # We'll start it when audio_media is available in the main flow
            # Pass None for now, will be set up properly in _execute_call_flow_safe
            self.continuous_listener = None  # Will be created when needed
            self._log_and_collect('info', "Continuous listening initialization prepared")
        except Exception as e:
            self._log_and_collect('error', f"Failed to initialize continuous listening: {e}")
    
    def set_call_details(self, phone_number, prm, lead_id, list_id, campaign_id, caller_state=None):
        """Set call details from SIP parameters"""
        self.phone_number = phone_number
        self.caller_state = caller_state
        self.call_data['phone_number'] = phone_number
        self.call_data['caller_state'] = caller_state
        self.call_data['vici_lead_id'] = lead_id
        self.call_data['vici_list_id'] = list_id
        self.call_data['vici_campaign_id'] = campaign_id

        try:
            state_info = f", State={caller_state}" if caller_state else ""
            self._log_and_collect('info', f"Call details: Phone={phone_number}{state_info}, Lead={lead_id}, List={list_id}, Campaign={campaign_id}")
            self._log_and_collect('info', f"Source: {prm.rdata.srcAddress}")
            
            # Log call to CRM immediately with phone number
            if self.crm_logger and not self.state.get('crm_logged'):
                call_id = self.crm_logger.log_call_start(self.call_data)
                if call_id:
                    self.call_data['id'] = call_id
                    self.state.set('crm_logged', True)
                    self._log_and_collect('info', f"✅ Call logged to CRM with ID: {call_id}")
                else:
                    self._log_and_collect('error', "Failed to log call start to CRM")
                    
        except:
            pass
    
    def onCallState(self, prm):
        """Handle call state changes"""
        try:
            ci = self.getInfo()
            self._log_and_collect('info', f"Call state: {ci.stateText}")
            
            if ci.state == pj.PJSIP_INV_STATE_DISCONNECTED:
                self._log_and_collect('info', "Call disconnected - initiating cleanup")
                self.state.set('is_active', False)
                
                # CRITICAL: Ensure disposition is set to NP if not explicitly set
                if not self.state.get('disposition_set'):
                    self.call_data['disposition'] = 'NP'
                    self._log_and_collect('warning', "Call disconnected without disposition - setting to NP")
                
                # Save final state to CRM before cleanup - defer to avoid PJSIP lock conflicts
                def deferred_crm_save():
                    try:
                        import time
                        pj.Endpoint.instance().libRegisterThread(f"crm_save_{time.time()}")
                        threading.current_thread()._pj_registered = True
                    except:
                        pass
                    self._save_final_state_to_crm()

                    # Stop background noise in deferred thread
                    if self.background_noise_player:
                        try:
                            self.background_noise_player.stop()
                        except Exception as e:
                            self._log_and_collect('warning', f"Background noise stop error: {e}")

                threading.Thread(target=deferred_crm_save, daemon=True).start()
                
                # CRITICAL: Don't call remove_call directly from PJSIP callback
                # Schedule it to run after this callback completes
                def deferred_removal():
                    try:
                        # REGISTER THIS THREAD WITH PJSIP!
                        pj.Endpoint.instance().libRegisterThread(f"removal_{self.getId()}")
                        threading.current_thread()._pj_registered = True
                    except Exception as e:
                        self._log_and_collect('warning', f"Failed to register removal thread: {e}")
                    
                    time.sleep(0.1)  # Let PJSIP finish its operations
                    self.account.remove_call(self)
                
                removal_thread = threading.Thread(
                    target=deferred_removal,
                    daemon=True
                )
                removal_thread.start()
                
        except pj.Error as e:
            self._log_and_collect('warning', f"PJSIP error in onCallState: {e.reason}")
            self.state.set('is_active', False)
            # Ensure we save to CRM even on error - defer to avoid PJSIP lock conflicts
            def deferred_crm_save():
                try:
                    import time
                    pj.Endpoint.instance().libRegisterThread(f"crm_save_{time.time()}")
                    threading.current_thread()._pj_registered = True
                except:
                    pass
                self._save_final_state_to_crm()
            
            threading.Thread(target=deferred_crm_save, daemon=True).start()
        except Exception as e:
            self._log_and_collect('error', f"Error in onCallState: {e}")
            self.state.set('is_active', False)
            # Ensure we save to CRM even on error - defer to avoid PJSIP lock conflicts
            def deferred_crm_save():
                try:
                    import time
                    pj.Endpoint.instance().libRegisterThread(f"crm_save_{time.time()}")
                    threading.current_thread()._pj_registered = True
                except:
                    pass
                self._save_final_state_to_crm()
            
            threading.Thread(target=deferred_crm_save, daemon=True).start()
    
    def onCallMediaState(self, prm):
        """Handle media state changes"""
        if self.state.get('conversation_started'):
            return
            
        try:
            ci = self.getInfo()
            
            for i, media in enumerate(ci.media):
                if (media.type == pj.PJMEDIA_TYPE_AUDIO and 
                    self.getMedia(i) and 
                    media.status == pj.PJSUA_CALL_MEDIA_ACTIVE):
                    
                    self._log_and_collect('info', "Media active - starting conversation")
                    self.audio_media = self.getAudioMedia(i)
                    self.state.set('media_active', True)
                    
                    # Start VMD if available
                    if self.voicemail_detector and self.state.get('is_active'):
                        try:
                            self._log_and_collect('info', "Starting VMD")
                            self.voicemail_detector.start_detection(self.audio_media)
                        except Exception as e:
                            self._log_and_collect('error', f"Failed to start VMD: {e}")
                    
                    # Start ringing detection if available (continuous throughout call)
                    if self.ringing_detector and self.state.get('is_active'):
                        try:
                            self._log_and_collect('info', "Starting continuous ringing detection")
                            self.ringing_detector.start_detection(self.audio_media)
                        except Exception as e:
                            self._log_and_collect('error', f"Failed to start ringing detection: {e}")
                    
                    # NOTE: Stereo call recording is handled by sharing continuous listener's recording
                    # Do NOT start separate recording here - it causes PJSIP deadlock
                    if self.stereo_call_recorder and self.state.get('is_active'):
                        self._log_and_collect('info', "🎙️ Stereo recorder available - will use shared recording from continuous listener")
                    else:
                        if not self.stereo_call_recorder:
                            self._log_and_collect('warning', "⚠️ No stereo call recorder available")
                        if not self.state.get('is_active'):
                            self._log_and_collect('warning', "⚠️ Call not active when trying to start stereo recording")
                    
                    # 🔥 CRITICAL FIX: Start recording IMMEDIATELY (synchronously) to prevent audio loss
                    # Under load, continuous listener initialization can take 200-2000ms
                    # Starting recorder first ensures zero audio loss
                    early_recorder = None
                    early_recording_file = None

                    if (self.state.get('is_active') and
                        hasattr(self, 'intent_detector') and
                        hasattr(self, 'parakeet_model') and
                        self.intent_detector and
                        self.parakeet_model):
                        try:
                            # STEP 1: Start recording IMMEDIATELY (blocks until PJSIP starts transmitting)
                            self._log_and_collect('info', "🔥 Starting audio recording IMMEDIATELY (zero-loss mode)")
                            early_recorder = SIPAudioRecorder(self.logger)
                            early_recording_file = early_recorder.start_recording(self.audio_media)

                            if early_recording_file:
                                self._log_and_collect('info', f"✅ Recording started synchronously: {early_recording_file}")
                                self._log_and_collect('info', "Audio is being captured NOW - no loss possible")
                            else:
                                self._log_and_collect('error', "❌ Failed to start early recording - will use cold start")
                                early_recorder = None

                            # STEP 2: Now initialize continuous listener (can take time under load)
                            from src.continuous_listener import ContinuousListener
                            self.continuous_listener = ContinuousListener(
                                self.logger,
                                self.intent_detector,
                                self.parakeet_model,
                                energy_threshold=self.global_energy_threshold,
                                rnnt_confidence_threshold=self.global_rnnt_confidence_threshold
                            )

                            # Add reference to singleton for RNNT confidence
                            if hasattr(self, 'parakeet_singleton'):
                                self.continuous_listener.parakeet_singleton = self.parakeet_singleton

                            # STEP 3: Attach the already-running recorder (warm start)
                            if early_recorder and early_recording_file:
                                self.continuous_listener.attach_to_running_recorder(early_recorder, early_recording_file)
                                self._log_and_collect('info', "🔥 Continuous listener attached to pre-started recorder (WARM START)")

                            # STEP 4: Start detection (thread will skip recorder creation in warm start)
                            if self.continuous_listener.start_detection(self.audio_media):
                                self._log_and_collect('info', "🎧 IMMEDIATE continuous listening started with ZERO audio loss")
                            else:
                                self._log_and_collect('warning', "Failed to start immediate continuous listening")
                                self.continuous_listener = None

                        except Exception as e:
                            self._log_and_collect('error', f"Immediate continuous listening error: {e}")
                    
                    # Start conversation thread
                    self._start_conversation_thread()
                    return
                    
        except Exception as e:
            self._log_and_collect('error', f"Error in onCallMediaState: {e}")
    
    def onCallTransferStatus(self, prm):
        """Handle SIP REFER transfer status responses"""
        try:
            # Mark that SIP REFER was actually sent (this callback only fires if packet was sent)
            if not self.sip_refer_sent:
                self.sip_refer_sent = True
                self._log_and_collect('info', "✓ SIP REFER confirmed sent (callback received)")

            # Log transfer status with detailed information
            status_info = f"Transfer status code: {prm.statusCode}"
            if hasattr(prm, 'reason') and prm.reason:
                status_info += f", reason: {prm.reason}"

            self._log_and_collect('info', f"SIP REFER status: {status_info}")
            
            # Store transfer details in call_data
            transfer_success = (200 <= prm.statusCode < 300)
            transfer_target = getattr(self, '_transfer_target', 'Unknown')
            
            # Update call data with transfer information
            self.call_data.update({
                'transfer_target': transfer_target,
                'transfer_status': 'SUCCESS' if transfer_success else 'FAILED',
                'transfer_response_code': prm.statusCode,
                'transfer_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'transfer_reason': getattr(prm, 'reason', '')
            })
            
            if transfer_success:
                self._log_and_collect('info', f"✅ SIP REFER successful (code: {prm.statusCode}) to {transfer_target}")
            else:
                self._log_and_collect('error', f"❌ SIP REFER failed (code: {prm.statusCode}) to {transfer_target}")
                
        except Exception as e:
            self._log_and_collect('error', f"Error in onCallTransferStatus: {e}")
    
    def _start_conversation_thread(self):
        """Start conversation processing thread"""
        if self.state.get('conversation_started'):
            return
            
        self.state.set('conversation_started', True)
        
        self.conversation_thread = threading.Thread(
            target=self._run_conversation_with_protection,
            name=f"conv_{self.getId()}",
            daemon=True
        )
        self.conversation_thread.start()
        
        # Register thread for cleanup
        self.resource_manager.register('conversation_thread', self.conversation_thread)
    
    def _run_conversation_with_protection(self):
        """Run conversation with full error protection"""
        try:
            # Register thread with PJSIP
            self._register_pjsip_thread()
            
            # Setup file logging
            self._setup_file_logger()
            
            self._log_and_collect('info', "Conversation thread started")
            
            # Initialize components
            self._initialize_all_components()
            
            # Load call flow
            self.call_flow = parse_call_flow_from_string(self.agent_config.script_content)
            if not self.call_flow:
                raise ValueError("Failed to load call flow")
            
            # Apply script overrides
            self._apply_script_overrides()
            
            # Setup speech recognition
            self._setup_speech_recognition()
            
            # Initialize audio buffer
            self.audio_buffer = AudioBuffer(sample_rate=8000, logger=self.logger, energy_threshold=self.global_energy_threshold)
            
            # Check if continuous listening is already running from onCallMediaState
            from src.config import CONTINUOUS_LISTENING_ENABLED
            if hasattr(self, 'continuous_listener') and self.continuous_listener:
                self._log_and_collect('info', "🎧 Continuous listening already active from media state")
            elif (CONTINUOUS_LISTENING_ENABLED and 
                  self.audio_media and 
                  self.intent_detector and
                  self.parakeet_model):
                # Fallback: Start continuous listening if not already started
                try:
                    from src.continuous_listener import ContinuousListener
                    self.continuous_listener = ContinuousListener(
                        self.logger,
                        self.intent_detector,
                        self.parakeet_model,
                        energy_threshold=self.global_energy_threshold,
                        rnnt_confidence_threshold=self.global_rnnt_confidence_threshold
                    )
                    # Add reference to singleton for RNNT confidence
                    if hasattr(self, 'parakeet_singleton'):
                        self.continuous_listener.parakeet_singleton = self.parakeet_singleton
                    
                    if self.continuous_listener.start_detection(self.audio_media):
                        self._log_and_collect('info', "🎧 Fallback continuous listening started before call flow")
                        time.sleep(0.1)  # Minimal delay for audio stream stability
                    else:
                        self._log_and_collect('warning', "Failed to start fallback continuous listening")
                        self.continuous_listener = None
                        
                except Exception as e:
                    self._log_and_collect('error', f"Fallback continuous listening error: {e}")
                    self.continuous_listener = None
            
            # Execute call flow
            self._execute_call_flow_safe()
            
        except Exception as e:
            self._log_and_collect('error', f"Conversation error: {e}")
            self.call_data['error'] = str(e)
            # Set disposition to NP for errors
            if not self.state.get('disposition_set'):
                self.call_data['disposition'] = 'NP'
            self._end_call_safe(CallOutcome.FAILED)
            
        finally:
            # Ensure cleanup if still active
            if self.state.get('is_active') and not self.state.get('call_transferred'):
                self._log_and_collect('info', "Conversation ended without transfer - hanging up")
                # Set disposition to NP if not set
                if not self.state.get('disposition_set'):
                    self.call_data['disposition'] = 'NP'
                self._end_call_safe(CallOutcome.HANGUP_EARLY)
    
    def _register_pjsip_thread(self):
        """Register current thread with PJSIP"""
        try:
            thread_name = threading.current_thread().name
            if not hasattr(threading.current_thread(), '_pj_registered'):
                pj.Endpoint.instance().libRegisterThread(thread_name)
                threading.current_thread()._pj_registered = True
                self._log_and_collect('info', f"Registered thread: {thread_name}")
        except Exception as e:
            self._log_and_collect('warning', f"Thread registration warning: {e}")
    
    def _setup_file_logger(self):
        """Setup file logging for this call"""
        try:
            agent_log_dir = os.path.join(LOG_DIR, self.agent_config.agent_id)
            os.makedirs(agent_log_dir, exist_ok=True)
            
            log_filename = f"{self.phone_number}-{self.getId()}.log"
            log_filepath = os.path.join(agent_log_dir, log_filename)
            
            self.file_handler = logging.FileHandler(log_filepath)
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)s] [%(agent_id)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            self.file_handler.setFormatter(formatter)
            
            self.logger.logger.addHandler(self.file_handler)
            # Mark as non-PJSIP resource
            self.resource_manager.register('file_handler', self.file_handler,
                                          lambda: self._cleanup_file_handler(),
                                          is_pjsip=False)
            
            self._log_and_collect('info', f"Logging to: {log_filepath}")
            
        except Exception as e:
            self._log_and_collect('error', f"Failed to setup file logger: {e}")
    
    def _cleanup_file_handler(self):
        """Cleanup file handler"""
        if self.file_handler:
            try:
                self.logger.logger.removeHandler(self.file_handler)
                self.file_handler.close()
            except:
                pass
    
    def _initialize_all_components(self):
        """Initialize all components with error handling"""
        try:
            # ViciDial integration
            if all([self.agent_config.server_url, 
                   self.agent_config.username, 
                   self.agent_config.password]):
                api = ViciDialAPI(
                    server_url=self.agent_config.server_url,
                    api_user=self.agent_config.username,
                    api_pass=self.agent_config.password
                )
                self.vicidial_integration = VoicebotViciDialIntegration(
                    api, self.agent_config.vicidial_campaign_id
                )
                self._log_and_collect('info', "ViciDial integration initialized")
            
            # Stereo call recorder should already be initialized in constructor
            if not self.stereo_call_recorder:
                self._log_and_collect('warning', "Stereo call recorder not initialized in constructor - creating fallback")
                self.stereo_call_recorder = StereoCallRecorder(self.logger)
                self.resource_manager.register('stereo_call_recorder', self.stereo_call_recorder,
                                              lambda: self._safe_cleanup(self.stereo_call_recorder, 'cleanup'),
                                              is_pjsip=False)
            else:
                self._log_and_collect('info', "Using stereo call recorder from constructor (race condition fix working)")
            
            # Audio components (PJSIP resources)
            self.audio_manager = SIPAudioManager(
                self.agent_config.noise_location,
                self.agent_config.background_noise_volume,
                self.logger,
                stereo_recording_callback=self.stereo_call_recorder.add_bot_audio
            )
            self.resource_manager.register('audio_manager', self.audio_manager,
                                          lambda: self._safe_cleanup(self.audio_manager, 'cleanup_all'),
                                          is_pjsip=True)  # Mark as PJSIP resource

            # Set interrupt callback for immediate DNC/NI termination during playback
            self.audio_manager.set_interrupt_callback(self._check_for_interrupt)

            self.audio_recorder = SIPAudioRecorder(self.logger)
            self.resource_manager.register('audio_recorder', self.audio_recorder,
                                          lambda: self._safe_cleanup(self.audio_recorder, 'cleanup'),
                                          is_pjsip=True)  # Mark as PJSIP resource

            # Continuous background noise player (NEW)
            self.background_noise_player = None
            if (self.agent_config.noise_location and
                self.agent_config.background_noise_volume > 0):
                try:
                    from src.sip_audio_manager import ContinuousBackgroundNoisePlayer
                    self.background_noise_player = ContinuousBackgroundNoisePlayer(
                        noise_path=self.agent_config.noise_location,
                        noise_volume=self.agent_config.background_noise_volume,
                        logger=self.logger
                    )
                    self.resource_manager.register('background_noise_player', self.background_noise_player,
                                                  lambda: self._safe_cleanup(self.background_noise_player, 'cleanup'),
                                                  is_pjsip=True)  # Mark as PJSIP resource
                    self._log_and_collect('info', f"🔊 Continuous background noise player initialized (volume={self.agent_config.background_noise_volume:.2f})")
                except Exception as e:
                    self._log_and_collect('warning', f"Failed to initialize continuous background noise: {e}")
                    self.background_noise_player = None  # Non-fatal, call can proceed
            
            # Transfer manager
            transfer_server_ip = self.source_ip or self.agent_config.server_ip
            self._log_and_collect('info', f"Initializing transfer manager for server: {transfer_server_ip} (source_ip: {self.source_ip})")
            self.transfer_manager = ViciDialApiTransfer(
                self.agent_config.did_transfer_qualified,
                self.agent_config.did_transfer_hangup,
                transfer_server_ip,
                self.logger
            )
            
            # Other components (non-PJSIP)
            self.outcome_handler = CallOutcomeHandler(self.vicidial_integration, self.logger)
            self.intent_detector = IntentDetector(hp_phrases=self.agent_config.honey_pot_sentences)
            
            # Initialize Qwen detector if available
            from src.config import USE_QWEN_INTENT
            if self.qwen_singleton and USE_QWEN_INTENT:
                self.qwen_detector = self.qwen_singleton.get_detector(self.logger)
                if self.qwen_detector:
                    self._log_and_collect('info', "Qwen intent detector initialized")
                else:
                    self._log_and_collect('warning', "Qwen detector not available - using keywords")
            
            # CRM logger might already be initialized
            if not self.crm_logger:
                self.crm_logger = SuiteCRMLogger(self.agent_config, self.logger, self)
            
            self._log_and_collect('info', "All components initialized")
            
        except Exception as e:
            self._log_and_collect('error', f"Component initialization failed: {e}")
            raise
    
    def _apply_script_overrides(self):
        """Apply script configuration overrides"""
        if 'max_clarification_retries' in self.call_flow:
            self.max_clarifications = int(self.call_flow['max_clarification_retries'])
            self._log_and_collect('info', f"Max clarifications: {self.max_clarifications}")
        
        if 'max_silence_retries' in self.call_flow:
            self.max_consecutive_silences = int(self.call_flow['max_silence_retries'])
            self._log_and_collect('info', f"Max silences: {self.max_consecutive_silences}")
    
    def _setup_speech_recognition(self):
        """Setup speech recognition with Parakeet singleton model"""
        try:
            import torch
            from src.config import USE_GPU, USE_LOCAL_MODEL, MODEL_PATH
            
            # Set device
            if USE_GPU and torch.cuda.is_available():
                self.device = 'cuda'
                self._log_and_collect('info', f"Using GPU for ASR: {torch.cuda.get_device_name(0)}")
            else:
                self.device = 'cpu'
                self._log_and_collect('info', "Using CPU for ASR")
            
            if self.parakeet_singleton:
                self.parakeet_model = self.parakeet_singleton.get_model(self.logger)
                if self.parakeet_model:
                    self._log_and_collect('info', "Speech recognition initialized with Parakeet TDT (singleton)")
                else:
                    self._log_and_collect('error', "Failed to get Parakeet model")
            else:
                # Fallback - load directly
                import nemo.collections.asr as nemo_asr
                
                self._log_and_collect('info', "Loading Parakeet model (fallback)...")
                if USE_LOCAL_MODEL:
                    self.parakeet_model = nemo_asr.models.ASRModel.restore_from(MODEL_PATH)
                else:
                    self.parakeet_model = nemo_asr.models.ASRModel.from_pretrained(
                        model_name="nvidia/parakeet-tdt-0.6b"
                    )
                self.parakeet_model = self.parakeet_model.to(self.device)
                self.parakeet_model.eval()
                self._log_and_collect('info', "Speech recognition initialized with Parakeet TDT (fallback)")
                    
        except Exception as e:
            self._log_and_collect('error', f"Speech recognition setup failed: {e}")
    
    def _execute_call_flow_safe(self):
        """Execute call flow with comprehensive error handling"""
        self._log_and_collect('info', f"Executing flow: {self.call_flow.get('name', 'Unknown')}")
        
        # If continuous listening wasn't started earlier, try to start it now
        # This is a fallback for edge cases
        from src.config import CONTINUOUS_LISTENING_ENABLED
        if (CONTINUOUS_LISTENING_ENABLED and 
            not self.continuous_listener and
            self.audio_media and 
            self.intent_detector and
            self.parakeet_model):
            try:
                from src.continuous_listener import ContinuousListener
                self.continuous_listener = ContinuousListener(
                    self.logger,
                    self.intent_detector,
                    self.parakeet_model,
                    energy_threshold=self.global_energy_threshold
                )
                # Add reference to singleton for RNNT confidence
                if hasattr(self, 'parakeet_singleton'):
                    self.continuous_listener.parakeet_singleton = self.parakeet_singleton
                
                if self.continuous_listener.start_detection(self.audio_media):
                    self._log_and_collect('info', "🎧 Fallback continuous listening started")
                    time.sleep(0.2)
                else:
                    self._log_and_collect('warning', "Failed to start fallback continuous listening")
                    self.continuous_listener = None
                    
            except Exception as e:
                self._log_and_collect('error', f"Fallback continuous listening error: {e}")
                self.continuous_listener = None
        else:
            if CONTINUOUS_LISTENING_ENABLED and self.continuous_listener:
                self._log_and_collect('info', "🎧 Using early-initialized continuous listening")
            elif CONTINUOUS_LISTENING_ENABLED:
                missing = []
                if not self.audio_media: missing.append("audio_media")
                if not self.intent_detector: missing.append("intent_detector") 
                if not self.parakeet_model: missing.append("parakeet_model")
                self._log_and_collect('warning', f"Continuous listening disabled - missing: {missing}")

        # Ensure stereo recording is started (fallback if onCallMediaState didn't start it)
        if (self.stereo_call_recorder and 
            self.audio_media and 
            not self.stereo_call_recorder.is_recording and
            self.state.get('is_active')):
            try:
                # First, try to use continuous listener's recording file if available
                continuous_recording_file = None
                if self.continuous_listener:
                    continuous_recording_file = self.continuous_listener.get_recording_file_path()
                
                if continuous_recording_file:
                    self._log_and_collect('info', f"🎙️ [FALLBACK] Using continuous listener's recording file: {continuous_recording_file}")
                    success = self.stereo_call_recorder.use_existing_recording(continuous_recording_file)
                    if success:
                        self._log_and_collect('info', "✅ [FALLBACK] Stereo recorder using shared recording file")
                    else:
                        self._log_and_collect('error', "❌ [FALLBACK] Failed to use shared recording - cannot create independent recording (would deadlock)")
                else:
                    self._log_and_collect('error', "🎙️ [FALLBACK] No continuous listener recording available - stereo recording will not work")
                    self._log_and_collect('warning', "Note: Independent stereo recording disabled to prevent PJSIP deadlock")
            except Exception as e:
                self._log_and_collect('error', f"❌ [FALLBACK] Exception starting stereo recording: {e}")

        # Start continuous background noise immediately after call is answered
        if self.background_noise_player and self.audio_media:
            try:
                success = self.background_noise_player.start(self.audio_media)
                if success:
                    self._log_and_collect('info', "🔊 Continuous background noise started")
                else:
                    self._log_and_collect('warning', "Background noise failed to start - call continues")
            except Exception as e:
                self._log_and_collect('warning', f"Background noise start error: {e} - call continues")

        self.current_step = self._get_initial_step()
        action_taken = False

        # Initial delay
        time.sleep(0.1)
        
        while (self.current_step != 'exit' and 
               self.state.get('is_active') and 
               not self.state.get('emergency_cleanup')):
            
            # Check ringing first (highest priority)
            if self.state.get('ringing_triggered'):
                self._log_and_collect('warning', "Ringing triggered - ending flow immediately")
                break
            
            # Check continuous listening intents (high priority)
            if self.continuous_listener and self.continuous_listener.is_intent_triggered():
                intent_type = self.continuous_listener.get_triggered_intent()
                self._log_and_collect('warning', f"🚫 {intent_type} intent detected via continuous listening - ending call")
                
                if intent_type == "DNC":
                    self.call_data['disposition'] = 'DNC'
                elif intent_type == "NI":
                    self.call_data['disposition'] = 'NI'
                elif intent_type == "OBSCENITY":
                    self.call_data['disposition'] = 'DNC'  # Obscenity is DNC
                elif intent_type == "HP":
                    self.call_data['disposition'] = 'HP'  # Hold/Press keywords
                elif intent_type == "CLBK":
                    self.call_data['disposition'] = 'CLBK'  # Callback requests
                else:
                    self.call_data['disposition'] = 'DNC'  # Default fallback
                
                self.state.set('disposition_set', True)
                self._end_call_safe(CallOutcome.NEGATIVE_INTENT)
                break
            
            # Check VMD
            if self.state.get('vmd_triggered'):
                self._log_and_collect('warning', "VMD triggered - ending flow")
                break
            
            # Get current step
            step = self.call_flow['steps'].get(self.current_step)
            if not step:
                self._log_and_collect('error', f"Step '{self.current_step}' not found")
                break
            
            self._log_and_collect('info', f"Step: {self.current_step}")
            
            # Process step
            try:
                if not self._process_step(step):
                    action_taken = True
                    break
            except Exception as e:
                self._log_and_collect('error', f"Step processing error: {e}")
                # Set disposition to NP for errors
                if not self.state.get('disposition_set'):
                    self.call_data['disposition'] = 'NP'
                break
            
            # Check if call still active
            if not self.state.get('is_active'):
                self._log_and_collect('info', "Call disconnected during flow")
                # Set disposition to NP if not set
                if not self.state.get('disposition_set'):
                    self.call_data['disposition'] = 'NP'
                break
            
            # Small delay between steps
            time.sleep(0.2)
        
        # Handle flow completion
        if not action_taken and self.state.get('is_active'):
            if self.state.get('ringing_triggered'):
                # Ringing already handled in callback, just return
                return
            elif self.state.get('vmd_triggered'):
                self.call_data['disposition'] = 'A'
                self.state.set('disposition_set', True)
                self._end_call_safe(CallOutcome.VOICEMAIL)
            else:
                # Set disposition to NP if not explicitly set
                if not self.state.get('disposition_set'):
                    self.call_data['disposition'] = 'NP'
                self._end_call_safe(CallOutcome.HANGUP_EARLY)
    
    def _get_initial_step(self):
        """
        Determine the initial step. 
        Now that we handle time-based and state-based audio file selection with flags,
        we simply return the standard 'start' step or check for a configured entry point.
        """
        # Check if we have call flow steps
        if 'steps' not in self.call_flow:
            return 'start'
        
        # Look for common entry point names
        for potential_start in ['hello', 'introduction', 'start']:
            if potential_start in self.call_flow['steps']:
                self._log_and_collect('debug', f"Using entry step: {potential_start}")
                return potential_start
        
        # If none of the common entry points exist, use the first step
        if self.call_flow['steps']:
            first_step = list(self.call_flow['steps'].keys())[0]
            self._log_and_collect('debug', f"Using first available step: {first_step}")
            return first_step
        
        # Ultimate fallback
        return 'start'
    
    def _process_step(self, step):
        """Process a single call flow step"""
        
        # Track current step and question for Qwen
        self.current_step_id = step.get('id', self.current_step)
        self.current_question_text = step.get('text', '')
        
        if self.current_question_text:
            self._log_and_collect('info', f"Question: {self.current_question_text[:100]}...")
        
        # Set disposition if specified
        if 'disposition' in step:
            # Special handling for SALE disposition with transfer action
            if step['disposition'] == 'SALE' and step.get('action') == 'transfer':
                # Set SALE immediately to satisfy state logic, will be overridden based on transfer result
                self.call_data['disposition'] = 'SALE'
                self.call_data['intended_disposition'] = 'SALE'  # Track that this needs transfer-based override
                self.state.set('disposition_set', True)
                self._log_and_collect('info', f"Disposition: {step['disposition']} (pending transfer result)")
            else:
                # Original logic for all other cases
                self.call_data['disposition'] = step['disposition']
                self.state.set('disposition_set', True)  # Mark that disposition was explicitly set
                self._log_and_collect('info', f"Disposition: {step['disposition']}")
        
        # Play audio if specified
        if 'audio_file' in step and self.audio_manager and self.audio_media:
            if not self._play_audio_safe(step['audio_file'], step):
                # Check if disposition was already set by interrupt handling (DNC/NI)
                if self.state.get('disposition_set'):
                    # Disposition already set (likely DNC/NI from continuous listening)
                    current_disposition = self.call_data.get('disposition', 'UNKNOWN')
                    self._log_and_collect('info', f"Playback failed but disposition already set to: {current_disposition}")
                else:
                    # Set NP disposition only if no disposition was set
                    self.call_data['disposition'] = 'NP'
                    self._log_and_collect('warning', "Playback failed - setting disposition to NP")
                return False
            
            # Add delay after audio playback to prevent echo
            # REMOVED: This delay prevents immediate response collection
            # from src.config import AUDIO_STOP_DELAY
            #time.sleep(0.1)
        
        # Check if still active
        if not self.state.get('is_active'):
            # Set NP disposition only if no disposition was set
            if not self.state.get('disposition_set'):
                self.call_data['disposition'] = 'NP'
            return False
        
        # Handle action
        if 'action' in step:
            action = step['action']
            self._log_and_collect('info', f"Action: {action}")
            
            if action == 'transfer':
                self._handle_transfer_safe()
                return False
            elif action == 'hangup':
                self._end_call_safe(CallOutcome.HANGUP_SCRIPTED)
                return False
        
        # Handle pause duration if specified
        if 'pause_duration' in step:
            pause_seconds = step['pause_duration']
            if pause_seconds > 0:
                self._log_and_collect('info', f"Pausing for {pause_seconds} seconds")
                time.sleep(pause_seconds)
        
        # Handle response waiting
        if step.get('wait_for_response', False):
            return self._handle_user_response_safe(step)
        else:
            self.current_step = step.get('next', 'exit')
            return True
    
    def _play_audio_safe(self, audio_file, step=None):
        """Play audio with continuous recording (no recording stops/starts)"""
        with self.audio_transition_lock:
            try:
                # Mark playback start for continuous listening
                if self.continuous_listener:
                    self.continuous_listener.mark_playback_start()
                
                # DO NOT stop any recording - let continuous listener capture everything
                # This allows for true overlapping audio like a real phone call
                
                # Check for greetings and us_states flags in step
                greetings = step.get('greetings', False) if step else False
                us_states = step.get('us_states', False) if step else False
                time_period = None
                state_code = None
                
                # Get time period if greetings flag is set
                if greetings:
                    try:
                        from src.timezone_utils import get_time_period, STATE_NAME_TO_CODE
                        if hasattr(self, 'caller_state') and self.caller_state:
                            # Convert full state name to code for timezone lookup
                            state_for_time = self.caller_state
                            if len(self.caller_state) > 2:
                                state_code_for_time = STATE_NAME_TO_CODE.get(self.caller_state, None)
                                if state_code_for_time:
                                    state_for_time = state_code_for_time
                            time_period = get_time_period(state_for_time)
                        else:
                            time_period = 'morning'  # Default fallback
                    except Exception as e:
                        self._log_and_collect('warning', f"Error getting time period: {e}")
                        time_period = 'morning'
                
                # Get state code if us_states flag is set
                if us_states:
                    if hasattr(self, 'caller_state') and self.caller_state:
                        # Convert full state name to two-letter code
                        from src.timezone_utils import STATE_NAME_TO_CODE
                        state_code = STATE_NAME_TO_CODE.get(self.caller_state, None)
                        if not state_code:
                            # Try as-is in case it's already a two-letter code
                            if len(self.caller_state) == 2:
                                state_code = self.caller_state.upper()
                            else:
                                self._log_and_collect('warning', f"Unknown state name: {self.caller_state}")
                                state_code = None
                        else:
                            self._log_and_collect('debug', f"Converted state: {self.caller_state} -> {state_code}")
                    else:
                        self._log_and_collect('warning', "us_states flag set but no caller_state available")
                
                audio_path = get_audio_path_for_agent(
                    audio_file, 
                    self.agent_config.voice_location,
                    greetings=greetings,
                    us_states=us_states,
                    time_period=time_period,
                    state_code=state_code
                )
                
                # Log which audio file variant is being used
                selected_file = os.path.basename(audio_path)
                if selected_file != audio_file:
                    self._log_and_collect('info', f"Using audio variant: {selected_file} (original: {audio_file})")
                
                if not audio_path or not os.path.exists(audio_path):
                    self._log_and_collect('error', f"Audio file not found: {audio_path}")
                    return False
                
                self.conversation_log.append(f"Bot: {os.path.basename(audio_path)}")
                
                # Play directly in current thread (which is already registered with PJSIP)
                try:
                    played_bytes = self.audio_manager.play_audio_file(audio_path, self.audio_media)
                    # Note: Bot audio is now captured by stereo_call_recorder.add_bot_audio() callback
                    # if played_bytes:
                    #     self.all_audio_chunks.append(played_bytes)
                    
                    # Check if playback was interrupted by DNC/NI
                    if self._check_for_interrupt():
                        # Get the intent type and set appropriate disposition BEFORE returning
                        if self.continuous_listener and self.continuous_listener.is_intent_triggered():
                            intent_type = self.continuous_listener.get_triggered_intent()
                            self._log_and_collect('warning', f"🚫 Playback interrupted by {intent_type} detection")
                            
                            # Set proper disposition based on intent type
                            if intent_type == "DNC":
                                self.call_data['disposition'] = 'DNC'
                            elif intent_type == "NI":
                                self.call_data['disposition'] = 'NI'
                            elif intent_type == "OBSCENITY":
                                self.call_data['disposition'] = 'DNC'  # Obscenity is DNC
                            elif intent_type == "HP":
                                self.call_data['disposition'] = 'HP'  # Hold/Press keywords
                            elif intent_type == "CLBK":
                                self.call_data['disposition'] = 'CLBK'  # Callback requests
                            else:
                                self.call_data['disposition'] = 'DNC'  # Default fallback
                            
                            self.state.set('disposition_set', True)  # Mark disposition as explicitly set
                        
                        self._log_and_collect('warning', "🚫 Audio playback was interrupted by DNC/NI - ending call immediately")
                        return False
                    
                    # Track playback end timestamp for echo prevention
                    self.last_playback_end = time.time()
                    self._log_and_collect('debug', f"Playback completed at {self.last_playback_end} (continuous recording active)")
                    
                    return True
                except Exception as e:
                    self._log_and_collect('error', f"Audio playback error: {e}")
                    return False
                    
            except pj.Error as e:
                # Use numeric error codes
                if e.status in [PJSIP_ERROR_NOT_FOUND, PJSIP_ERROR_INVALID]:
                    self._log_and_collect('info', "Call disconnected during playback")
                else:
                    self._log_and_collect('error', f"PJSIP error: {e.reason}")
                return False
            except Exception as e:
                self._log_and_collect('error', f"Audio playback error: {e}")
                return False
            finally:
                # Always mark playback end for continuous listening
                if self.continuous_listener:
                    self.continuous_listener.mark_playback_end()
    

    def _handle_user_response_safe(self, step):
        """Handle user response with timeout protection"""
        try:
            # Check if ringing was detected first (highest priority)
            if self.state.get('ringing_triggered'):
                self._log_and_collect('info', "Ringing detected - call already ending")
                return False
            
            # Check if voicemail was detected
            if self.state.get('vmd_triggered'):
                self._log_and_collect('info', "Voicemail detected - ending with A disposition")
                self.call_data['disposition'] = 'A'
                self.state.set('disposition_set', True)
                self._end_call_safe(CallOutcome.VOICEMAIL)
                return False
            
            timeout = step.get('timeout', 10)
            user_response = self._listen_for_response_safe(timeout)
            
            if not self.state.get('is_active'):
                return False
            
            # Check again after listening in case ringing or VMD triggered during listening
            if self.state.get('ringing_triggered'):
                self._log_and_collect('info', "Ringing detected during response - call already ending")
                return False
            
            if self.state.get('vmd_triggered'):
                self._log_and_collect('info', "Voicemail detected during response - ending with A disposition")
                self.call_data['disposition'] = 'A'
                self.state.set('disposition_set', True)
                self._end_call_safe(CallOutcome.VOICEMAIL)
                return False
            
            if user_response == "TIMEOUT_NO_RESPONSE":
                self.conversation_log.append("User: <silence>")
                self.consecutive_silence_count += 1
                
                if self.consecutive_silence_count >= self.max_consecutive_silences:
                    self._log_and_collect('warning', "Max silences exceeded")
                    
                    # Check if audio activity was detected to differentiate between
                    # truly silent calls vs calls with audio but no speech
                    has_audio_activity = False
                    if self.ringing_detector:
                        try:
                            has_audio_activity = self.ringing_detector.has_audio_activity()
                        except Exception as e:
                            self._log_and_collect('warning', f"Failed to check audio activity: {e}")
                    
                    if has_audio_activity:
                        self.call_data['disposition'] = 'DAIR 2'
                        self._log_and_collect('info', "Setting disposition to DAIR 2 (audio detected but no speech)")
                    else:
                        self.call_data['disposition'] = 'DAIR'
                        self._log_and_collect('info', "Setting disposition to DAIR (no audio activity detected)")
                    
                    self.state.set('disposition_set', True)
                    self._end_call_safe(CallOutcome.NOT_QUALIFIED)
                    return False
                
                self.current_step = step.get('timeout_next', self.current_step)
                return True
            
            # Reset silence count
            self.consecutive_silence_count = 0
            self.conversation_log.append(f"User: {user_response}")
            
            # Check for negative intent
            if self.intent_detector and self.intent_detector.negative_detected:
                # Set appropriate disposition based on intent type
                if self.intent_detector.negative_intent == 'not_interested':
                    self.current_step = 'not_interested'
                elif self.intent_detector.negative_intent in ['do_not_call', 'obscenity', 'hold_press', 'callback']:
                    # Set disposition before routing to graceful_exit
                    if self.intent_detector.negative_intent == 'hold_press':
                        self.call_data['disposition'] = 'HP'
                    elif self.intent_detector.negative_intent == 'callback':
                        self.call_data['disposition'] = 'CLBK'
                    elif self.intent_detector.negative_intent == 'obscenity':
                        self.call_data['disposition'] = 'DNC'
                    elif self.intent_detector.negative_intent == 'do_not_call':
                        self.call_data['disposition'] = 'DNC'
                    self.state.set('disposition_set', True)
                    self._log_and_collect('info', f"Disposition set to: {self.call_data['disposition']} based on intent: {self.intent_detector.negative_intent}")
                    self.current_step = 'graceful_exit'
                return True
            
            # Use Qwen for intent detection (no fallback to keywords)
            from src.config import USE_QWEN_INTENT
            if USE_QWEN_INTENT:
                next_step = self._evaluate_conditions_with_qwen(step, user_response)
            else:
                next_step = self._evaluate_conditions(step, user_response)
            
            if next_step is None:
                self.clarification_count += 1
                
                if self.clarification_count >= self.max_clarifications:
                    self._log_and_collect('warning', "Max clarifications exceeded")
                    self.call_data['disposition'] = 'DNC'
                    self.state.set('disposition_set', True)
                    self._end_call_safe(CallOutcome.NOT_QUALIFIED)
                    return False
                
                self.current_step = step.get('no_match_next', self.current_step)
            else:
                self.clarification_count = 0
                self.current_step = next_step
            
            return True
            
        except Exception as e:
            self._log_and_collect('error', f"Response handling error: {e}")
            # Set NP disposition for errors
            if not self.state.get('disposition_set'):
                self.call_data['disposition'] = 'NP'
            return False
    
    def _listen_for_response_safe(self, timeout):
        """Listen for user response using continuous audio stream - exactly like sequential but from continuous recording"""
        
        if not self.continuous_listener or not self.continuous_listener.is_active:
            self._log_and_collect('warning', "No continuous listener available - cannot extract response")
            return "TIMEOUT_NO_RESPONSE"
        
        if not self.parakeet_model:
            self._log_and_collect('warning', "No Parakeet model available for speech recognition")
            return "TIMEOUT_NO_RESPONSE"
        
        try:
            # Clear the continuous listener's buffer for fresh response detection
            # REMOVED: No wait needed for immediate response collection
            time.sleep(0.3)
            # Reset the VAD state for clean response detection
            self.continuous_listener.clear_audio_buffer()
            
            # Flush any pending transcriptions before switching to main listener
            try:
                flushed = self.continuous_listener.flush_pending_transcription("main_listener_switch")
                if flushed:
                    # Collect the flushed transcriptions immediately
                    self._collect_continuous_transcriptions(since_time=time.time() - 5.0)
                    self._log_and_collect('info', f"Flushed {len(flushed)} pending transcriptions before main listener")
            except Exception as e:
                self._log_and_collect('warning', f"Error flushing pending transcriptions: {e}")
            
            # Set flag to prevent continuous listener from interfering
            self.continuous_listener.main_flow_listening = True
            
            start_time = time.time()
            last_check_time = time.time()
            audio_accumulator = bytearray()
            
            self._log_and_collect('debug', f"Starting response listening (timeout={timeout}s)")
            
            while (time.time() - start_time) < timeout and self.state.get('is_active'):
                # Check VMD
                if self.state.get('vmd_triggered'):
                    return "TIMEOUT_NO_RESPONSE"
                
                # Get new audio data from continuous stream every 100ms
                current_time = time.time()
                if current_time - last_check_time >= 0.1:  # Check every 100ms
                    # Get current audio buffer state
                    has_audio, audio_bytes = self.continuous_listener.get_audio_buffer_state()
                    
                    if has_audio and audio_bytes:
                        # Get only new audio since last check
                        new_audio_len = len(audio_bytes) - len(audio_accumulator)
                        if new_audio_len > 0:
                            new_audio = audio_bytes[len(audio_accumulator):]
                            audio_accumulator.extend(new_audio)
                            
                            # Check if we should process based on VAD
                            should_process = self.continuous_listener.add_audio_chunk_for_processing(new_audio)
                            
                            if should_process and len(audio_accumulator) > 1600:  # At least 100ms
                                # Process accumulated audio
                                self._log_and_collect('debug', f"Processing {len(audio_accumulator)} bytes of audio")
                                
                                result = self._process_audio_with_parakeet_improved(bytes(audio_accumulator))
                                
                                if result:
                                    if isinstance(result, tuple):
                                        text, confidence = result
                                    else:
                                        text, confidence = result, None
                                    
                                    if text and len(text.strip()) > 0:
                                        # Echo detection
                                        if self._is_likely_echo(text):
                                            self._log_and_collect('warning', f"Ignoring likely echo: '{text}'")
                                            # Clear and continue
                                            audio_accumulator.clear()
                                            self.continuous_listener.clear_audio_buffer()
                                            continue
                                        
                                        # Log with confidence score
                                        conf_str = f" (confidence: {confidence:.2f})" if confidence is not None else ""
                                        self._log_and_collect('info', f"User: '{text}'{conf_str}")
                                        
                                        # Check intent
                                        if self.intent_detector:
                                            self.intent_detector.detect_intent(text)
                                        
                                        # Clear flag before returning
                                        self.continuous_listener.main_flow_listening = False
                                        return text
                                
                                # Clear accumulator after processing
                                audio_accumulator.clear()
                                self.continuous_listener.clear_audio_buffer()
                    
                    last_check_time = current_time
                
                time.sleep(0.05)  # Small sleep to prevent CPU spinning
            
            # Process any remaining audio at timeout
            if len(audio_accumulator) > 1600:  # At least 100ms
                result = self._process_audio_with_parakeet_improved(bytes(audio_accumulator))
                if result:
                    if isinstance(result, tuple):
                        text, confidence = result
                    else:
                        text, confidence = result, None
                    
                    if text and len(text.strip()) > 0:
                        if not self._is_likely_echo(text):
                            conf_str = f" (confidence: {confidence:.2f})" if confidence is not None else ""
                            self._log_and_collect('info', f"User (final): '{text}'{conf_str}")
                            
                            if self.intent_detector:
                                self.intent_detector.detect_intent(text)
                            
                            # Clear flag before returning
                            self.continuous_listener.main_flow_listening = False
                            return text
            
            # If no response found within timeout
            self._log_and_collect('debug', "No response found within timeout")
            return "TIMEOUT_NO_RESPONSE"
                
        except Exception as e:
            self._log_and_collect('error', f"Continuous listener response extraction error: {e}")
            return "TIMEOUT_NO_RESPONSE"
        finally:
            # Clear the flag
            if self.continuous_listener:
                self.continuous_listener.main_flow_listening = False
    
    def _check_for_interrupt(self):
        """Check if DNC/NI intent was detected and should interrupt playback"""
        if self.continuous_listener and self.continuous_listener.is_intent_triggered():
            return True
        
        # Also check the main intent detector
        if self.intent_detector and hasattr(self.intent_detector, 'negative_detected') and self.intent_detector.negative_detected:
            return True
            
        return False
    
    def _is_likely_echo(self, text):
        """Check if transcribed text is likely bot echo based on timing only"""
        if not text or not text.strip():
            return False
        
        # Simple timing-based echo detection
        time_since_playback = time.time() - self.last_playback_end if self.last_playback_end > 0 else float('inf')
        
        # If transcription occurs too close to playback, likely echo
        if time_since_playback < 0.8:  # Within 800ms of playback
            self._log_and_collect('warning', f"Echo detected - {time_since_playback:.3f}s after playback: '{text}'")
            return True
        
        return False
    
    def _process_audio_with_parakeet_improved(self, audio_bytes):
        """Process audio with Parakeet TDT model with improved preprocessing and fixed dtype"""
        try:
            import torch
            import tempfile
            
            # Convert to numpy array - use float64 for processing
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float64)
            
            # Normalize to [-1, 1]
            audio_array = audio_array / 32768.0
            
            # Apply pre-emphasis filter to boost high frequencies
            pre_emphasis = 0.97
            audio_array = np.append(audio_array[0], audio_array[1:] - pre_emphasis * audio_array[:-1]).copy()  # Fix negative stride
            
            # Apply bandpass filter (300Hz - 3400Hz for speech)
            nyquist = 4000  # Half of 8kHz sample rate
            low = 300 / nyquist
            high = 3400 / nyquist
            b, a = scipy_signal.butter(5, [low, high], btype='band')
            audio_array = scipy_signal.filtfilt(b, a, audio_array).copy()  # Fix negative stride
            
            # Normalize audio level
            max_val = np.max(np.abs(audio_array))
            if max_val > 0:
                audio_array = audio_array * (0.9 / max_val)
            
            # Ensure contiguous array before converting to tensor
            audio_array = np.ascontiguousarray(audio_array)
            
            # Convert to tensor with float64 dtype
            audio_tensor = torch.from_numpy(audio_array).to(torch.float64).unsqueeze(0)
            
            # Resample from 8kHz to 16kHz with better quality
            # First convert to float32 for torchaudio
            audio_tensor_32 = audio_tensor.to(torch.float32)
            resampler = torchaudio.transforms.Resample(
                orig_freq=8000,
                new_freq=16000,
                resampling_method='sinc_interp_hann'
            )
            audio_16k = resampler(audio_tensor_32)
            
            # Apply noise reduction using spectral subtraction
            audio_16k_np = audio_16k.squeeze().numpy().copy()  # Ensure contiguous after squeeze
            
            # Simple spectral subtraction for noise reduction
            # Estimate noise from first 100ms (assuming it's mostly silence/noise)
            noise_sample_count = min(1600, len(audio_16k_np) // 10)
            if noise_sample_count > 0 and len(audio_16k_np) > noise_sample_count:
                noise_profile = audio_16k_np[:noise_sample_count]
                
                # Apply FFT
                fft_size = 512
                hop_size = 256
                
                # Pad signal if necessary
                pad_amount = fft_size - (len(audio_16k_np) % fft_size)
                if pad_amount != fft_size:
                    audio_16k_np = np.pad(audio_16k_np, (0, pad_amount), mode='constant')
                
                # Estimate noise spectrum
                noise_fft = np.fft.rfft(noise_profile[:fft_size])
                noise_power = np.abs(noise_fft) ** 2
                
                # Process full signal in frames
                num_frames = (len(audio_16k_np) - fft_size) // hop_size + 1
                processed_frames = []
                
                for i in range(num_frames):
                    start = i * hop_size
                    end = start + fft_size
                    frame = audio_16k_np[start:end]
                    
                    frame_fft = np.fft.rfft(frame)
                    frame_power = np.abs(frame_fft) ** 2
                    frame_phase = np.angle(frame_fft)
                    
                    # Spectral subtraction with oversubtraction factor
                    alpha = 2.0  # Oversubtraction factor
                    clean_power = frame_power - alpha * noise_power
                    clean_power = np.maximum(clean_power, 0.1 * frame_power)
                    
                    # Reconstruct
                    clean_fft = np.sqrt(clean_power) * np.exp(1j * frame_phase)
                    clean_frame = np.fft.irfft(clean_fft)
                    
                    processed_frames.append(clean_frame[:hop_size])
                
                # Combine frames and ensure contiguous
                if processed_frames:
                    audio_16k_np = np.concatenate(processed_frames).copy()
            
            # Ensure proper shape and dtype for saving
            audio_16k_save = audio_16k_np.astype(np.float32)
            
            # Normalize again
            max_val = np.max(np.abs(audio_16k_save))
            if max_val > 0:
                audio_16k_save = audio_16k_save * (0.95 / max_val)
            
            # Save to temporary file with proper format
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                # Save with soundfile for better compatibility
                sf.write(
                    tmp_file.name,
                    audio_16k_save,
                    16000,
                    subtype='PCM_16'
                )
                tmp_path = tmp_file.name
            
            try:
                # Run inference with Parakeet using thread-safe method with native confidence
                with torch.no_grad():
                    # Use new transcribe_with_confidence method for both TDT and RNNT
                    if self.parakeet_singleton and hasattr(self.parakeet_singleton, 'transcribe_with_confidence'):
                        # Use new native confidence method
                        text, confidence = self.parakeet_singleton.transcribe_with_confidence(
                            tmp_path,
                            batch_size=1,
                            num_workers=0,
                            verbose=False
                        )
                        
                        # Log model type and confidence for debugging
                        model_type = self.parakeet_singleton.get_model_type()
                        if text:
                            self._log_and_collect('debug', f"Parakeet {model_type}: '{text}' (confidence: {confidence:.3f})")
                        
                        # Apply proper confidence thresholds based on model type
                        if model_type == "RNNT":
                            min_confidence = self.global_rnnt_confidence_threshold
                        else:
                            # TDT uses old score-based system
                            min_confidence = 100.0
                        
                        if confidence < min_confidence:
                            self._log_and_collect('debug', f"Low confidence ({confidence:.3f} < {min_confidence}), rejecting: '{text}'")
                            return None
                        
                    else:
                        # Fallback to old method
                        if self.parakeet_singleton:
                            transcriptions = self.parakeet_singleton.transcribe_safe(
                                [tmp_path],
                                batch_size=1,
                                return_hypotheses=False,
                                num_workers=0,
                                verbose=False
                            )
                        else:
                            transcriptions = self.parakeet_model.transcribe(
                                [tmp_path],
                                batch_size=1,
                                return_hypotheses=False,
                                num_workers=0,
                                verbose=False
                            )
                        
                        if transcriptions and len(transcriptions) > 0:
                            hypothesis = transcriptions[0]
                            
                            # Extract text
                            if hasattr(hypothesis, 'text'):
                                text = hypothesis.text
                            else:
                                text = str(hypothesis)
                            
                            # Extract confidence (old TDT method)
                            confidence = 0.0
                            if hasattr(hypothesis, 'score'):
                                confidence = hypothesis.score
                            elif hasattr(hypothesis, 'confidence'):
                                confidence = hypothesis.confidence
                            
                            # Apply TDT confidence threshold
                            if confidence < 100.0:
                                return None
                        else:
                            return None
                    
                    # Clean up text if we have it
                    if text:
                        text = text.strip()
                        # Remove common ASR artifacts
                        import re
                        text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
                        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                        
                        # Only filter out very short transcriptions (likely noise)
                        if len(text.strip()) < 2:
                            return None
                        
                        # Return text and confidence if it has actual content
                        if text and len(text) > 0:
                            return text, confidence
                        
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
            return None
            
        except Exception as e:
            self._log_and_collect('error', f"Parakeet processing error: {e}")
            return None
    
    def _evaluate_conditions(self, step, user_response):
        """Evaluate step conditions against user response"""
        # Check for negative intent override
        if self.intent_detector and self.intent_detector.negative_detected:
            if self.intent_detector.negative_intent == 'not_interested':
                return 'not_interested'
            elif self.intent_detector.negative_intent in ['do_not_call', 'obscenity', 'hold_press', 'callback']:
                # Set appropriate disposition based on intent type
                if self.intent_detector.negative_intent == 'hold_press':
                    self.call_data['disposition'] = 'HP'
                elif self.intent_detector.negative_intent == 'callback':
                    self.call_data['disposition'] = 'CLBK'
                elif self.intent_detector.negative_intent == 'obscenity':
                    self.call_data['disposition'] = 'DNC'
                elif self.intent_detector.negative_intent == 'do_not_call':
                    self.call_data['disposition'] = 'DNC'
                self.state.set('disposition_set', True)
                self._log_and_collect('info', f"Disposition set to: {self.call_data['disposition']} based on intent: {self.intent_detector.negative_intent}")
                return 'graceful_exit'
        
        if 'conditions' not in step:
            return step.get('next', 'exit')
        
        for condition in step['conditions']:
            condition_type = condition.get('type')
            
            # Handle new condition types from script
            if condition_type in ['positive', 'negative', 'hostile']:
                # These are handled by intent detection above or Qwen
                continue
                
            elif condition_type == 'contains':
                keywords = condition.get('keywords', [])
                if any(kw.lower() in user_response.lower() for kw in keywords):
                    return condition.get('next')
                    
            elif condition_type == 'regex':
                pattern = condition.get('pattern', '')
                if re.search(pattern, user_response, re.IGNORECASE):
                    return condition.get('next')
        
        return None
    
    def _evaluate_conditions_with_qwen(self, step, user_response):
        """Evaluate conditions using Qwen context-aware detection (no fallback)"""
        from src.config import QWEN_TIMEOUT
        
        # First check for negative intents (keep existing keyword detection for safety)
        if self.intent_detector and self.intent_detector.negative_detected:
            self._log_and_collect('info', f"Negative intent detected: {self.intent_detector.negative_intent}")
            # Set appropriate disposition based on intent type
            if self.intent_detector.negative_intent == 'hold_press':
                self.call_data['disposition'] = 'HP'
            elif self.intent_detector.negative_intent == 'callback':
                self.call_data['disposition'] = 'CLBK'
            elif self.intent_detector.negative_intent == 'obscenity':
                self.call_data['disposition'] = 'DNC'
            elif self.intent_detector.negative_intent == 'do_not_call':
                self.call_data['disposition'] = 'DNC'
            elif self.intent_detector.negative_intent == 'not_interested':
                self.call_data['disposition'] = 'NI'
            self.state.set('disposition_set', True)
            self._log_and_collect('info', f"Disposition set to: {self.call_data['disposition']} based on intent: {self.intent_detector.negative_intent}")
            return "graceful_exit"
        
        # Check if Qwen is available and we have a question
        if not self.qwen_detector or not self.current_question_text:
            self._log_and_collect('warning', "Qwen not available or no question - cannot evaluate")
            return None
        
        # Use Qwen for intent detection
        try:
            start_time = time.time()
            
            # Get Qwen decision
            qwen_intent = self.qwen_detector.detect_intent(
                self.current_question_text,
                user_response,
                timeout=QWEN_TIMEOUT
            )
            
            latency = time.time() - start_time
            
            # Update metrics
            self.qwen_metrics['total_calls'] += 1
            if self.qwen_metrics['total_calls'] > 1:
                self.qwen_metrics['avg_latency'] = (
                    (self.qwen_metrics['avg_latency'] * (self.qwen_metrics['total_calls'] - 1) + latency) /
                    self.qwen_metrics['total_calls']
                )
            else:
                self.qwen_metrics['avg_latency'] = latency
            
            self._log_and_collect('info', f"Qwen intent: {qwen_intent} (latency: {latency:.3f}s)")
            
            if qwen_intent:
                # Map Qwen intent to next step
                next_step = self._map_qwen_intent_to_step(step, qwen_intent)
                self.qwen_metrics['successful_decisions'] += 1
                return next_step
            else:
                # Qwen returned None (error)
                self._log_and_collect('warning', "Qwen returned None - no decision available")
                self.qwen_metrics['failed_decisions'] += 1
                return None
                
        except Exception as e:
            self._log_and_collect('error', f"Qwen evaluation error: {e}")
            self.qwen_metrics['failed_decisions'] += 1
            return None

    def _map_qwen_intent_to_step(self, step, qwen_intent):
        """Map Qwen intent (positive/negative/clarifying) to next step"""
        conditions = step.get('conditions', [])
        
        if not conditions:
            return step.get('next', 'exit')
        
        # First priority: Check for intent-type conditions (new format)
        for condition in conditions:
            if condition.get('type') == 'intent' and condition.get('intent') == qwen_intent:
                next_step = condition.get('next')
                self._log_and_collect('debug', f"Matched intent condition: {qwen_intent} -> {next_step}")
                return next_step
        
        # Handle clarifying intent specifically
        if qwen_intent == "clarifying":
            # User is asking for more information
            # Look for a condition with intent='clarifying' in the step
            for condition in conditions:
                if condition.get('type') == 'intent' and condition.get('intent') == 'clarifying':
                    next_step = condition.get('next')
                    self._log_and_collect('debug', f"Qwen classified as clarifying - routing to: {next_step}")
                    return next_step
            
            # If no explicit clarifying condition, treat as neutral (trigger clarification)
            self._log_and_collect('debug', f"Qwen classified as clarifying but no route - triggering clarification")
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
            self._log_and_collect('debug', f"Qwen classified response as neutral - triggering clarification")
            return None
        
        # Default fallback
        return step.get('next', 'exit')

    def _find_condition_by_type(self, step, condition_type):
        """Find condition in step by type field"""
        conditions = step.get('conditions', [])
        
        for condition in conditions:
            if condition.get('type') == condition_type:
                return condition.get('next')
        
        return None
    
    def _handle_transfer_safe(self):
        """Handle call transfer with error handling"""
        self._log_and_collect('info', "Initiating transfer")
        
        if self.transfer_manager:
            try:
                if self.transfer_manager.transfer_qualified_call(self, self.phone_number):
                    self.state.set('call_transferred', True)
                    self._end_call_safe(CallOutcome.TRANSFERRED)
                else:
                    self._end_call_safe(CallOutcome.QUALIFIED)
            except Exception as e:
                self._log_and_collect('error', f"Transfer failed: {e}")
                self._end_call_safe(CallOutcome.QUALIFIED)
        else:
            self._end_call_safe(CallOutcome.QUALIFIED)
    
    def _handle_voicemail_detection(self):
        """Handle voicemail detection callback"""
        if self.state.get('vmd_triggered'):
            return
            
        self.state.set('vmd_triggered', True)
        self._log_and_collect('warning', "Voicemail detected")
        
        # Stop audio playback
        try:
            if self.audio_manager:
                self.audio_manager.cleanup_player()
        except:
            pass
        
        self.call_data['is_voicemail'] = True
        self.conversation_log.append("Bot: <Voicemail Detected>")
    
    def _handle_ringing_detection(self):
        """Handle ringing detection callback - immediately end call with RI disposition"""
        if self.state.get('ringing_triggered'):
            return
            
        self.state.set('ringing_triggered', True)
        self._log_and_collect('warning', "🔔 Ringing detected - ending call immediately")
        
        # Stop all audio operations
        try:
            if self.audio_manager:
                self.audio_manager.cleanup_player()
        except:
            pass
        
        # Set disposition and end call immediately
        self.call_data['disposition'] = 'RI'
        self.call_data['is_ringing'] = True
        self.state.set('disposition_set', True)
        self.conversation_log.append("Bot: <Ringing Detected>")
        
        # End call with RINGING outcome
        self._end_call_safe(CallOutcome.RINGING, perform_hangup=True)
    
    def _save_final_state_to_crm(self):
        """Save final state to CRM when call disconnects unexpectedly"""
        try:
            # Always update CRM if we have an ID, even if already updated
            if self.call_data.get('id') and self.crm_logger:
                # Collect any final continuous transcriptions before building transcript
                try:
                    final_transcriptions = self._collect_continuous_transcriptions(final_collection=True)
                    if final_transcriptions:
                        self._log_and_collect('info', f"Collected {len(final_transcriptions)} final continuous transcriptions for database")
                except Exception as e:
                    self._log_and_collect('warning', f"Error collecting final continuous transcriptions: {e}")

                # Check for incomplete SALE transfers (user disconnected before SIP REFER was sent)
                if (self.call_data.get('intended_disposition') == 'SALE' and
                    not self.sip_refer_sent):
                    self.call_data['disposition'] = 'SALE Failed'
                    self._log_and_collect('info', "Disposition: SALE Failed (SIP REFER never sent)")

                # Prepare final data
                final_data = copy.deepcopy(self.call_data)
                final_data['transcript'] = '\n'.join(self.conversation_log)
                final_data['end_time'] = time.time()
                final_data['duration'] = int(final_data['end_time'] - final_data['start_time'])
                
                # Ensure NP disposition for unexpected disconnects
                if final_data['disposition'] in ['INITIATED', 'UNKNOWN']:
                    final_data['disposition'] = 'NP'
                
                # Set drop step
                if self.current_step:
                    final_data['call_drop_step'] = self.current_step
                
                # Add collected call logs
                if hasattr(self, 'call_log_collector'):
                    final_data['call_logs'] = self.call_log_collector.get_logs()
                
                # Save recording BEFORE updating CRM (critical for early drops)
                if not final_data.get('filename'):
                    try:
                        # Check if we're in a PJSIP-registered thread
                        if hasattr(threading.current_thread(), '_pj_registered'):
                            filename, mimetype = self._save_recording_safe(final_data.get('id'))
                            if filename:
                                final_data['filename'] = filename
                                final_data['file_mime_type'] = mimetype
                                self._log_and_collect('info', f"Recording saved for early disconnect: {filename}")
                            else:
                                self._log_and_collect('warning', "No recording available for early disconnect")
                        else:
                            self._log_and_collect('warning', "Skipping recording save - not in PJSIP thread")
                    except Exception as e:
                        self._log_and_collect('error', f"Failed to save recording on disconnect: {e}")
                
                # Log to CRM
                self.crm_logger.log_call_end(final_data)
                self.state.set('crm_updated', True)
                self._log_and_collect('info', "Saved final state to CRM on disconnect")
                
                # Also report to ViciDial if outcome handler is available
                if self.outcome_handler:
                    try:
                        # Determine appropriate outcome for unexpected disconnect
                        outcome = CallOutcome.HANGUP_EARLY
                        self._log_and_collect('info', f"Reporting unexpected disconnect to ViciDial with outcome: {outcome.value}")
                        self.outcome_handler.process_call_outcome(
                            self.phone_number, outcome, final_data, None
                        )
                    except Exception as e:
                        self._log_and_collect('error', f"Failed to report to ViciDial on disconnect: {e}")
        except Exception as e:
            self._log_and_collect('error', f"Failed to save final state to CRM: {e}")
    
    def _end_call_safe(self, outcome, intent_data=None, perform_hangup=True):
        """End call with comprehensive cleanup"""
        # Check if already ended
        if not self.state.get('is_active'):
            return
        
        self._log_and_collect('info', f"Ending call: {outcome.value}")
        self.state.set('is_active', False)
        
        # Log Qwen metrics if used
        if self.qwen_metrics['total_calls'] > 0:
            self._log_and_collect('info', f"Qwen Metrics: {self.qwen_metrics}")
            
            # Get global metrics from detector
            if self.qwen_detector:
                try:
                    global_metrics = self.qwen_detector.get_metrics()
                    cache_hit_rate = global_metrics.get('cache_hit_rate', 0)
                    self._log_and_collect('info', f"Qwen Global: Cache hit rate={cache_hit_rate:.2%}")
                except Exception as e:
                    self._log_and_collect('warning', f"Error getting Qwen global metrics: {e}")
        
        try:
            # Check for incomplete SALE transfers (user disconnected before SIP REFER was sent)
            if (self.call_data.get('intended_disposition') == 'SALE' and
                not self.sip_refer_sent):
                self.call_data['disposition'] = 'SALE Failed'
                self._log_and_collect('info', "Disposition: SALE Failed (SIP REFER never sent)")
            
            # Set disposition based on outcome if not explicitly set
            if not self.state.get('disposition_set'):
                if outcome in [CallOutcome.HANGUP_EARLY, CallOutcome.FAILED]:
                    self.call_data['disposition'] = 'NP'
                elif outcome == CallOutcome.NOT_QUALIFIED:
                    # DNC is already set in the flow
                    if self.call_data['disposition'] not in ['DNC', 'NI']:
                        self.call_data['disposition'] = 'NP'
                elif outcome == CallOutcome.VOICEMAIL:
                    self.call_data['disposition'] = 'A'
                elif outcome == CallOutcome.RINGING:
                    self.call_data['disposition'] = 'RI'
            
            # Set drop step if needed
            if outcome in [CallOutcome.HANGUP_EARLY, CallOutcome.FAILED, 
                          CallOutcome.NOT_QUALIFIED, CallOutcome.VOICEMAIL, CallOutcome.RINGING]:
                if self.current_step:
                    self.call_data['call_drop_step'] = self.current_step
                    self._log_and_collect('warning', f"Dropped at: {self.current_step}")
            
            # Collect any final continuous transcriptions before building transcript
            try:
                final_transcriptions = self._collect_continuous_transcriptions(final_collection=True)
                if final_transcriptions:
                    self._log_and_collect('info', f"Collected {len(final_transcriptions)} final continuous transcriptions for end call")
            except Exception as e:
                self._log_and_collect('warning', f"Error collecting final continuous transcriptions: {e}")
            
            # Prepare final data
            final_data = copy.deepcopy(self.call_data)
            final_data['transcript'] = '\n'.join(self.conversation_log)
            final_data['end_time'] = time.time()
            final_data['duration'] = int(final_data['end_time'] - final_data['start_time'])
            
            # Add collected call logs
            if hasattr(self, 'call_log_collector'):
                final_data['call_logs'] = self.call_log_collector.get_logs()
            
            # CRITICAL: Perform hangup FIRST to ensure immediate disconnection
            # Recording save can happen after hangup without delaying disconnection
            if perform_hangup and not self.state.get('call_transferred'):
                self._log_and_collect('info', "🚀 Performing immediate hangup before recording save")
                self._perform_hangup_safe()
            
            # Save recording (now happens after hangup to not delay disconnection)
            filename, mimetype = self._save_recording_safe(final_data.get('id'))
            if filename:
                final_data['filename'] = filename
                final_data['file_mime_type'] = mimetype
            
            # Process outcome
            if self.outcome_handler:
                try:
                    self.outcome_handler.process_call_outcome(
                        self.phone_number, outcome, final_data, intent_data
                    )
                except Exception as e:
                    self._log_and_collect('error', f"Outcome processing failed: {e}")
            
            # Log to CRM (always update if not already updated)
            if self.crm_logger and not self.state.get('crm_updated'):
                try:
                    self.crm_logger.log_call_end(final_data)
                    self.state.set('crm_updated', True)
                except Exception as e:
                    self._log_and_collect('error', f"CRM logging failed: {e}")
                
        except Exception as e:
            self._log_and_collect('error', f"Error ending call: {e}")
    
    def _perform_hangup_safe(self):
        """Perform hangup with error handling"""
        try:
            # Check if call is still active
            ci = self.getInfo()
            if ci.state >= pj.PJSIP_INV_STATE_DISCONNECTED:
                return
                
            # Try transfer-based hangup first
            if self.transfer_manager:
                if self.transfer_manager.hangup_call_via_did(self):
                    self.state.set('call_transferred', True)
                    # CRITICAL: Also hang up our side immediately for instant disconnection
                    try:
                        self._log_and_collect('info', "🔌 Immediately hanging up local side after SIP REFER")
                        self.hangup(pj.CallOpParam(True))
                    except Exception as hangup_e:
                        # If local hangup fails, it's likely already disconnected
                        self._log_and_collect('info', f"Local hangup after REFER: {hangup_e} (likely already disconnected)")
                    return
            
            # Fallback to direct hangup
            self.hangup(pj.CallOpParam(True))
            
        except pj.Error as e:
            # Use numeric error codes
            if e.status in [PJSIP_ERROR_NOT_FOUND, PJSIP_ERROR_INVALID]:
                self._log_and_collect('info', f"Hangup error (likely already disconnected): {e.reason}")
            else:
                self.logger.error(f"Hangup error: {e}")
        except Exception as e:
            self.logger.error(f"Hangup error: {e}")
    
    def _save_recording_safe(self, call_id):
        """Save call recording with error handling - prioritizes stereo recording"""
        if not call_id:
            return None, None
        
        # Try stereo recording first (preferred)
        if hasattr(self, 'stereo_call_recorder') and self.stereo_call_recorder and self.stereo_call_recorder.is_recording:
            try:
                # CRITICAL: Ensure continuous listener recording is stopped first
                # This finalizes the WAV file so stereo recorder can read it properly
                if hasattr(self, 'continuous_listener') and self.continuous_listener:
                    self._log_and_collect('info', "🛑 Stopping continuous listener to finalize recording file")
                    
                    # Since hangup already happened, we can afford to wait for proper stop
                    # This is critical for recording file integrity
                    try:
                        # Flush any remaining transcriptions before stopping
                        try:
                            flushed = self.continuous_listener.flush_pending_transcription("call_recording_cleanup")
                            if flushed:
                                self._collect_continuous_transcriptions(since_time=time.time() - 10.0)
                                self._log_and_collect('info', f"Flushed {len(flushed)} final transcriptions before stop")
                        except Exception as flush_e:
                            self._log_and_collect('warning', f"Error flushing transcriptions on recording cleanup: {flush_e}")
                        
                        # Check thread safety before stopping continuous listener
                        if hasattr(threading.current_thread(), '_pj_registered'):
                            self.continuous_listener.stop_detection()
                            self._log_and_collect('info', "✅ Continuous listener stopped successfully")
                        else:
                            self._log_and_collect('warning', "⚠️ Skipping continuous listener stop - not in PJSIP thread")
                    except Exception as stop_e:
                        self._log_and_collect('error', f"Error stopping continuous listener: {stop_e}")
                    
                    # Brief delay to ensure WAV file is fully written to disk
                    import time
                    time.sleep(0.1)
                
                filepath = os.path.join(SUITECRM_UPLOAD_DIR, call_id)
                os.makedirs(SUITECRM_UPLOAD_DIR, exist_ok=True)
                
                self._log_and_collect('info', f"💿 Saving stereo recording to SuiteCRM: {filepath}")
                success = self.stereo_call_recorder.save_stereo_recording(filepath)
                
                if success:
                    self._log_and_collect('info', f"✅ Stereo recording saved to SuiteCRM: {filepath}")
                    return f"{call_id}", 'audio/wav'
                else:
                    self._log_and_collect('warning', "⚠️ Stereo recording failed, trying fallback")
                    
            except Exception as e:
                self._log_and_collect('error', f"Failed to save stereo recording: {e}")
        
        # Fallback to mono recording (legacy/backup)
        if self.all_audio_chunks:
            try:
                filepath = os.path.join(SUITECRM_UPLOAD_DIR, call_id)
                os.makedirs(SUITECRM_UPLOAD_DIR, exist_ok=True)
                
                audio_data = b''.join(self.all_audio_chunks)
                
                import wave
                with wave.open(filepath, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(8000)
                    wf.writeframes(audio_data)
                
                self._log_and_collect('info', f"📻 Mono recording saved as fallback: {filepath}")
                return f"{call_id}.wav", 'audio/wav'
                
            except Exception as e:
                self._log_and_collect('error', f"Failed to save mono recording: {e}")
                return None, None
        
        self._log_and_collect('warning', "No recording available to save")
        return None, None
    
    def _safe_cleanup(self, resource, method_name):
        """Safely cleanup a resource"""
        if not resource:
            return
            
        try:
            method = getattr(resource, method_name, None)
            if method and callable(method):
                method()
        except Exception as e:
            self._log_and_collect('error', f"Cleanup error for {method_name}: {e}")
    
    def cleanup(self):
        """Comprehensive cleanup of all resources"""
        if self.state.get('cleanup_done'):
            return
            
        self.state.set('cleanup_done', True)
        self._log_and_collect('info', "Starting cleanup")
        try:
            if hasattr(self, 'account') and hasattr(self, 'agent_config'):
                self.account.busy_agents.discard(self.agent_config.agent_id)
        except:
            pass

        # Set emergency cleanup flag
        self.state.set('emergency_cleanup', True)
        
        # Stop continuous listener first (non-PJSIP resource)
        # (may already be stopped in _save_recording_safe for stereo recording)
        if hasattr(self, 'continuous_listener') and self.continuous_listener:
            try:
                # Check if it's still active before stopping
                if hasattr(self.continuous_listener, 'is_active') and self.continuous_listener.is_active:
                    # Flush any remaining transcriptions before stopping
                    try:
                        flushed = self.continuous_listener.flush_pending_transcription("main_cleanup")
                        if flushed:
                            self._collect_continuous_transcriptions(since_time=time.time() - 10.0)
                            self._log_and_collect('info', f"Flushed {len(flushed)} final transcriptions during cleanup")
                    except Exception as flush_e:
                        self._log_and_collect('warning', f"Error flushing transcriptions during cleanup: {flush_e}")
                    
                    self._log_and_collect('info', "Stopping continuous listener")
                    self.continuous_listener.stop_detection()
                else:
                    self._log_and_collect('info', "Continuous listener already stopped")
                self.continuous_listener = None
            except Exception as e:
                self.logger.error(f"Error stopping continuous listener: {e}")
                self.continuous_listener = None
        
        # Note: Stereo call recording is now saved in _save_final_state_to_crm() via _save_recording_safe()
        # This ensures it's properly integrated with the database
        
        # Ensure final disposition is set
        if self.call_data['disposition'] in ['INITIATED', 'UNKNOWN']:
            self.call_data['disposition'] = 'NP'
            self._log_and_collect('warning', "Setting final disposition to NP during cleanup")
        
        # Save final state to CRM if needed
        self._save_final_state_to_crm()
        
        # Stop conversation thread FIRST before any PJSIP operations
        if self.conversation_thread and self.conversation_thread.is_alive():
            if threading.current_thread() != self.conversation_thread:
                self.conversation_thread.join(timeout=2)
        
        # CRITICAL: Only cleanup PJSIP resources if we're in a registered thread
        # This prevents the assertion failure
        try:
            # Check if current thread is registered with PJSIP
            if hasattr(threading.current_thread(), '_pj_registered'):
                # Safe to cleanup PJSIP resources
                self.resource_manager.cleanup_all()
            else:
                # Not in PJSIP thread - cleanup non-PJSIP resources only
                self._cleanup_non_pjsip_resources()
        except Exception as e:
            self._log_and_collect('error', f"Cleanup error: {e}")
        
        # Force garbage collection
        gc.collect()
        
        self._log_and_collect('info', "Cleanup complete")
    
    def _cleanup_non_pjsip_resources(self):
        """Cleanup only non-PJSIP resources (safe from any thread)"""
        # Stop continuous listener (non-PJSIP resource)
        if hasattr(self, 'continuous_listener') and self.continuous_listener:
            try:
                # Flush any remaining transcriptions before stopping
                try:
                    flushed = self.continuous_listener.flush_pending_transcription("non_pjsip_cleanup")
                    if flushed:
                        self._collect_continuous_transcriptions(since_time=time.time() - 10.0)
                        self._log_and_collect('info', f"Flushed {len(flushed)} final transcriptions in non-PJSIP cleanup")
                except Exception as flush_e:
                    self._log_and_collect('warning', f"Error flushing transcriptions in non-PJSIP cleanup: {flush_e}")
                
                self.continuous_listener.stop_detection()
                self.continuous_listener = None
            except Exception as e:
                self.logger.error(f"Error stopping continuous listener: {e}")
        
        # Clean file handler
        if hasattr(self, 'file_handler') and self.file_handler:
            try:
                self.logger.logger.removeHandler(self.file_handler)
                self.file_handler.close()
            except:
                pass
        
        # Clear data
        self.all_audio_chunks.clear()
        self.conversation_log.clear()
        
        # Shutdown executor
        if hasattr(self, 'operation_executor'):
            try:
                self.operation_executor.shutdown(wait=False)
            except:
                pass
    
    def cleanup_safe(self):
        """Public cleanup method with error handling"""
        try:
            self.cleanup()
        except Exception as e:
            self._log_and_collect('error', f"Cleanup failed: {e}")
    
    def force_cleanup(self):
        """Force immediate cleanup"""
        self.state.update_multiple({
            'is_active': False,
            'emergency_cleanup': True
        })
        self.cleanup_safe()