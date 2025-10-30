#!/usr/bin/env python3
import os
# -----------------------------
# System & Path Config
# -----------------------------
# Parakeet Model Configuration
MODEL_PATH = "models/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo"
USE_LOCAL_MODEL = True
USE_GPU = True  # Enable GPU usage for Parakeet

# -----------------------------
# RNNT Model Configuration (NEW)
# -----------------------------
# Enable Parakeet RNNT 1.1b with native confidence support
USE_RNNT_MODEL = True  # Set to False to use TDT instead

# RNNT model download/storage path
RNNT_MODEL_PATH = "models/parakeet-rnnt-1.1b"

# RNNT Confidence Configuration
# Minimum confidence threshold for accepting transcriptions (0.0 to 1.0)
RNNT_CONFIDENCE_THRESHOLD = 0.5

# Confidence thresholds for different quality levels
RNNT_LOW_CONFIDENCE_THRESHOLD = 0.3   # Below this: ignore completely  
RNNT_HIGH_CONFIDENCE_THRESHOLD = 0.7  # Above this: high quality

# Confidence aggregation method for multi-word responses
# Options: "prod" (product), "mean" (average), "min" (minimum), "max" (maximum)
RNNT_CONFIDENCE_AGGREGATION = "prod"

# Confidence extraction method
# Options: "max_prob" (maximum probability), "entropy" (entropy-based)
RNNT_CONFIDENCE_METHOD = "max_prob"

# Enable detailed confidence logging for debugging
RNNT_CONFIDENCE_DEBUG = False

# Fallback behavior when RNNT fails
RNNT_FALLBACK_TO_TDT = True  # Use TDT as fallback if RNNT unavailable
# -----------------------------
# Core Audio Settings
# -----------------------------
# SIP uses 8kHz, but Parakeet requires 16kHz. Upsampling is handled internally.
SIP_SAMPLE_RATE = 8000
SIP_CHANNELS = 1
# -----------------------------
# Silero VAD Configuration
# -----------------------------
# ML-based voice activity detection - PRIMARY speech detection method
# Energy-based VAD has been removed entirely to eliminate volume bias
#
# Uses the energy_threshold value from database (per campaign) as Silero speech probability threshold
# This is NOT an energy/volume threshold - it's ML confidence that audio contains speech
#
# Threshold interpretation (configured in database energy_threshold field):
# 0.0 - Accept almost all audio (Silero will filter only extreme noise)
# 0.25 - Very sensitive (catches whispers and quiet speech, minimal false positives)
# 0.3 - Sensitive (good for quiet speakers)
# 0.4 - Balanced for most use cases
# 0.5 - Conservative (recommended default)
# 0.6 - Strict (filters more aggressively)
# 0.7 - Very strict (only clear speech)
#
# Key benefits over energy-based VAD:
# - Detects quiet/whispered speech (low volume but clear speech patterns)
# - Filters loud noise (high volume but no speech patterns)
# - No volume bias - purely acoustic pattern recognition
#
# Note: Each call gets its own Silero VAD instances (AudioBuffer + ContinuousListener)
# Memory cost: ~10MB per concurrent call
# -----------------------------
# Intent Detection Threshold
# -----------------------------
# Minimum confidence score (0.0 to 1.0) required to trigger a negative intent.
MIN_INTENT_CONFIDENCE = 0.65
# -----------------------------
# SIP Server Settings
# -----------------------------
SIP_PORT = 5060
SIP_BIND_ADDRESS = "0.0.0.0"
MAX_CALL_DURATION = 300  # 5 minutes max call duration
# -----------------------------
# Audio Timing Settings
# -----------------------------
# Fine-tuning delays to prevent audio issues like clipping or repeating.
AUDIO_STOP_DELAY = 0.5      # Delay in seconds after audio finishes to ensure clean stop (increased to prevent echo)
SIP_PLAYER_CLEANUP_DELAY = 0.3  # Extra delay for SIP player cleanup

# -----------------------------
# Qwen Intent Detection Configuration
# -----------------------------
# Master switch for Qwen context-aware intent detection
USE_QWEN_INTENT = True

# Model configuration - can be changed to "Qwen/Qwen2.5-3B" for smaller memory footprint
QWEN_MODEL_NAME = "JungZoona/T3Q-qwen2.5-14b-v1.0-e3"

# Cache configuration for Q&A pairs (LRU cache) - increased for 100 concurrent
QWEN_CACHE_SIZE = 25000

# Use 4-bit quantization for lower memory usage (recommended for production)
QWEN_USE_QUANTIZATION = True

# Maximum inference time in seconds before timeout
QWEN_TIMEOUT = 1.0

# GPU lock timeout configuration (for handling concurrent calls)
QWEN_GPU_LOCK_TIMEOUT = 2.0  # Max seconds to wait for GPU lock
QWEN_INFERENCE_TIMEOUT = 1.5  # Max seconds for actual inference
QWEN_TOTAL_TIMEOUT = 3.5  # Total timeout (lock wait + inference)

# Log comparison between Qwen and keyword decisions for analysis
QWEN_LOG_COMPARISON = False

# Questions enabled for Qwen (empty list = all questions)
# Start with specific questions for gradual rollout
QWEN_ENABLED_QUESTIONS = []

# Fallback configuration
QWEN_FALLBACK_TO_KEYWORDS = False  # No fallback - Qwen only
QWEN_MAX_RETRIES = 1  # Maximum retries for failed inference

# Performance monitoring
QWEN_WARMUP_ON_STARTUP = True  # Perform warmup inference on startup
QWEN_HEALTH_CHECK_INTERVAL = 300  # Health check every 5 minutes

# -----------------------------
# Ringing Detection Configuration
# -----------------------------
# Enable continuous ringing detection throughout call duration
RINGING_DETECTION_ENABLED = True

# Minimum number of complete ringback cycles to confirm ringing
RINGING_MIN_CYCLES = 2

# Confidence threshold for ringing pattern matching (0.0 to 1.0)
RINGING_CONFIDENCE_THRESHOLD = 0.75

# Audio chunk size for ringing analysis (samples)
RINGING_CHUNK_SIZE = 1024

# Relative threshold: target frequencies must be this many times stronger than background
RINGING_RELATIVE_THRESHOLD = 7.0

# Regional ringback frequency configurations (Hz)
RINGING_FREQUENCIES = {
    'us_canada': [440.0, 480.0],  # Default US/Canada pattern
    'uk': [400.0, 450.0],
    'europe': [425.0],
    'australia': [400.0, 425.0]
}

# Active region for ringing detection
RINGING_ACTIVE_REGION = 'us_canada'

# -----------------------------
# Continuous Listening Configuration
# -----------------------------
# Enable continuous listening throughout the entire call
CONTINUOUS_LISTENING_ENABLED = True

# Audio buffer size in seconds (how much audio to keep in memory)
CONTINUOUS_BUFFER_SIZE = 300  # 5 minutes of continuous audio

# Intent detection intervals
INTENT_DETECTION_INTERVAL = 0.5  # Check for intents every 500ms
INTENT_WINDOW_SIZE = 2.0  # Analyze 2-second windows of audio for intents

# Minimum speech duration to process for intent detection (seconds)
MIN_SPEECH_DURATION_FOR_INTENT = 0.3

# Response extraction timing
RESPONSE_TIMEOUT_DEFAULT = 10  # Default timeout for user responses (seconds)
RESPONSE_SILENCE_THRESHOLD = 1.5  # Seconds of silence before considering response complete

# Audio processing settings for continuous listening
CONTINUOUS_TARGET_SAMPLE_RATE = 16000  # Upsample SIP audio to 16kHz for better processing
CONTINUOUS_CHUNK_INTERVAL = 0.05  # Process audio in 50ms chunks

# -----------------------------
# Voicemail Detection (VMD) Configuration
# -----------------------------
# Enable ML-based voicemail detection
VMD_ENABLED = True

# Detection duration in seconds (how long to record before ML analysis)
VMD_DETECTION_DURATION = 7.0

# ML model confidence threshold (0.0-1.0)
# Higher = more conservative (fewer false positives, might miss some voicemails)
# Lower = more aggressive (detect more voicemails, might have false positives)
VMD_CONFIDENCE_THRESHOLD = 0.60

# Model path for wav2vec voicemail detector
VMD_MODEL_PATH = "models/voicemail_detector"

# Minimum audio length in bytes for analysis
VMD_MIN_AUDIO_LENGTH = 1600  # ~100ms at 8kHz, 16-bit

# End call immediately when voicemail detected (vs just logging)
VMD_END_CALL_ON_DETECTION = True

# -----------------------------
# Continuous Background Noise Configuration
# -----------------------------
# Continuous background noise is automatically enabled when configured in the database.
# The system uses the existing e_campaigns table fields:
#
# Database Configuration (e_campaigns table):
#   - background_noise_volume: Controls volume for both continuous player and mixing (0.0 to 1.0)
#     * 0.0 = Disabled (no continuous background noise)
#     * 0.1-0.3 = Subtle ambient noise (recommended for most campaigns)
#     * 0.4-0.6 = Medium ambient noise
#     * 0.7-1.0 = Strong ambient noise
#   - noise_location: Path to background noise audio file (linked via e_noise table)
#
# Implementation:
#   - ContinuousBackgroundNoisePlayer: Dedicated PJSIP player for looping background noise
#   - Runs continuously throughout entire call (during bot speech AND listening periods)
#   - Independent of main audio playback system
#   - Graceful failure: If player fails, call continues normally
#   - Automatic cleanup on call end
#
# Benefits:
#   - Natural call ambience with no "dead air" during listening periods
#   - Professional audio quality throughout call
#   - Layered with existing bot speech mixing for consistent sound
#
# Memory Impact:
#   - ~2-3 MB per call (looped audio file + player instance)
#   - Temp files automatically cleaned up after call ends
#
# To disable continuous background noise for a campaign:
#   - Set background_noise_volume = 0 in e_campaigns table for that campaign
# -----------------------------