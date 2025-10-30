#!/usr/bin/env python3
"""
Production-Grade SuiteCRM Integrated SIP Bot Server
- Bulletproof multi-threading with comprehensive resource management
- Advanced deadlock prevention and recovery
- Crash-resistant with automatic recovery
- Handles hundreds of thousands of calls without failure
- Using NVIDIA Parakeet TDT for speech recognition
"""

import pjsua2 as pj
import time
import sys
import threading
import random
import logging
import gc
import weakref
import queue
import psutil
import resource  # For system resource limits
import subprocess  # For network kernel optimizations
import traceback
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
import os
import re
import signal

# --- Configuration ---
LOG_DIR = "/var/log/sip-bot"
HEARTBEAT_FILE = os.path.join(LOG_DIR, "sip_bot.heartbeat")
MAX_THREADS = 3000  # Maximum concurrent threads (optimized for 20 cores + high concurrency)
THREAD_POOL_SIZE = 1000  # Thread pool for background tasks (supporting 1000+ concurrent calls)
CLEANUP_INTERVAL = 30  # Seconds between cleanup cycles
DEADLOCK_TIMEOUT = 120  # Maximum call duration before forced cleanup
MEMORY_CHECK_INTERVAL = 60  # Memory monitoring interval
MAX_MEMORY_PERCENT = 88  # Maximum memory usage before cleanup (optimized for 62GB RAM)
PJSIP_HANDLE_TIMEOUT = 10  # Timeout for PJSIP event handling

# --- Global State ---
ep: pj.Endpoint = None
SHUTTING_DOWN = threading.Event()
EMERGENCY_SHUTDOWN = threading.Event()
thread_pool = None
resource_monitor = None
performance_monitor = None # Added for performance dashboard

# --- Thread-Safe Resource Registry ---
class ResourceRegistry:
    """Central registry for all active resources with automatic cleanup"""
    def __init__(self):
        self.lock = threading.RLock()
        self.calls = weakref.WeakValueDictionary()
        self.threads = weakref.WeakSet()
        self.media_resources = weakref.WeakSet()
        self.cleanup_queue = queue.Queue()
        self.stats = {
            'total_calls': 0,
            'active_calls': 0,
            'failed_calls': 0,
            'cleaned_calls': 0,
            'memory_cleanups': 0
        }
        
    def register_call(self, call_id, call_instance):
        with self.lock:
            self.calls[call_id] = call_instance
            self.stats['total_calls'] += 1
            self.stats['active_calls'] = len(self.calls)
            
    def unregister_call(self, call_id):
        with self.lock:
            if call_id in self.calls:
                del self.calls[call_id]
            self.stats['active_calls'] = len(self.calls)
            self.stats['cleaned_calls'] += 1
            
    def register_thread(self, thread):
        with self.lock:
            self.threads.add(thread)
            
    def get_stats(self):
        with self.lock:
            return self.stats.copy()
            
    def emergency_cleanup(self):
        """Emergency cleanup of all resources"""
        with self.lock:
            log.critical(f"EMERGENCY CLEANUP: Cleaning {len(self.calls)} calls")
            for call_id, call in list(self.calls.items()):
                try:
                    call.force_cleanup()
                except:
                    pass
            self.calls.clear()
            gc.collect()

resource_registry = ResourceRegistry()

# --- Thread-Safe PJSIP Operations ---
class PJSIPThreadManager:
    """Manages PJSIP thread registration and operations"""
    def __init__(self):
        self.registered_threads = set()
        self.lock = threading.Lock()
        
    def register_thread(self, thread_name=None):
        """Thread-safe PJSIP thread registration"""
        thread_id = threading.get_ident()
        with self.lock:
            if thread_id in self.registered_threads:
                return True
                
            try:
                if not thread_name:
                    thread_name = f"thread_{thread_id}"
                pj.Endpoint.instance().libRegisterThread(thread_name)
                self.registered_threads.add(thread_id)
                threading.current_thread()._pj_registered = True
                return True
            except Exception as e:
                log.warning(f"Failed to register thread {thread_name}: {e}")
                return False
                
    @contextmanager
    def pjsip_thread_context(self, thread_name=None):
        """Context manager for PJSIP thread operations"""
        self.register_thread(thread_name)
        try:
            yield
        finally:
            pass  # Thread registration persists

pjsip_thread_manager = PJSIPThreadManager()

# --- Enhanced Parakeet Model Singleton (TDT + RNNT Support) ---
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
                    
                    # Set device
                    if USE_GPU and torch.cuda.is_available():
                        device = 'cuda'
                        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                    else:
                        device = 'cpu'
                        log.info("Using CPU for inference")
                    
                    # Load RNNT model if enabled
                    if USE_RNNT_MODEL:
                        try:
                            from src.parakeet_rnnt import create_rnnt_model
                            
                            log.info("Loading global Parakeet RNNT 1.1b model (singleton)...")
                            
                            # Use provided threshold or fallback to config default
                            threshold = confidence_threshold if confidence_threshold is not None else RNNT_CONFIDENCE_THRESHOLD
                            
                            self._model = create_rnnt_model(
                                model_path=RNNT_MODEL_PATH,
                                confidence_threshold=threshold,
                                device=device,
                                logger=logger or log
                            )
                            
                            # Ensure model is ready
                            if self._model.ensure_model_loaded():
                                self._model_type = "RNNT"
                                log.info("✅ Global Parakeet RNNT 1.1b model loaded successfully")
                            else:
                                raise Exception("RNNT model failed to load")
                                
                        except Exception as rnnt_error:
                            log.error(f"Failed to load RNNT model: {rnnt_error}")
                            
                            if RNNT_FALLBACK_TO_TDT:
                                log.info("Falling back to TDT model...")
                                self._model = None  # Reset for TDT loading
                            else:
                                raise rnnt_error
                    
                    # Load TDT model if RNNT not enabled or failed with fallback
                    if self._model is None:
                        import nemo.collections.asr as nemo_asr
                        
                        log.info("Loading global Parakeet TDT model (singleton)...")
                        
                        if USE_LOCAL_MODEL:
                            log.info(f"Loading local Parakeet model from {MODEL_PATH}")
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
                                
                                log.info("CUDA graph optimization disabled for multi-threading")
                            except Exception as e:
                                log.warning(f"Could not disable CUDA graph optimization: {e}")
                        
                        log.info("✅ Global Parakeet TDT model loaded successfully")
                    
                    self._load_event.set()
                    
                except Exception as e:
                    self._load_error = str(e)
                    log.error(f"Failed to load Parakeet model: {e}")
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
        return self._model is not None

parakeet_singleton = ParakeetModelSingleton()

# --- Enhanced Qwen Model Singleton ---
from src.config import USE_QWEN_INTENT

qwen_singleton = None
if USE_QWEN_INTENT:
    try:
        from src.qwen_singleton import QwenModelSingleton
        qwen_singleton = QwenModelSingleton.get_instance()
    except ImportError as e:
        qwen_singleton = None

# --- Enhanced Logging ---
class ProductionLogger:
    """Production-grade logger with rotation and performance monitoring"""
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(agent_id)s] [%(name)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        console_handler.addFilter(self.AgentLogFilter())
        
        if not self.logger.handlers:
            self.logger.addHandler(console_handler)
            
        # File handler with rotation
        try:
            from logging.handlers import RotatingFileHandler
            os.makedirs(LOG_DIR, exist_ok=True)
            file_handler = RotatingFileHandler(
                os.path.join(LOG_DIR, 'sip_bot_server.log'),
                maxBytes=100*1024*1024,  # 100MB
                backupCount=10
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(self.AgentLogFilter())
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to setup file logging: {e}")
    
    class AgentLogFilter(logging.Filter):
        def filter(self, record):
            if not hasattr(record, 'agent_id'):
                record.agent_id = 'SIPServer'
            return True

production_logger = ProductionLogger()
log = production_logger.logger

# --- Performance Metrics Logger (Dashboard) ---
class PerformanceMetricsLogger:
    """Separate logger for real-time performance metrics"""
    def __init__(self):
        self.logger = logging.getLogger('performance_metrics')
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # CRITICAL: Prevent propagation to parent loggers
        
        # Remove any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handler for performance metrics
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            file_handler = logging.FileHandler(
                os.path.join(LOG_DIR, 'performance.log'),
                mode='w'  # Overwrite on restart
            )
            formatter = logging.Formatter('%(message)s')  # Just the message, no timestamp
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        except Exception as e:
            print(f"Failed to setup performance logging: {e}")
    
    def log_metrics(self, message):
        """Log performance metrics"""
        self.logger.info(message)

performance_logger = PerformanceMetricsLogger()

# --- Real-Time Performance Monitor (Dashboard) ---
class RealTimePerformanceMonitor:
    """Monitors and outputs real-time bot performance metrics"""
    def __init__(self, account_manager):
        self.account_manager = account_manager
        self.running = False
        self.monitor_thread = None
        self.start_time = time.time()
        self.rejection_reasons = {
            'no_agents': 0,
            'untrusted_ip': 0,
            'no_campaign': 0,
            'server_shutdown': 0,
            'creation_failed': 0
        }
        self.rejection_lock = threading.Lock()
        
    def start(self):
        """Start the performance monitor"""
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="performance_monitor",
            daemon=True
        )
        self.monitor_thread.start()
        performance_logger.log_metrics("=== SIP Bot Performance Monitor Started ===")
        
    def stop(self):
        """Stop the performance monitor"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
    
    def log_rejection(self, reason):
        """Log a call rejection reason"""
        with self.rejection_lock:
            if reason in self.rejection_reasons:
                self.rejection_reasons[reason] += 1
    
    def _monitor_loop(self):
        """Main monitoring loop - updates every 0.5 seconds"""
        while self.running and not SHUTTING_DOWN.is_set():
            try:
                self._generate_metrics()
                time.sleep(0.5)
            except Exception as e:
                log.error(f"Performance monitor error: {e}")
                time.sleep(1.0)
    
    def _generate_metrics(self):
        """Generate comprehensive performance metrics"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Get account manager stats
        with self.account_manager.lock:
            active_calls = len(self.account_manager.active_calls)
            busy_agents = len(self.account_manager.busy_agents)
            total_agents = len(self.account_manager.agent_configs)
            available_agents = total_agents - busy_agents
            stats = self.account_manager.performance_stats.copy()
        
        # Get system stats
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Get process stats
            current_process = psutil.Process()
            threads = current_process.num_threads()
            fds = current_process.num_fds() if hasattr(current_process, 'num_fds') else 0
            memory_mb = current_process.memory_info().rss / (1024 * 1024)
        except Exception:
            memory = type('obj', (object,), {'percent': 0, 'available': 0})()
            cpu_percent = 0
            threads = 0
            fds = 0
            memory_mb = 0
        
        # Get rejection stats
        with self.rejection_lock:
            rejection_stats = self.rejection_reasons.copy()
        
        # Get resource registry stats
        registry_stats = resource_registry.get_stats()
        
        # Format the output
        metrics = []
        metrics.append("=" * 80)
        metrics.append(f"SIP BOT REAL-TIME PERFORMANCE - {datetime.now().strftime('%H:%M:%S')}")
        metrics.append("=" * 80)
        
        # Call Statistics
        metrics.append("\n📊 CALL STATISTICS:")
        metrics.append(f"   Active Calls:     {active_calls:>6}")
        metrics.append(f"   Total Processed:  {stats['accepted_calls']:>6}")
        metrics.append(f"   Successful:       {stats['successful_calls']:>6}")
        metrics.append(f"   Failed:           {stats['failed_calls']:>6}")
        metrics.append(f"   Rejected:         {stats['rejected_calls']:>6}")
        
        # Agent Statistics  
        metrics.append(f"\n🤖 AGENT STATISTICS (Total: {total_agents}):")
        metrics.append(f"   Available:        {available_agents:>6}")
        metrics.append(f"   Busy:             {busy_agents:>6}")
        metrics.append(f"   Utilization:      {(busy_agents/total_agents*100):>5.1f}%")
        
        # Rejection Breakdown
        total_rejections = sum(rejection_stats.values())
        if total_rejections > 0:
            metrics.append(f"\n❌ REJECTION REASONS:")
            metrics.append(f"   No Agents:        {rejection_stats['no_agents']:>6}")
            metrics.append(f"   Untrusted IP:     {rejection_stats['untrusted_ip']:>6}")
            metrics.append(f"   No Campaign:      {rejection_stats['no_campaign']:>6}")
            metrics.append(f"   Shutdown:         {rejection_stats['server_shutdown']:>6}")
            metrics.append(f"   Creation Failed:  {rejection_stats['creation_failed']:>6}")
        
        # System Resources
        metrics.append(f"\n💻 SYSTEM RESOURCES:")
        metrics.append(f"   CPU Usage:        {cpu_percent:>5.1f}%")
        metrics.append(f"   Memory Usage:     {memory.percent:>5.1f}%")
        metrics.append(f"   Bot Memory:       {memory_mb:>5.0f} MB")
        metrics.append(f"   Threads:          {threads:>6}")
        metrics.append(f"   File Descriptors: {fds:>6}")
        
        # Registry Stats
        metrics.append(f"\n📈 RESOURCE REGISTRY:")
        metrics.append(f"   Tracked Calls:    {registry_stats['active_calls']:>6}")
        metrics.append(f"   Cleaned Calls:    {registry_stats['cleaned_calls']:>6}")
        metrics.append(f"   Memory Cleanups:  {registry_stats['memory_cleanups']:>6}")
        
        # Performance Rates
        if uptime > 0:
            calls_per_minute = (stats['accepted_calls'] / uptime) * 60
            success_rate = (stats['successful_calls'] / max(stats['accepted_calls'], 1)) * 100
            metrics.append(f"\n⚡ PERFORMANCE:")
            metrics.append(f"   Calls/Minute:     {calls_per_minute:>5.1f}")
            metrics.append(f"   Success Rate:     {success_rate:>5.1f}%")
            metrics.append(f"   Uptime:           {uptime/60:>5.1f} min")
        
        # Status
        if active_calls >= available_agents and available_agents > 0:
            metrics.append(f"\n⚠️  STATUS: HIGH LOAD - {available_agents} agents remaining")
        elif available_agents == 0:
            metrics.append(f"\n🚨 STATUS: ALL AGENTS BUSY - New calls will be REJECTED")
        else:
            metrics.append(f"\n✅ STATUS: HEALTHY - {available_agents} agents available")
        
        metrics.append("")  # Empty line at end
        
        # Write all metrics at once
        performance_logger.log_metrics("\n".join(metrics))

# --- System Resource Monitor ---
class SystemResourceMonitor:
    """Monitors system resources and triggers cleanup when needed"""
    def __init__(self):
        self.running = False
        self.monitor_thread = None
        self.last_memory_cleanup = time.time()
        
    def start(self):
        self.running = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="resource_monitor",
            daemon=True
        )
        self.monitor_thread.start()
        
    def stop(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
    def _monitor_loop(self):
        """Monitor system resources"""
        pjsip_thread_manager.register_thread("resource_monitor")
        
        while self.running and not SHUTTING_DOWN.is_set():
            try:
                # Check memory usage
                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent(interval=1)
                
                stats = resource_registry.get_stats()
                log.info(f"System: Memory={memory_percent:.1f}%, CPU={cpu_percent:.1f}%, "
                        f"Calls={stats['active_calls']}/{stats['total_calls']}")
                
                # Trigger cleanup if memory is high
                if memory_percent > MAX_MEMORY_PERCENT:
                    current_time = time.time()
                    if current_time - self.last_memory_cleanup > 60:
                        log.warning(f"High memory usage ({memory_percent}%), triggering cleanup")
                        self._trigger_memory_cleanup()
                        self.last_memory_cleanup = current_time
                        resource_registry.stats['memory_cleanups'] += 1
                
                # Update heartbeat
                self._update_heartbeat()
                
                time.sleep(MEMORY_CHECK_INTERVAL)
                
            except Exception as e:
                log.error(f"Error in resource monitor: {e}")
                time.sleep(MEMORY_CHECK_INTERVAL)
    
    def _trigger_memory_cleanup(self):
        """Trigger garbage collection and cleanup"""
        gc.collect()
        
        # Force cleanup of old calls
        with resource_registry.lock:
            current_time = time.time()
            for call_id, call in list(resource_registry.calls.items()):
                try:
                    if hasattr(call, 'call_start_time'):
                        age = current_time - call.call_start_time
                        if age > DEADLOCK_TIMEOUT:
                            log.warning(f"Force cleaning old call {call_id} (age={age:.0f}s)")
                            call.force_cleanup()
                except:
                    pass
    
    def _update_heartbeat(self):
        """Update heartbeat file"""
        try:
            with open(HEARTBEAT_FILE, "w") as f:
                stats = resource_registry.get_stats()
                f.write(f"{datetime.now().isoformat()}|{stats['active_calls']}|{stats['total_calls']}")
        except:
            pass

# --- Advanced Deadlock Prevention ---
class DeadlockPrevention:
    """Advanced deadlock prevention with timeout-based operations"""
    def __init__(self):
        self.operation_timeouts = {}
        self.lock = threading.Lock()
        
    @contextmanager
    def timeout_operation(self, operation_id, timeout=30):
        """Execute operation with timeout"""
        start_time = time.time()
        with self.lock:
            self.operation_timeouts[operation_id] = start_time
            
        try:
            yield
        finally:
            with self.lock:
                if operation_id in self.operation_timeouts:
                    del self.operation_timeouts[operation_id]
                    
            elapsed = time.time() - start_time
            if elapsed > timeout:
                log.warning(f"Operation {operation_id} took {elapsed:.1f}s (timeout={timeout}s)")

deadlock_prevention = DeadlockPrevention()

# --- Main Account Manager ---
class SIPAccountManager(pj.Account):
    """Production-grade account manager with comprehensive error handling"""
    
    def __init__(self, agent_configs, global_energy_threshold=0.045, global_rnnt_confidence_threshold=0.5):
        super().__init__()
        self.agent_configs = {cfg.agent_id: cfg for cfg in agent_configs}
        self.global_energy_threshold = global_energy_threshold
        self.global_rnnt_confidence_threshold = global_rnnt_confidence_threshold
        self.phone_extractor = None  # Lazy load
        self.active_calls = {}
        self.busy_agents = set()
        self.lock = threading.RLock()
        self.call_counter = 0
        self.last_cleanup = time.time()
        
        # Performance tracking
        self.performance_stats = {
            'accepted_calls': 0,
            'rejected_calls': 0,
            'failed_calls': 0,
            'successful_calls': 0
        }
        
        # Parse comma-separated server IPs and create a set of all valid IPs
        self.valid_server_ips = set()
        for cfg in agent_configs:
            if cfg.server_ip:
                # Split by comma and strip whitespace, add all IPs to the set
                ips = [ip.strip() for ip in cfg.server_ip.split(',') if ip.strip()]
                self.valid_server_ips.update(ips)
        
        log.info(f"✅ Loaded {len(agent_configs)} agents from {len(self.valid_server_ips)} server IPs: {sorted(self.valid_server_ips)}")
        
        # Start background cleanup thread
        self.cleanup_thread = threading.Thread(
            target=self._periodic_cleanup,
            name="periodic_cleanup",
            daemon=True
        )
        self.cleanup_running = True
        self.cleanup_thread.start()
        
        log.info("Production SIP Account Manager initialized")
    
    def _lazy_load_phone_extractor(self):
        """Lazy load phone extractor to avoid import issues"""
        if self.phone_extractor is None:
            from src.sip_phone_extractor import SIPPhoneExtractor
            self.phone_extractor = SIPPhoneExtractor()
        return self.phone_extractor
    
    def onIncomingCall(self, prm: pj.OnIncomingCallParam):
        """Handle incoming call with comprehensive error handling"""
        call_id = None
        temp_call = None
        
        try:
            # Check if shutting down
            if SHUTTING_DOWN.is_set() or EMERGENCY_SHUTDOWN.is_set():
                log.warning("Rejecting call - server shutting down")
                prm.code = pj.PJSIP_SC_SERVICE_UNAVAILABLE
                if performance_monitor:
                    performance_monitor.log_rejection('server_shutdown')
                return
            
            # Extract source IP
            source_ip = prm.rdata.srcAddress.split(':')[0]
            
            # Validate source
            if source_ip not in self.valid_server_ips:
                log.warning(f"Rejected call from untrusted IP: {source_ip}")
                prm.code = pj.PJSIP_SC_FORBIDDEN
                self.performance_stats['rejected_calls'] += 1
                if performance_monitor:
                    performance_monitor.log_rejection('untrusted_ip')
                return
            
            log.info(f"📞 Accepted call from {source_ip}")
            
            # Extract headers
            sip_message = prm.rdata.wholeMsg
            campaign_id = self._extract_header(sip_message, 'X-VICIdial-Campaign-Id')
            lead_id = self._extract_header(sip_message, 'X-VICIdial-Lead-Id')
            list_id = self._extract_header(sip_message, 'X-VICIdial-List-Id')

            # Campaign ID is now optional - log if present
            if campaign_id:
                log.info(f"Campaign ID: {campaign_id}")
            else:
                log.info("No campaign ID - accepting call anyway")
            
            # Select agent with timeout
            selected_agent = self._select_agent_with_timeout(campaign_id, source_ip, timeout=2.0)
            
            if not selected_agent:
                campaign_info = f" (campaign: {campaign_id})" if campaign_id else ""
                log.warning(f"No available agents for source IP {source_ip}{campaign_info}")
                prm.code = pj.PJSIP_SC_BUSY_HERE
                self.performance_stats['rejected_calls'] += 1
                if performance_monitor:
                    performance_monitor.log_rejection('no_agents')
                return
            
            # Create bot instance with timeout protection
            with deadlock_prevention.timeout_operation(f"create_bot_{prm.callId}", timeout=5):
                from suitecrm_bot_instance import SuiteCRMBotInstance
                
                bot_instance = SuiteCRMBotInstance(
                    account=self,
                    call_id=prm.callId,
                    agent_config=selected_agent,
                    source_ip=source_ip,
                    parakeet_singleton=parakeet_singleton,
                    qwen_singleton=qwen_singleton,
                    resource_registry=resource_registry,
                    global_energy_threshold=self.global_energy_threshold,
                    global_rnnt_confidence_threshold=self.global_rnnt_confidence_threshold
                )
                
                # Extract phone number with state
                phone_extractor = self._lazy_load_phone_extractor()
                extraction_result = phone_extractor.extract_phone_with_state(prm)
                phone_number = extraction_result.phone_number
                caller_state = extraction_result.caller_state
                
                # Set call details with state and ViciDial campaign_id
                bot_instance.set_call_details(phone_number, prm, lead_id, list_id, campaign_id, caller_state)
                
                # Register call
                call_id = bot_instance.getId()
                with self.lock:
                    self.active_calls[call_id] = bot_instance
                    self.call_counter += 1
                    self.performance_stats['accepted_calls'] += 1
                
                resource_registry.register_call(call_id, bot_instance)
                
                state_info = f" from {caller_state}" if caller_state else ""
                log.info(f"Created call {call_id} for {phone_number}{state_info} (Lead: {lead_id})")
                
                # Answer call
                call_prm = pj.CallOpParam(True)
                call_prm.statusCode = pj.PJSIP_SC_OK
                bot_instance.answer(call_prm)
                
        except Exception as e:
            log.error(f"Failed to handle incoming call: {e}", exc_info=True)
            self.performance_stats['failed_calls'] += 1
            if performance_monitor:
                performance_monitor.log_rejection('creation_failed')
            
            # Clean up on failure
            if selected_agent:
                with self.lock:
                    self.busy_agents.discard(selected_agent.agent_id)
            
            if call_id:
                self.remove_call_safe(call_id)
            
            # Reject the call
            try:
                if not temp_call:
                    temp_call = pj.Call(self, prm.callId)
                temp_call.hangup(pj.CallOpParam())
            except:
                pass
    
    def _extract_header(self, sip_message, header):
        """Extract SIP header value"""
        try:
            match = re.search(rf'^{re.escape(header)}:\s*(.*?)\r?$', 
                            sip_message, re.IGNORECASE | re.MULTILINE)
            return match.group(1).strip() if match else None
        except:
            return None
    
    def _is_ip_in_server_list(self, source_ip, server_ip_list):
        """Check if source IP is in comma-separated server IP list"""
        if not server_ip_list:
            return False
        
        # Split by comma, strip whitespace, and check if source_ip is in the list
        valid_ips = [ip.strip() for ip in server_ip_list.split(',') if ip.strip()]
        return source_ip in valid_ips
    
    def _select_agent_with_timeout(self, campaign_id, source_ip, timeout=2.0):
        """Select available agent with timeout protection (campaign_id no longer used for filtering)"""
        deadline = time.time() + timeout

        while time.time() < deadline:
            with self.lock:
                # Find matching agents (by server IP only, campaign filtering removed)
                matching = {
                    aid: cfg for aid, cfg in self.agent_configs.items()
                    if self._is_ip_in_server_list(source_ip, cfg.server_ip)
                }

                if not matching:
                    return None

                # Find available agents
                available = list(set(matching.keys()) - self.busy_agents)

                if available:
                    selected_id = random.choice(available)
                    selected_agent = self.agent_configs[selected_id]
                    self.busy_agents.add(selected_id)
                    campaign_info = f" (campaign: {campaign_id})" if campaign_id else ""
                    log.info(f"Selected agent {selected_id[:8]} (busy: {len(self.busy_agents)}){campaign_info}")
                    return selected_agent

            time.sleep(0.1)

        return None
    
    def remove_call(self, bot_instance):
        """Remove call with proper cleanup"""
        if not bot_instance:
            return
            
        call_id = None
        try:
            call_id = bot_instance.getId()
        except:
            return
            
        self.remove_call_safe(call_id, bot_instance)
    
    def remove_call_safe(self, call_id, bot_instance=None):
        """Thread-safe call removal"""
        if not call_id:
            return
        if not self.cleanup_running:
            return
        with self.lock:
            # Get bot instance if not provided
            if not bot_instance:
                bot_instance = self.active_calls.get(call_id)
            
            if not bot_instance:
                return
            
            # Get agent ID
            agent_id = None
            try:
                agent_id = bot_instance.agent_config.agent_id
            except:
                pass
            
            # Remove from active calls FIRST
            if call_id in self.active_calls:
                del self.active_calls[call_id]
                log.info(f"Removed call {call_id} (active: {len(self.active_calls)})")
            
            # Release agent
            if agent_id:
                self.busy_agents.discard(agent_id)
                log.info(f"Released agent {agent_id[:8]} (busy: {len(self.busy_agents)})")
            
            # Update stats
            self.performance_stats['successful_calls'] += 1
        
        # CRITICAL FIX: Register the current thread with PJSIP before cleanup
        # This ensures cleanup happens in a PJSIP-registered thread
        def cleanup_in_background():
            try:
                # Register this thread with PJSIP FIRST
                pjsip_thread_manager.register_thread(f"cleanup_{call_id}")
                
                # Give time for PJSIP to settle
                time.sleep(0.1)
                
                # Now safe to cleanup
                bot_instance.cleanup_safe()
            except Exception as e:
                log.error(f"Background cleanup error: {e}")
        
        # Use thread pool for cleanup if available
        if thread_pool:
            thread_pool.submit(cleanup_in_background)
        else:
            # Fallback to direct thread
            cleanup_thread = threading.Thread(
                target=cleanup_in_background,
                daemon=True
            )
            cleanup_thread.start()
        
        # Unregister from registry
        resource_registry.unregister_call(call_id)
    
    def _periodic_cleanup(self):
        """Periodic cleanup of stuck resources"""
        pjsip_thread_manager.register_thread("cleanup")
        
        while self.cleanup_running and not SHUTTING_DOWN.is_set():
            try:
                time.sleep(CLEANUP_INTERVAL)
                
                current_time = time.time()
                
                # Log stats
                with self.lock:
                    log.info(f"Cleanup: Active={len(self.active_calls)}, "
                            f"Busy={len(self.busy_agents)}, "
                            f"Stats={self.performance_stats}")
                
                # Clean stuck calls
                self._cleanup_stuck_calls(current_time)
                
                # Clean orphaned agents
                self._cleanup_orphaned_agents()
                
                # Trigger GC if needed
                if current_time - self.last_cleanup > 300:  # Every 5 minutes
                    gc.collect()
                    self.last_cleanup = current_time
                    log.info("Performed garbage collection")
                
            except Exception as e:
                log.error(f"Cleanup error: {e}", exc_info=True)
    
    def _cleanup_stuck_calls(self, current_time):
        """Clean up stuck calls"""
        stuck_calls = []
        force_remove_calls = []
        
        with self.lock:
            for call_id, bot in list(self.active_calls.items()):
                try:
                    if hasattr(bot, 'call_start_time'):
                        age = current_time - bot.call_start_time
                        if age > DEADLOCK_TIMEOUT * 1.25:  # 4.5 minutes = force remove (270s)
                            force_remove_calls.append((call_id, bot, age))
                        elif age > DEADLOCK_TIMEOUT * 1:  # 4 minutes = normal cleanup (240s)
                            stuck_calls.append((call_id, bot, age))
                except:
                    stuck_calls.append((call_id, bot, 0))
        
        # Force remove extremely stuck calls directly
        for call_id, bot, age in force_remove_calls:
            log.warning(f"Force removing stuck call {call_id} (age={age:.0f}s)")
            with self.lock:
                if call_id in self.active_calls:
                    del self.active_calls[call_id]
                # Release agent if possible
                try:
                    agent_id = bot.agent_config.agent_id
                    self.busy_agents.discard(agent_id)
                except:
                    pass
        
        # Normal cleanup for moderately stuck calls
        for call_id, bot, age in stuck_calls:
            log.warning(f"Cleaning stuck call {call_id} (age={age:.0f}s)")
            self.remove_call_safe(call_id, bot)
    
    def _cleanup_orphaned_agents(self):
        """Clean up orphaned busy agents"""
        with self.lock:
            active_agents = set()
            for bot in self.active_calls.values():
                try:
                    active_agents.add(bot.agent_config.agent_id)
                except:
                    pass
            
            orphaned = self.busy_agents - active_agents
            if orphaned:
                log.warning(f"Releasing {len(orphaned)} orphaned agents")
                self.busy_agents -= orphaned
            if len(self.active_calls) == 0 and len(self.busy_agents) > 0:
                log.warning(f"No active calls but {len(self.busy_agents)} agents busy - clearing all")
                self.busy_agents.clear()
    
    def shutdown(self):
        """Graceful shutdown"""
        log.info("Account manager shutting down...")
        
        self.cleanup_running = False
        
        # Stop cleanup thread
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        # Hangup all calls with proper error handling
        with self.lock:
            calls_to_hangup = list(self.active_calls.values())
            # Clear references BEFORE trying to hangup to prevent race conditions
            self.active_calls.clear()
            self.busy_agents.clear()
        
        if calls_to_hangup:
            log.info(f"Hanging up {len(calls_to_hangup)} active calls...")
            for call in calls_to_hangup:
                try:
                    # Check if call is still valid before attempting hangup
                    call_info = call.getInfo()
                    if call_info.state < pj.PJSIP_INV_STATE_DISCONNECTED:
                        call.hangup(pj.CallOpParam(True))
                except pj.Error as e:
                    # Ignore errors for already terminated calls
                    if e.status != 171140:  # PJSIP_ESESSIONTERMINATED
                        log.debug(f"Error during shutdown hangup: {e.reason}")
                except Exception as e:
                    log.debug(f"Error during shutdown: {e}")
        
        log.info("Account manager shutdown complete")

# --- Signal Handlers ---
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    global SHUTTING_DOWN, EMERGENCY_SHUTDOWN
    
    if signum == signal.SIGUSR1:
        log.info("Restart signal received")
        SHUTTING_DOWN.set()
        time.sleep(2)
        os.execv(sys.executable, [sys.executable] + sys.argv)
    elif signum == signal.SIGUSR2:
        log.critical("EMERGENCY SHUTDOWN")
        EMERGENCY_SHUTDOWN.set()
        resource_registry.emergency_cleanup()
        sys.exit(1)
    else:
        log.warning(f"Shutdown signal {signum} received")
        SHUTTING_DOWN.set()

# --- Main Function ---
def execute_whitelist_commands(agent_configs):
    """
    Execute whitelist commands from campaign configurations at server startup
    Only executes unique commands (de-duplicates by campaign)
    """
    if not agent_configs:
        return

    # Collect unique commands per campaign to avoid duplicate execution
    campaign_commands = {}
    for config in agent_configs:
        campaign_id = config.vicidial_campaign_id
        command = config.white_list_command

        # Only add if command exists and campaign not already processed
        if command and command.strip() and campaign_id not in campaign_commands:
            campaign_commands[campaign_id] = command.strip()

    if not campaign_commands:
        log.info("No whitelist commands to execute")
        return

    log.info(f"🔧 Executing {len(campaign_commands)} whitelist command(s) from campaigns...")

    for campaign_id, command in campaign_commands.items():
        try:
            log.info(f"📋 Campaign {campaign_id}: Executing whitelist command: {command}")

            # Execute command with timeout (30 seconds max)
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                log.info(f"✅ Campaign {campaign_id}: Whitelist command executed successfully")
                if result.stdout:
                    log.debug(f"   Output: {result.stdout.strip()}")
            else:
                log.warning(f"⚠️ Campaign {campaign_id}: Whitelist command exited with code {result.returncode}")
                if result.stderr:
                    log.warning(f"   Error: {result.stderr.strip()}")

        except subprocess.TimeoutExpired:
            log.error(f"❌ Campaign {campaign_id}: Whitelist command timed out (>30s)")
        except Exception as e:
            log.error(f"❌ Campaign {campaign_id}: Failed to execute whitelist command: {e}")

    log.info("✅ Whitelist command execution completed")

def configure_system_resources():
    """Configure system resource limits for maximum concurrent calls"""
    try:
        # File descriptor limits (critical for SIP connections)
        current_nofile_soft, current_nofile_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        log.info(f"Current file descriptor limits: soft={current_nofile_soft}, hard={current_nofile_hard}")

        # Set file descriptors to maximum
        resource.setrlimit(resource.RLIMIT_NOFILE, (65535, 65535))
        log.info("✅ File descriptor limit increased to 65535")

        # Stack size for threads (prevent stack overflow with many threads)
        try:
            resource.setrlimit(resource.RLIMIT_STACK, (16*1024*1024, resource.RLIM_INFINITY))
            log.info("✅ Stack size limit increased to 16MB")
        except ValueError:
            log.warning("Could not set unlimited stack size, using system default")

        # Process/thread limits
        current_nproc_soft, current_nproc_hard = resource.getrlimit(resource.RLIMIT_NPROC)
        log.info(f"Current process limits: soft={current_nproc_soft}, hard={current_nproc_hard}")

        # Network kernel optimizations (if running as root/sudo)
        try:
            if os.geteuid() == 0:  # Running as root
                log.info("Applying network kernel optimizations...")
                subprocess.run(['sysctl', '-w', 'net.core.somaxconn=8192'], check=True)
                subprocess.run(['sysctl', '-w', 'net.core.netdev_max_backlog=65536'], check=True)
                subprocess.run(['sysctl', '-w', 'net.ipv4.tcp_max_syn_backlog=65536'], check=True)
                log.info("✅ Network kernel parameters optimized")
            else:
                log.info("Not running as root - skipping kernel parameter optimization")
        except Exception as e:
            log.warning(f"Could not apply kernel optimizations: {e}")

        log.info("✅ System resource configuration completed")

    except Exception as e:
        log.error(f"Error configuring system resources: {e}")
        log.warning("Continuing with default system limits...")

def main():
    global ep, thread_pool, resource_monitor, performance_monitor
    
    # CRITICAL: Configure system resources FIRST for maximum concurrency
    configure_system_resources()
    
    # Setup signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGUSR1, signal_handler)  # Restart
    signal.signal(signal.SIGUSR2, signal_handler)  # Emergency
    
    account_manager = None
    
    try:
        # Import dependencies
        from suitecrm_bot_instance import SuiteCRMBotInstance
        from src.sip_phone_extractor import SIPPhoneExtractor
        from src.suitecrm_integration import fetch_active_agent_configs
        from src.server_id_service import server_id_service
        
        log.info("="*60)
        log.info("🚀 Starting Production SIP Bot Server")
        log.info("="*60)
        
        # Create directories
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # Initialize centralized server ID at startup
        log.info("🔗 Retrieving server ID from centralized API...")
        server_id = server_id_service.retrieve_server_id(max_retries=3)
        if server_id:
            log.info(f"✅ Server ID retrieved successfully: {server_id}")
        else:
            log.warning("⚠️ Server ID retrieval failed - calls will log without server ID")
        
        # Initialize thread pool
        thread_pool = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)
        
        # Start resource monitor
        resource_monitor = SystemResourceMonitor()
        resource_monitor.start()
        
        # Fetch agent configurations
        log.info("Loading agent configurations...")
        agent_configs = fetch_active_agent_configs()
        
        if not agent_configs:
            log.critical("No active agents found")
            return
        
        log.info(f"✅ Loaded {len(agent_configs)} agents")
        
        # Extract global thresholds from first agent config (campaign settings)
        global_energy_threshold = 0.045  # Default fallback
        global_rnnt_confidence_threshold = 0.5  # Default fallback
        
        if agent_configs:
            first_config = agent_configs[0]
            global_energy_threshold = first_config.energy_threshold
            global_rnnt_confidence_threshold = first_config.rnnt_confidence_threshold
            log.info(f"📊 Global thresholds from database: Energy={global_energy_threshold}, RNNT_Confidence={global_rnnt_confidence_threshold}")
        else:
            log.info(f"📊 Using default thresholds: Energy={global_energy_threshold}, RNNT_Confidence={global_rnnt_confidence_threshold}")

        # Execute whitelist commands from campaigns
        execute_whitelist_commands(agent_configs)

        # Pre-load HP Disposition Cache (all historical records)
        log.info("🔄 Initializing HP Disposition Cache...")
        try:
            from src.hp_disposition_cache import get_hp_cache

            hp_cache = get_hp_cache()
            hp_cache.start_preload(blocking=False, shutdown_event=SHUTTING_DOWN)

            log.info("✅ HP Disposition Cache loading in background")
        except ImportError as e:
            log.warning(f"⚠️ HP Disposition Cache not available: {e}")
        except Exception as e:
            log.error(f"❌ Failed to initialize HP cache: {e}")
            log.info("Continuing without HP cache - calls will not be filtered")

        # Pre-load Parakeet model with database threshold
        log.info("Pre-loading Parakeet TDT model...")
        parakeet_model = parakeet_singleton.get_model(log, confidence_threshold=global_rnnt_confidence_threshold)
        if not parakeet_model:
            log.warning("⚠️ Parakeet model not loaded - will retry on demand")
        
        # Pre-load Qwen model if enabled
        if USE_QWEN_INTENT and qwen_singleton:
            from src.config import QWEN_WARMUP_ON_STARTUP
            log.info("Pre-loading Qwen intent detector...")
            qwen_detector = qwen_singleton.get_detector(log)
            if not qwen_detector:
                log.warning("⚠️ Qwen detector not loaded - will use keyword fallback")
            else:
                log.info("✅ Qwen intent detector ready")
                if QWEN_WARMUP_ON_STARTUP:
                    try:
                        qwen_detector.warmup()
                    except Exception as e:
                        log.warning(f"Qwen warmup failed: {e}")
        elif USE_QWEN_INTENT:
            log.warning("⚠️ Qwen enabled but singleton not available - using keyword fallback")
        
        # Initialize PJSIP
        ep = pj.Endpoint()
        ep.libCreate()
        
        ep_cfg = pj.EpConfig()
        ep_cfg.logConfig.level = 3
        ep_cfg.logConfig.consoleLevel = 3
        
        # Dynamic limits optimized for MAXIMUM concurrent load
        max_calls = 1000  # Support up to 1000 concurrent calls
        max_media_ports = 10000  # 10 ports per call × 1000 calls
        
        ep_cfg.uaConfig.maxCalls = max_calls
        ep_cfg.medConfig.maxMediaPorts = max_media_ports
        ep_cfg.medConfig.hasIoqueue = True
        ep_cfg.medConfig.clockRate = 8000
        ep_cfg.medConfig.channelCount = 1
        # Configure audio conference bridge for MAXIMUM concurrent load
        ep_cfg.medConfig.maxAudioFrameSize = 2000
        ep_cfg.medConfig.ptime = 20
        ep_cfg.medConfig.quality = 4  # Balance between quality and performance
        # CRITICAL: Reduce audio latency for better responsiveness at high load
        ep_cfg.medConfig.sndRecLatency = 20  # Recording latency (ms)
        ep_cfg.medConfig.sndPlayLatency = 40  # Playback latency (ms)
        # CRITICAL: Set conference bridge slots for concurrent audio streams
        # Each call needs multiple slots (players + recorders + audio sources)
        conf_bridge_slots = 10000  # Support 1000 calls with multiple audio streams each
        ep_cfg.medConfig.maxMediaPorts = 10000  # Aligned with max_media_ports
        
        log.info(f"PJSIP limits: {max_calls} calls, {max_media_ports} media ports")
        
        ep.libInit(ep_cfg)
        ep.audDevManager().setNullDev()
        
        # Create transport
        transport_cfg = pj.TransportConfig()
        transport_cfg.port = 5060
        ep.transportCreate(pj.PJSIP_TRANSPORT_UDP, transport_cfg)
        
        ep.libStart()
        log.info("✅ PJSIP initialized")
        
        # Create account
        acc_cfg = pj.AccountConfig()
        acc_cfg.idUri = "sip:localhost"
        acc_cfg.regConfig.registerOnAdd = False
        
        account_manager = SIPAccountManager(agent_configs, global_energy_threshold, global_rnnt_confidence_threshold)
        account_manager.create(acc_cfg)
        
        # Start performance monitor
        performance_monitor = RealTimePerformanceMonitor(account_manager)
        performance_monitor.start()
        
        # Register main thread
        pjsip_thread_manager.register_thread("main")
        
        log.info("\n" + "="*60)
        log.info("🤖 PRODUCTION SIP BOT SERVER READY")
        log.info(f"📞 Port: 5060 | Agents: {len(agent_configs)}")
        log.info("🧠 Using NVIDIA Parakeet TDT for ASR")
        log.info("="*60 + "\n")
        
        # Main event loop
        while not SHUTTING_DOWN.is_set() and not EMERGENCY_SHUTDOWN.is_set():
            try:
                ep.libHandleEvents(PJSIP_HANDLE_TIMEOUT)
            except Exception as e:
                log.error(f"Error in event loop: {e}")
                time.sleep(0.1)
        
    except Exception as e:
        log.critical(f"Fatal error: {e}", exc_info=True)
        EMERGENCY_SHUTDOWN.set()
        
    finally:
        log.info("Shutting down...")
        
        # Stop monitors
        if performance_monitor:
            performance_monitor.stop()

        if resource_monitor:
            resource_monitor.stop()

        # Shutdown HP cache
        try:
            from src.hp_disposition_cache import get_hp_cache
            hp_cache = get_hp_cache()
            hp_cache.shutdown(timeout=5)
            log.info("✅ HP Disposition Cache shutdown complete")
        except:
            pass

        # Shutdown account manager
        if account_manager:
            account_manager.shutdown()
        
        # Shutdown thread pool
        if thread_pool:
            thread_pool.shutdown(wait=True)
        
        # Wait for all threads to complete
        log.info("Waiting for threads to complete...")
        time.sleep(3)
        
        # Destroy PJSIP
        if ep:
            try:
                # CRITICAL: Unregister all threads before destroying endpoint
                log.info("Destroying PJSIP endpoint...")
                # Give time for any pending PJSIP operations
                time.sleep(1)
                ep.libDestroy()
            except Exception as e:
                log.debug(f"Error destroying endpoint (expected during shutdown): {e}")
            ep = None
        
        log.info("✅ Shutdown complete")

if __name__ == "__main__":
    main()