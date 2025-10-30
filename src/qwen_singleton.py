#!/usr/bin/env python3
"""
Qwen Model Singleton for Production SIP Bot

Manages a single Qwen instance across all calls for memory efficiency
Includes health monitoring and graceful failure handling
"""

import threading
import time
import logging
from typing import Optional, Dict, Any

class QwenModelSingleton:
    """Thread-safe singleton for Qwen intent detector"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        if QwenModelSingleton._instance is not None:
            raise RuntimeError("QwenModelSingleton is a singleton class")
        
        self._detector = None
        self._detector_lock = threading.Lock()
        self._last_health_check = 0
        self._health_status = {
            'status': 'uninitialized',
            'last_check': 0,
            'consecutive_failures': 0,
            'model_loaded': False,
            'last_error': None
        }
        
    @classmethod
    def get_instance(cls):
        """Get or create singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def get_detector(self, logger=None) -> Optional['OptimizedQwenIntentDetector']:
        """
        Get the Qwen detector instance, creating it if necessary
        Thread-safe lazy initialization
        """
        if logger is None:
            logger = logging.getLogger(__name__)
            
        with self._detector_lock:
            if self._detector is None:
                try:
                    logger.info("Initializing Qwen detector singleton...")
                    
                    # Import here to avoid circular imports
                    from src.qwen_intent_detector import OptimizedQwenIntentDetector
                    from src.config import QWEN_CACHE_SIZE, QWEN_USE_QUANTIZATION, QWEN_MODEL_NAME
                    
                    # Create detector with configuration
                    self._detector = OptimizedQwenIntentDetector(
                        cache_size=QWEN_CACHE_SIZE,
                        use_quantization=QWEN_USE_QUANTIZATION,
                        model_name=QWEN_MODEL_NAME,
                        logger=logger
                    )
                    
                    # Update health status
                    self._health_status.update({
                        'status': 'healthy',
                        'last_check': time.time(),
                        'consecutive_failures': 0,
                        'model_loaded': True,
                        'last_error': None
                    })
                    
                    logger.info("✅ Qwen detector singleton initialized successfully")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize Qwen detector: {e}")
                    self._health_status.update({
                        'status': 'failed',
                        'last_check': time.time(),
                        'consecutive_failures': self._health_status['consecutive_failures'] + 1,
                        'model_loaded': False,
                        'last_error': str(e)
                    })
                    self._detector = None
                    return None
        
        return self._detector
    
    def health_check(self, logger=None) -> Dict[str, Any]:
        """
        Perform health check on the detector
        Returns health status dictionary
        """
        if logger is None:
            logger = logging.getLogger(__name__)
            
        current_time = time.time()
        
        # Check if we need to run health check
        from src.config import QWEN_HEALTH_CHECK_INTERVAL
        if (current_time - self._last_health_check) < QWEN_HEALTH_CHECK_INTERVAL:
            return self._health_status
        
        self._last_health_check = current_time
        
        with self._detector_lock:
            if self._detector is None:
                self._health_status.update({
                    'status': 'uninitialized',
                    'last_check': current_time,
                    'model_loaded': False
                })
                return self._health_status
            
            try:
                # Test inference with a simple question
                test_question = "Do you have Medicare Part A and Part B?"
                test_answer = "Yes, I do."
                
                start_time = time.time()
                result = self._detector.detect_intent(
                    test_question, 
                    test_answer, 
                    timeout=2.0
                )
                response_time = time.time() - start_time
                
                if result in ["positive", "negative"]:
                    # Successful health check
                    self._health_status.update({
                        'status': 'healthy',
                        'last_check': current_time,
                        'consecutive_failures': 0,
                        'model_loaded': True,
                        'last_error': None,
                        'test_response_time': response_time,
                        'test_result': result
                    })
                else:
                    # Test failed
                    self._health_status.update({
                        'status': 'test_failed',
                        'last_check': current_time,
                        'consecutive_failures': self._health_status['consecutive_failures'] + 1,
                        'test_error': f"Invalid result: {result}"
                    })
                    
            except Exception as e:
                logger.warning(f"Qwen health check failed: {e}")
                self._health_status.update({
                    'status': 'unhealthy',
                    'last_check': current_time,
                    'consecutive_failures': self._health_status['consecutive_failures'] + 1,
                    'last_error': str(e)
                })
        
        return self._health_status
    
    def get_metrics(self) -> Optional[Dict[str, Any]]:
        """Get detector performance metrics"""
        with self._detector_lock:
            if self._detector is None:
                return None
            return self._detector.get_metrics()
    
    def clear_cache(self, logger=None):
        """Clear the detector cache"""
        if logger is None:
            logger = logging.getLogger(__name__)
            
        with self._detector_lock:
            if self._detector:
                self._detector.clear_cache()
                logger.info("Qwen cache cleared")
            else:
                logger.warning("No Qwen detector to clear cache from")
    
    def get_cache_size(self) -> int:
        """Get current cache size"""
        with self._detector_lock:
            if self._detector:
                return self._detector.get_cache_size()
            return 0
    
    def shutdown(self, logger=None):
        """Gracefully shutdown the detector"""
        if logger is None:
            logger = logging.getLogger(__name__)
            
        with self._detector_lock:
            if self._detector:
                logger.info("Shutting down Qwen detector...")
                try:
                    self._detector.cleanup()
                except Exception as e:
                    logger.error(f"Error during Qwen cleanup: {e}")
                finally:
                    self._detector = None
                    self._health_status.update({
                        'status': 'shutdown',
                        'last_check': time.time(),
                        'model_loaded': False
                    })
                    logger.info("✅ Qwen detector shutdown completed")
            else:
                logger.info("Qwen detector already shutdown")
    
    def restart(self, logger=None):
        """Restart the detector (shutdown and reinitialize)"""
        if logger is None:
            logger = logging.getLogger(__name__)
            
        logger.info("Restarting Qwen detector...")
        
        # Shutdown current detector
        self.shutdown(logger)
        
        # Wait a moment
        time.sleep(1)
        
        # Reinitialize
        detector = self.get_detector(logger)
        if detector:
            logger.info("✅ Qwen detector restarted successfully")
            return True
        else:
            logger.error("❌ Failed to restart Qwen detector")
            return False
    
    def is_healthy(self) -> bool:
        """Quick health status check"""
        return (
            self._health_status.get('status') == 'healthy' and
            self._health_status.get('model_loaded', False) and
            self._health_status.get('consecutive_failures', 0) < 3
        )
    
    def get_status_summary(self) -> str:
        """Get a brief status summary string"""
        status = self._health_status.get('status', 'unknown')
        failures = self._health_status.get('consecutive_failures', 0)
        loaded = self._health_status.get('model_loaded', False)
        
        if status == 'healthy' and loaded:
            return "✅ Healthy"
        elif status == 'uninitialized':
            return "⏳ Not initialized"
        elif status == 'failed':
            return f"❌ Failed (errors: {failures})"
        elif status == 'unhealthy':
            return f"⚠️ Unhealthy (errors: {failures})"
        elif status == 'shutdown':
            return "🔴 Shutdown"
        else:
            return f"❓ Unknown ({status})"

# Global singleton instance
qwen_singleton = QwenModelSingleton.get_instance()