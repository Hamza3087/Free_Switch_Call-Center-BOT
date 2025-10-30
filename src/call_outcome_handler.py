#!/usr/bin/env python3
"""
Call Outcome Handler
Manages call outcomes, dispositions, and ViciDial reporting
"""

import time
import threading
import queue
import uuid
from typing import Dict, Any, Optional
from enum import Enum
import logging

class CallOutcome(Enum):
    """Internal state representation of how a call ended."""
    QUALIFIED = "qualified"
    NOT_QUALIFIED = "not_qualified"
    NEGATIVE_INTENT = "negative_intent"
    VOICEMAIL = "voicemail"
    RINGING = "ringing"
    HANGUP_EARLY = "hangup_early"
    FAILED = "failed"
    TRANSFERRED = "transferred"
    HANGUP_SCRIPTED = "hangup_scripted"

class CallDisposition:
    """ViciDial disposition codes are now defined in the call flow scripts."""
    pass

class AsyncViciDialReporter:
    """
    Asynchronous ViciDial reporter that prevents HTTP API calls from blocking call cleanup
    Uses background worker threads and queuing to ensure call flow is never delayed by ViciDial latency
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, num_workers=3):
        self.num_workers = num_workers
        self.work_queue = queue.Queue(maxsize=500)  # Prevent memory exhaustion
        self.workers = []
        self.shutdown_event = threading.Event()
        self.stats = {
            'queued_reports': 0,
            'completed_reports': 0,
            'failed_reports': 0,
            'queue_full_errors': 0
        }
        self.stats_lock = threading.Lock()
        self._logger = logging.getLogger(__name__)
        
        # Fallback file logging when ViciDial unavailable
        self.fallback_log_dir = "/var/log/sip-bot/vicidial_fallback"
        import os
        os.makedirs(self.fallback_log_dir, exist_ok=True)
        
        self._start_workers()
    
    @classmethod
    def get_instance(cls):
        """Singleton pattern for global async ViciDial reporter"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _start_workers(self):
        """Start background worker threads"""
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"AsyncViciDialReporter-Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        self._logger.info(f"✅ Started {self.num_workers} async ViciDial workers")
    
    def _worker_loop(self):
        """Background worker thread loop"""
        while not self.shutdown_event.is_set():
            try:
                # Get work item with timeout
                work_item = self.work_queue.get(timeout=1.0)
                if work_item is None:  # Shutdown signal
                    break
                
                operation, args = work_item
                try:
                    if operation == 'report_disposition':
                        self._execute_disposition_report(*args)
                    
                    with self.stats_lock:
                        self.stats['completed_reports'] += 1
                        
                except Exception as e:
                    with self.stats_lock:
                        self.stats['failed_reports'] += 1
                    self._logger.error(f"ViciDial report failed: {e}", exc_info=True)
                    
                    # Fallback to file logging
                    self._fallback_log(operation, args, str(e))
                
                finally:
                    self.work_queue.task_done()
                    
            except queue.Empty:
                continue  # Timeout, check shutdown event
            except Exception as e:
                self._logger.error(f"ViciDial worker loop error: {e}", exc_info=True)
    
    def _execute_disposition_report(self, phone_number, disposition, call_data, vicidial_integration):
        """Execute ViciDial disposition report"""
        try:
            # Call the actual ViciDial API
            success = vicidial_integration.send_disposition_directly(
                phone_number, disposition, call_data
            )
            
            if success:
                self._logger.debug(f"Async: ViciDial disposition {disposition} reported for {phone_number}")
            else:
                raise Exception(f"ViciDial API returned failure for {phone_number}")
                
        except Exception as e:
            self._logger.error(f"ViciDial disposition report failed for {phone_number}: {e}")
            raise
    
    def _fallback_log(self, operation, args, error):
        """Write to fallback file when ViciDial unavailable"""
        try:
            from datetime import datetime
            fallback_file = f"{self.fallback_log_dir}/failed_vicidial_ops_{datetime.now().strftime('%Y%m%d')}.log"
            with open(fallback_file, 'a') as f:
                timestamp = datetime.now().isoformat()
                f.write(f"{timestamp} | {operation} | ERROR: {error} | ARGS: {args}\n")
        except Exception as e:
            self._logger.error(f"Failed to write ViciDial fallback log: {e}")
    
    def queue_disposition_report(self, phone_number, disposition, call_data, vicidial_integration) -> str:
        """
        Queue ViciDial disposition report for background processing
        Returns report ID immediately without waiting for HTTP call
        """
        report_id = str(uuid.uuid4())
        
        try:
            work_item = ('report_disposition', (phone_number, disposition, call_data, vicidial_integration))
            self.work_queue.put(work_item, block=False)
            
            with self.stats_lock:
                self.stats['queued_reports'] += 1
            
            self._logger.debug(f"ViciDial disposition {disposition} queued for {phone_number} (ID: {report_id})")
            return report_id
            
        except queue.Full:
            with self.stats_lock:
                self.stats['queue_full_errors'] += 1
            
            self._logger.warning("Async ViciDial queue full - falling back to file logging")
            self._fallback_log('report_disposition', (phone_number, disposition, call_data), "Queue full")
            return report_id
    
    def get_stats(self):
        """Get async ViciDial reporter statistics"""
        with self.stats_lock:
            return self.stats.copy()
    
    def shutdown(self, timeout=10):
        """Gracefully shutdown async ViciDial reporter"""
        self._logger.info("Shutting down async ViciDial reporter...")
        
        # Signal shutdown and add None items to wake up workers
        self.shutdown_event.set()
        for _ in range(self.num_workers):
            try:
                self.work_queue.put(None, timeout=1)
            except queue.Full:
                pass
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout/self.num_workers)
        
        # Wait for remaining queue items
        try:
            self.work_queue.join()
        except:
            pass
        
        stats = self.get_stats()
        self._logger.info(f"Async ViciDial reporter shutdown complete. Final stats: {stats}")

class CallOutcomeHandler:
    """Handles call outcomes and ViciDial reporting"""
    
    def __init__(self, vicidial_integration=None, logger=None):
        self.vicidial_integration = vicidial_integration
        self.logger = logger or logging.getLogger(__name__)
        
        # Get async ViciDial reporter instance for non-blocking API calls
        self.async_vicidial_reporter = AsyncViciDialReporter.get_instance()
        
        self.logger.info("[Call Outcome] Initialized call outcome handler with async ViciDial reporting")
    
    def process_call_outcome(self, phone_number: str, outcome: CallOutcome, 
                           call_data: Dict[str, Any], intent_data: Dict[str, Any] = None) -> bool:
        """
        Process the final call outcome.
        The disposition is expected to be set in call_data from the script.
        This method sets final timestamps, generates comments, and reports to ViciDial.
        """
        try:
            self.logger.info(f"[Call Outcome] Processing outcome: {outcome.value} for {phone_number}")
            
            # Set final timestamps and internal outcome
            call_data['outcome'] = outcome.value
            call_data['end_time'] = time.time()
            call_data['duration'] = int(call_data['end_time'] - call_data['start_time'])
            
            # The disposition should already be in call_data, set by the bot instance from the script.
            # We provide a fallback here for unexpected edge cases.
            disposition = call_data.get('disposition')
            if not disposition or disposition == 'UNKNOWN':
                self.logger.warning(f"Disposition not set by script for outcome '{outcome.value}'. Using fallback.")
                fallback_map = {
                    CallOutcome.VOICEMAIL: "A",
                    CallOutcome.RINGING: "RI",
                    CallOutcome.NEGATIVE_INTENT: "NI",
                    CallOutcome.HANGUP_EARLY: "NP",
                    CallOutcome.FAILED: "DAIR",
                    # MODIFIED: Changed fallback for NOT_QUALIFIED to DNC
                    CallOutcome.NOT_QUALIFIED: "DNC",
                }
                disposition = fallback_map.get(outcome, "NP")
                call_data['disposition'] = disposition
                self.logger.info(f"Fallback disposition set to '{disposition}'")

            # Generate descriptive comments for the call log
            comments = self._generate_comments(outcome, call_data, intent_data)
            call_data['comments'] = comments
            
            # Queue ViciDial disposition report for async processing
            return self._report_to_vicidial_async(phone_number, disposition, call_data)
                
        except Exception as e:
            self.logger.error(f"[Call Outcome] Error processing outcome: {e}", exc_info=True)
            return False

    def _generate_comments(self, outcome: CallOutcome, call_data: Dict[str, Any], intent_data: Optional[Dict[str, Any]]) -> str:
        """Generates descriptive comments for the call log based on the final state."""
        disposition = call_data.get('disposition', 'UNKNOWN')
        duration = call_data.get('duration', 0)
        base_comment = f"Voicebot Call. Disposition: {disposition}. Duration: {duration}s."

        if outcome == CallOutcome.NEGATIVE_INTENT:
            intent = intent_data.get('intent', 'unknown') if intent_data else 'unknown'
            call_data['intent_detected'] = intent
            return f"Voicebot: Negative intent detected - '{intent}'. {base_comment}"
        
        if outcome == CallOutcome.VOICEMAIL:
            call_data['is_voicemail'] = True
            return f"Voicebot: Voicemail detected. {base_comment}"

        if outcome == CallOutcome.RINGING:
            call_data['is_ringing'] = True
            return f"Voicebot: Ringing detected. {base_comment}"

        if outcome == CallOutcome.TRANSFERRED:
            return f"Voicebot: Qualified and transferred to agent. {base_comment}"
        
        # Use pre-set comments for specific failures if available
        if 'comments' in call_data and call_data['comments']:
            return f"Voicebot: {call_data['comments']} {base_comment}"

        if call_data.get('error'):
            return f"Voicebot: Call failed with error: {call_data['error']}. {base_comment}"

        # Default comment includes transcript summary if no other specific comment was generated
        transcript = call_data.get('transcript', '')
        if transcript:
            transcript_summary = transcript.replace('\n', ' | ').strip()[:100]
            return f"{base_comment} Transcript: {transcript_summary}..."
        
        return base_comment
    
    def _report_to_vicidial_async(self, phone_number: str, disposition: str, 
                                 call_data: Dict[str, Any]) -> bool:
        """Queue ViciDial disposition report for async processing - non-blocking"""
        if not self.vicidial_integration:
            self.logger.warning(f"[Call Outcome] ViciDial integration not available. Cannot report {disposition}.")
            return True
        
        try:
            # Use async reporter to queue disposition report - returns immediately
            report_id = self.async_vicidial_reporter.queue_disposition_report(
                phone_number, disposition, call_data, self.vicidial_integration
            )
            
            self.logger.info(f"[Call Outcome] ✅ ViciDial disposition {disposition} queued for {phone_number} (ID: {report_id})")
            return True  # Always return True since we successfully queued it
            
        except Exception as e:
            self.logger.error(f"[Call Outcome] Error queueing ViciDial report: {e}", exc_info=True)
            return False
    
    def _report_to_vicidial(self, phone_number: str, disposition: str, 
                          call_data: Dict[str, Any]) -> bool:
        """Legacy synchronous ViciDial reporting - kept for backward compatibility"""
        if not self.vicidial_integration:
            self.logger.warning(f"[Call Outcome] ViciDial integration not available. Cannot report {disposition}.")
            return True
        
        try:
            success = self.vicidial_integration.send_disposition_directly(
                phone_number, disposition, call_data
            )
            
            if success:
                self.logger.info(f"[Call Outcome] ✅ Reported {disposition} to ViciDial for {phone_number}")
            else:
                self.logger.error(f"[Call Outcome] ❌ Failed to report {disposition} to ViciDial")
            
            return success
            
        except Exception as e:
            self.logger.error(f"[Call Outcome] Error reporting to ViciDial: {e}", exc_info=True)
            return False
    
    def create_callback_request(self, phone_number: str, call_data: Dict[str, Any], 
                              callback_datetime: str = None) -> bool:
        """Create a callback request in ViciDial"""
        if not self.vicidial_integration:
            return False
        
        try:
            if not callback_datetime:
                from datetime import datetime, timedelta
                callback_time = datetime.now() + timedelta(minutes=2)
                callback_datetime = callback_time.strftime('%Y-%m-%d %H:%M:%S')
            
            success = self.vicidial_integration.handle_callback_request(
                phone_number,
                callback_datetime,
                call_data
            )
            
            if success:
                self.logger.info(f"[Call Outcome] ✅ Callback request created for {phone_number}")
            else:
                self.logger.error(f"[Call Outcome] ❌ Failed to create callback request")
            
            return success
            
        except Exception as e:
            self.logger.error(f"[Call Outcome] Error creating callback request: {e}", exc_info=True)
            return False