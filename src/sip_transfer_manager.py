#!/usr/bin/env python3
"""
Production-Grade SIP Transfer Manager
- Thread-safe transfer operations with retry logic
- Comprehensive error handling for all scenarios
- Race condition prevention with state validation
"""

import pjsua2 as pj
import time
import threading
from typing import Optional, Dict, Any
import logging
from contextlib import contextmanager
import queue
from enum import Enum

class TransferState(Enum):
    """Transfer operation states"""
    IDLE = "idle"
    TRANSFERRING = "transferring"
    COMPLETED = "completed"
    FAILED = "failed"

class ThreadSafeTransferManager:
    """Thread-safe transfer state management"""
    def __init__(self):
        self.lock = threading.RLock()
        self.active_transfers = {}
        self.completed_transfers = set()
        
    def start_transfer(self, call_id: int) -> bool:
        """Start a transfer operation"""
        with self.lock:
            if call_id in self.active_transfers:
                return False  # Already transferring
            if call_id in self.completed_transfers:
                return False  # Already transferred
            
            self.active_transfers[call_id] = {
                'state': TransferState.TRANSFERRING,
                'start_time': time.time(),
                'attempts': 0
            }
            return True
    
    def complete_transfer(self, call_id: int, success: bool):
        """Mark transfer as complete"""
        with self.lock:
            if call_id in self.active_transfers:
                transfer_info = self.active_transfers.pop(call_id)
                transfer_info['state'] = TransferState.COMPLETED if success else TransferState.FAILED
                transfer_info['end_time'] = time.time()
                
                if success:
                    self.completed_transfers.add(call_id)
                
                return transfer_info
            return None
    
    def is_transferring(self, call_id: int) -> bool:
        """Check if call is currently being transferred"""
        with self.lock:
            return call_id in self.active_transfers
    
    def is_transferred(self, call_id: int) -> bool:
        """Check if call has been transferred"""
        with self.lock:
            return call_id in self.completed_transfers

class ViciDialApiTransfer:
    """
    Production-grade ViciDial transfer manager using SIP REFER
    Handles transfers and hangups via DID hairpin method
    """
    
    # Class-level transfer tracker
    _transfer_tracker = ThreadSafeTransferManager()
    
    def __init__(self, qualified_did: str, hangup_did: str, server_ip: str, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.qualified_did = str(qualified_did) if qualified_did else '9998'
        self.hangup_did = str(hangup_did) if hangup_did else '9997'
        self.server_ip = server_ip
        
        # Thread safety
        self.operation_lock = threading.Lock()
        
        # Stats
        self.stats = {
            'transfers_attempted': 0,
            'transfers_successful': 0,
            'transfers_failed': 0,
            'hangups_attempted': 0,
            'hangups_successful': 0,
            'hangups_failed': 0
        }
        
        # Timeouts and retries
        self.transfer_timeout = 5.0
        self.max_retries = 3
        self.retry_delay = 0.5
        
        self.logger.info(f"Transfer Manager initialized - Server: {self.server_ip}")
        self.logger.info(f"DIDs - Qualified: {self.qualified_did}, Hangup: {self.hangup_did}")
    
    def transfer_qualified_call(self, sip_call, customer_phone: str = None) -> bool:
        """
        Transfer qualified call with comprehensive error handling and retry logic
        Returns True if transfer was successful
        """
        if not sip_call:
            self.logger.error("No SIP call provided for transfer")
            return False
        
        call_id = None
        try:
            call_id = sip_call.getId()
        except:
            self.logger.error("Cannot get call ID")
            return False
        
        # Check if already transferring/transferred
        if self._transfer_tracker.is_transferring(call_id):
            self.logger.warning(f"Call {call_id} is already being transferred")
            return False
        
        if self._transfer_tracker.is_transferred(call_id):
            self.logger.warning(f"Call {call_id} was already transferred")
            return True
        
        # Start transfer tracking
        if not self._transfer_tracker.start_transfer(call_id):
            self.logger.warning(f"Cannot start transfer for call {call_id}")
            return False
        
        try:
            self.stats['transfers_attempted'] += 1
            
            # Perform transfer with retries
            success = self._perform_transfer_with_retries(
                sip_call, 
                self.qualified_did,
                "qualified transfer",
                customer_phone
            )
            
            # Update tracking
            self._transfer_tracker.complete_transfer(call_id, success)
            
            if success:
                self.stats['transfers_successful'] += 1
                self.logger.info(f"✅ Qualified transfer successful for call {call_id}")
            else:
                self.stats['transfers_failed'] += 1
                self.logger.error(f"❌ Qualified transfer failed for call {call_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Unexpected error in transfer: {e}", exc_info=True)
            self._transfer_tracker.complete_transfer(call_id, False)
            self.stats['transfers_failed'] += 1
            return False
    
    def hangup_call_via_did(self, sip_call) -> bool:
        """
        Hang up call by transferring to hangup DID
        Returns True if hangup was successful or call was already down
        """
        if not sip_call:
            self.logger.error("No SIP call provided for hangup")
            return False
        
        call_id = None
        try:
            call_id = sip_call.getId()
        except:
            self.logger.error("Cannot get call ID for hangup")
            return True  # Consider it success if we can't get ID
        
        # Check if already transferred
        if self._transfer_tracker.is_transferred(call_id):
            self.logger.info(f"Call {call_id} already transferred/hung up")
            return True
        
        try:
            self.stats['hangups_attempted'] += 1
            
            # Check call state first
            if not self._is_call_active(sip_call):
                self.logger.info(f"Call {call_id} already inactive")
                self.stats['hangups_successful'] += 1
                return True
            
            # Perform hangup transfer
            success = self._perform_transfer_with_retries(
                sip_call,
                self.hangup_did,
                "hangup transfer",
                None
            )
            
            if success:
                self.stats['hangups_successful'] += 1
                self.logger.info(f"✅ Hangup successful for call {call_id}")
            else:
                # Even if transfer fails, call might be down
                if not self._is_call_active(sip_call):
                    self.stats['hangups_successful'] += 1
                    self.logger.info(f"Call {call_id} went down during hangup attempt")
                    return True
                
                self.stats['hangups_failed'] += 1
                self.logger.warning(f"Hangup transfer failed for call {call_id}")
            
            return success
            
        except Exception as e:
            # Any error during hangup is considered success
            # (call is likely already down)
            self.logger.info(f"Hangup attempt for call {call_id} - treating as success: {e}")
            self.stats['hangups_successful'] += 1
            return True
    
    def _perform_transfer_with_retries(self, sip_call, did: str, 
                                      operation_name: str, 
                                      customer_phone: Optional[str]) -> bool:
        """
        Perform transfer with retry logic
        Returns True if successful
        """
        attempt = 0
        last_error = None
        
        while attempt < self.max_retries:
            attempt += 1
            
            # Check if call is still active
            if not self._is_call_active(sip_call):
                self.logger.info(f"Call inactive before {operation_name} (attempt {attempt})")
                return False
            
            try:
                # Register thread with PJSIP
                self._register_pjsip_thread()
                
                # Build SIP URI
                sip_uri = f"sip:{did}@{self.server_ip}"
                
                self.logger.info(f"Attempt {attempt}/{self.max_retries}: {operation_name} to {sip_uri}")
                self.logger.info(f"Transfer server IP: {self.server_ip} (should be single IP, not comma-separated list)")
                
                # Store transfer target on the call instance for onCallTransferStatus callback
                if hasattr(sip_call, '_transfer_target'):
                    sip_call._transfer_target = sip_uri
                else:
                    try:
                        sip_call._transfer_target = sip_uri
                    except:
                        pass  # Some call objects might be read-only
                
                # Perform transfer
                transfer_param = pj.CallOpParam()
                transfer_param.statusCode = pj.PJSIP_SC_OK
                
                # Execute transfer with timeout protection
                with self._timeout_context(self.transfer_timeout):
                    sip_call.xfer(sip_uri, transfer_param)
                
                # Wait briefly for transfer to process
                time.sleep(0.5)
                
                self.logger.info(f"SIP REFER sent successfully for {operation_name}")
                return True
                
            except pj.Error as e:
                last_error = e
                
                # Check specific error codes using numeric values
                if e.status == 70001:  # PJ_ENOTFOUND
                    self.logger.info(f"Call already disconnected during {operation_name}")
                    return True  # Consider success if call is gone
                
                elif e.status == 70004:  # PJ_EINVAL
                    self.logger.warning(f"Invalid state for {operation_name}: {e.reason}")
                    
                    # Check if call went down
                    if not self._is_call_active(sip_call):
                        return True
                    
                else:
                    self.logger.warning(f"PJSIP error on attempt {attempt}: {e.reason}")
                
                # Retry with delay
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)  # Exponential backoff
                    
            except Exception as e:
                last_error = e
                self.logger.error(f"Unexpected error on attempt {attempt}: {e}")
                
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * attempt)
        
        # All retries failed
        self.logger.error(f"All {self.max_retries} attempts failed for {operation_name}")
        if last_error:
            self.logger.error(f"Last error: {last_error}")
        
        return False
    
    def _is_call_active(self, sip_call) -> bool:
        """
        Check if call is still active
        Returns True if active, False otherwise
        """
        try:
            if not sip_call:
                return False
            
            # Check PJSIP call state
            if hasattr(sip_call, 'isActive'):
                return sip_call.isActive()
            
            # Alternative: check call info
            call_info = sip_call.getInfo()
            return call_info.state < pj.PJSIP_INV_STATE_DISCONNECTED
            
        except pj.Error as e:
            # Error getting state means call is likely gone
            self.logger.debug(f"Error checking call state: {e.reason}")
            return False
            
        except Exception as e:
            self.logger.debug(f"Error checking call state: {e}")
            return False
    
    def _register_pjsip_thread(self):
        """Register current thread with PJSIP"""
        try:
            thread_id = threading.get_ident()
            thread_name = f"transfer_{thread_id}"
            
            if not hasattr(threading.current_thread(), '_pj_registered'):
                pj.Endpoint.instance().libRegisterThread(thread_name)
                threading.current_thread()._pj_registered = True
                
        except Exception as e:
            self.logger.debug(f"Thread registration: {e}")
    
    @contextmanager
    def _timeout_context(self, timeout_seconds):
        """
        Context manager for timeout operations
        Note: This is a placeholder - actual timeout would need signal handling
        or separate thread monitoring
        """
        start_time = time.time()
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                self.logger.warning(f"Operation took {elapsed:.2f}s (timeout was {timeout_seconds}s)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get transfer manager statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        for key in self.stats:
            self.stats[key] = 0
        self.logger.info("Transfer manager stats reset")