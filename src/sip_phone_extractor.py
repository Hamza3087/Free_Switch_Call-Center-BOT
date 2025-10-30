#!/usr/bin/env python3
"""
Production-Grade SIP Phone Number Extractor
- Multiple extraction strategies with fallbacks
- Comprehensive error handling
- Thread-safe operations
"""

import re
import time
import logging
import hashlib
from typing import Optional, List, Tuple
from dataclasses import dataclass
import threading

@dataclass
class PhoneExtractionResult:
    """Result of phone extraction attempt"""
    phone_number: str
    extraction_method: str
    confidence: float  # 0.0 to 1.0
    is_fallback: bool
    caller_state: Optional[str] = None  # US state detected from area code

class ExtractionStrategy:
    """Base class for extraction strategies"""
    def __init__(self, name: str, priority: int):
        self.name = name
        self.priority = priority
        self.logger = logging.getLogger(__name__)
    
    def extract(self, sip_message: str) -> Optional[str]:
        """Extract phone number from SIP message"""
        raise NotImplementedError
    
    def validate(self, number: str) -> Optional[str]:
        """Validate and normalize phone number"""
        if not number:
            return None
        
        # Remove all non-digits
        cleaned = re.sub(r'\D', '', number)
        
        # Check for valid lengths
        if len(cleaned) == 11 and cleaned.startswith('1'):
            return cleaned  # US number with country code
        elif len(cleaned) == 10:
            return '1' + cleaned  # US number without country code
        elif len(cleaned) == 7:
            return None  # Too short, likely extension
        elif len(cleaned) > 11 and len(cleaned) <= 15:
            return cleaned  # International number
        else:
            return None  # Invalid length

class RequestURIStrategy(ExtractionStrategy):
    """Extract from INVITE Request-URI"""
    def __init__(self):
        super().__init__("Request-URI", 1)
        self.pattern = re.compile(r'INVITE\s+sip:([^@\s]+)@', re.IGNORECASE)
    
    def extract(self, sip_message: str) -> Optional[str]:
        match = self.pattern.search(sip_message)
        if match:
            return self.validate(match.group(1))
        return None

class FromHeaderStrategy(ExtractionStrategy):
    """Extract from From header"""
    def __init__(self):
        super().__init__("From-Header", 2)
        self.patterns = [
            re.compile(r'From:.*?<sip:([^@>]+)@', re.IGNORECASE),
            re.compile(r'From:\s*"?[^"]*"?\s*<sip:([^@>]+)@', re.IGNORECASE),
            re.compile(r'From:\s*sip:([^@\s;]+)@', re.IGNORECASE)
        ]
    
    def extract(self, sip_message: str) -> Optional[str]:
        for pattern in self.patterns:
            match = pattern.search(sip_message)
            if match:
                result = self.validate(match.group(1))
                if result:
                    return result
        return None

class ToHeaderStrategy(ExtractionStrategy):
    """Extract from To header"""
    def __init__(self):
        super().__init__("To-Header", 3)
        self.patterns = [
            re.compile(r'To:.*?<sip:([^@>]+)@', re.IGNORECASE),
            re.compile(r'To:\s*sip:([^@\s;]+)@', re.IGNORECASE)
        ]
    
    def extract(self, sip_message: str) -> Optional[str]:
        for pattern in self.patterns:
            match = pattern.search(sip_message)
            if match:
                result = self.validate(match.group(1))
                if result:
                    return result
        return None

class ContactHeaderStrategy(ExtractionStrategy):
    """Extract from Contact header"""
    def __init__(self):
        super().__init__("Contact-Header", 4)
        self.pattern = re.compile(r'Contact:.*?<sip:([^@>]+)@', re.IGNORECASE)
    
    def extract(self, sip_message: str) -> Optional[str]:
        match = self.pattern.search(sip_message)
        if match:
            return self.validate(match.group(1))
        return None

class PAIHeaderStrategy(ExtractionStrategy):
    """Extract from P-Asserted-Identity header"""
    def __init__(self):
        super().__init__("PAI-Header", 5)
        self.pattern = re.compile(r'P-Asserted-Identity:.*?<sip:([^@>]+)@', re.IGNORECASE)
    
    def extract(self, sip_message: str) -> Optional[str]:
        match = self.pattern.search(sip_message)
        if match:
            return self.validate(match.group(1))
        return None

class RemotePartyIDStrategy(ExtractionStrategy):
    """Extract from Remote-Party-ID header"""
    def __init__(self):
        super().__init__("Remote-Party-ID", 6)
        self.pattern = re.compile(r'Remote-Party-ID:.*?<sip:([^@>]+)@', re.IGNORECASE)
    
    def extract(self, sip_message: str) -> Optional[str]:
        match = self.pattern.search(sip_message)
        if match:
            return self.validate(match.group(1))
        return None

class GenericScanStrategy(ExtractionStrategy):
    """Generic scan for phone numbers in entire message"""
    def __init__(self):
        super().__init__("Generic-Scan", 99)
        self.patterns = [
            re.compile(r'\b1?(\d{3})[- .]?(\d{3})[- .]?(\d{4})\b'),  # US format
            re.compile(r'\b(1?\d{10,11})\b'),  # Continuous digits
            re.compile(r'sip:(\d{7,15})[@\s;]'),  # SIP URI with digits
        ]
    
    def extract(self, sip_message: str) -> Optional[str]:
        for pattern in self.patterns:
            matches = pattern.findall(sip_message)
            for match in matches:
                # Handle different match types
                if isinstance(match, tuple):
                    # Formatted number
                    number = ''.join(match)
                else:
                    number = match
                
                result = self.validate(number)
                if result:
                    return result
        return None

class SIPPhoneExtractor:
    """
    Production-grade phone number extractor with multiple strategies
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize extraction strategies
        self.strategies = [
            RequestURIStrategy(),
            FromHeaderStrategy(),
            ToHeaderStrategy(),
            ContactHeaderStrategy(),
            PAIHeaderStrategy(),
            RemotePartyIDStrategy(),
            GenericScanStrategy()
        ]
        
        # Sort by priority
        self.strategies.sort(key=lambda s: s.priority)
        
        # Initialize state detector
        self.state_detector = None  # Lazy load
        
        # Cache for performance
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = 1000
        
        # Stats
        self.stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'fallback_extractions': 0,
            'cache_hits': 0,
            'errors': 0,
            'state_detections': 0
        }
        
        self.logger.info("Phone extractor initialized with {} strategies".format(
            len(self.strategies)))
    
    def _lazy_load_state_detector(self):
        """Lazy load state detector to avoid import issues"""
        if self.state_detector is None:
            from src.state_detector import StateDetector
            self.state_detector = StateDetector()
        return self.state_detector
    
    def extract_phone_number(self, prm) -> str:
        """
        Extract phone number from incoming call parameters
        Returns phone number or fallback identifier
        """
        self.stats['total_extractions'] += 1
        
        try:
            # Get SIP message
            sip_message = None
            if hasattr(prm, 'rdata') and hasattr(prm.rdata, 'wholeMsg'):
                sip_message = prm.rdata.wholeMsg
            
            if not sip_message:
                self.logger.warning("No SIP message available")
                self.stats['fallback_extractions'] += 1
                return self._generate_fallback_identifier(prm)
            
            # Check cache
            cache_key = self._get_cache_key(sip_message)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                self.stats['successful_extractions'] += 1
                self.logger.info(f"✅ Found in cache: {cached_result}")
                return cached_result
            
            # Try each strategy
            result = self._extract_with_strategies(sip_message)
            
            if result.phone_number and not result.is_fallback:
                # Cache successful extraction
                self._cache_result(cache_key, result.phone_number)
                self.stats['successful_extractions'] += 1
                
                state_info = f" (State: {result.caller_state})" if result.caller_state else ""
                self.logger.info(f"✅ Extracted via {result.extraction_method}: {result.phone_number}{state_info}")
                return result.phone_number
            else:
                # Fallback
                self.stats['fallback_extractions'] += 1
                self.logger.warning(f"⚠️ Using fallback: {result.phone_number}")
                return result.phone_number
                
        except Exception as e:
            self.logger.error(f"Extraction error: {e}", exc_info=True)
            self.stats['errors'] += 1
            self.stats['fallback_extractions'] += 1
            return self._generate_error_fallback()
    
    def extract_phone_with_state(self, prm) -> PhoneExtractionResult:
        """
        Extract phone number with state information
        Returns PhoneExtractionResult with state
        """
        self.stats['total_extractions'] += 1
        
        try:
            # Get SIP message
            sip_message = None
            if hasattr(prm, 'rdata') and hasattr(prm.rdata, 'wholeMsg'):
                sip_message = prm.rdata.wholeMsg
            
            if not sip_message:
                self.logger.warning("No SIP message available")
                self.stats['fallback_extractions'] += 1
                return PhoneExtractionResult(
                    phone_number=self._generate_fallback_identifier(prm),
                    extraction_method="Fallback",
                    confidence=0.0,
                    is_fallback=True,
                    caller_state=None
                )
            
            # Check cache (for phone number only)
            cache_key = self._get_cache_key(sip_message)
            cached_phone = self._get_cached_result(cache_key)
            if cached_phone:
                self.stats['cache_hits'] += 1
                self.stats['successful_extractions'] += 1
                # Still need to detect state for cached results
                state = self._detect_state_safe(cached_phone)
                self.logger.info(f"✅ Found in cache: {cached_phone}")
                return PhoneExtractionResult(
                    phone_number=cached_phone,
                    extraction_method="Cache",
                    confidence=1.0,
                    is_fallback=False,
                    caller_state=state
                )
            
            # Try each strategy
            result = self._extract_with_strategies(sip_message)
            
            if result.phone_number and not result.is_fallback:
                # Cache successful extraction
                self._cache_result(cache_key, result.phone_number)
                self.stats['successful_extractions'] += 1
                
                state_info = f" (State: {result.caller_state})" if result.caller_state else ""
                self.logger.info(f"✅ Extracted via {result.extraction_method}: {result.phone_number}{state_info}")
            else:
                # Fallback
                self.stats['fallback_extractions'] += 1
                self.logger.warning(f"⚠️ Using fallback: {result.phone_number}")
            
            return result
                
        except Exception as e:
            self.logger.error(f"Extraction error: {e}", exc_info=True)
            self.stats['errors'] += 1
            self.stats['fallback_extractions'] += 1
            return PhoneExtractionResult(
                phone_number=self._generate_error_fallback(),
                extraction_method="Error",
                confidence=0.0,
                is_fallback=True,
                caller_state=None
            )
    
    def _extract_with_strategies(self, sip_message: str) -> PhoneExtractionResult:
        """
        Try each extraction strategy in order
        Returns extraction result with state detection
        """
        for strategy in self.strategies:
            try:
                phone = strategy.extract(sip_message)
                if phone:
                    # Detect state for the phone number
                    state = self._detect_state_safe(phone)
                    return PhoneExtractionResult(
                        phone_number=phone,
                        extraction_method=strategy.name,
                        confidence=1.0 - (strategy.priority / 100.0),
                        is_fallback=False,
                        caller_state=state
                    )
            except Exception as e:
                self.logger.debug(f"Strategy {strategy.name} failed: {e}")
        
        # All strategies failed - generate fallback
        fallback = self._generate_fallback_from_message(sip_message)
        return PhoneExtractionResult(
            phone_number=fallback,
            extraction_method="Fallback",
            confidence=0.0,
            is_fallback=True,
            caller_state=None  # No state for fallback IDs
        )
    
    def _detect_state_safe(self, phone_number: str) -> Optional[str]:
        """
        Safely detect state from phone number
        """
        try:
            state_detector = self._lazy_load_state_detector()
            state = state_detector.detect_state(phone_number)
            if state:
                self.stats['state_detections'] += 1
            return state
        except Exception as e:
            self.logger.debug(f"State detection failed for {phone_number}: {e}")
            return None
    
    def _get_cache_key(self, sip_message: str) -> str:
        """Generate cache key from SIP message"""
        # Extract key parts of message for caching
        key_parts = []
        
        # Extract INVITE line
        invite_match = re.search(r'INVITE\s+[^\r\n]+', sip_message)
        if invite_match:
            key_parts.append(invite_match.group())
        
        # Extract From header
        from_match = re.search(r'From:[^\r\n]+', sip_message, re.IGNORECASE)
        if from_match:
            key_parts.append(from_match.group())
        
        # Create hash
        key_string = '|'.join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[str]:
        """Get cached extraction result"""
        with self.cache_lock:
            return self.cache.get(cache_key)
    
    def _cache_result(self, cache_key: str, phone_number: str):
        """Cache extraction result"""
        with self.cache_lock:
            # Limit cache size
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest entries (simple FIFO)
                to_remove = list(self.cache.keys())[:100]
                for key in to_remove:
                    del self.cache[key]
            
            self.cache[cache_key] = phone_number
    
    def _generate_fallback_from_message(self, sip_message: str) -> str:
        """Generate fallback ID from message content"""
        # Try to extract any useful identifier
        identifiers = []
        
        # Look for Call-ID
        call_id_match = re.search(r'Call-ID:\s*([^\r\n]+)', sip_message, re.IGNORECASE)
        if call_id_match:
            call_id = call_id_match.group(1).strip()
            # Take first 6 chars of Call-ID
            identifiers.append(call_id[:6])
        
        # Use timestamp
        timestamp = str(int(time.time()))[-6:]
        
        if identifiers:
            return f"VD{identifiers[0][:4]}{timestamp[-4:]}"
        else:
            return f"VD{timestamp}"
    
    def _generate_fallback_identifier(self, prm) -> str:
        """Generate trackable fallback identifier"""
        try:
            # Try to get any unique identifier from prm
            timestamp = str(int(time.time()))[-6:]
            
            # Try to get call ID if available
            if hasattr(prm, 'callId'):
                call_id = str(prm.callId)[-4:]
                return f"VD{call_id}{timestamp[-4:]}"
            
            return f"VD{timestamp}"
            
        except Exception:
            return self._generate_error_fallback()
    
    def _generate_error_fallback(self) -> str:
        """Generate error fallback identifier"""
        timestamp = str(int(time.time()))[-6:]
        error_count = self.stats.get('errors', 0)
        return f"ERR{error_count:02d}{timestamp[-4:]}"
    
    def format_phone_for_display(self, phone: str) -> str:
        """
        Format phone number for display
        Handles both real numbers and fallback IDs
        """
        if not phone:
            return "Unknown"
        
        # Check if it's a fallback ID
        if phone.startswith(('VD', 'ERR')):
            return phone  # Return as-is
        
        # Check if it's a valid phone number
        if not phone.isdigit():
            return phone
        
        # Format US numbers
        if len(phone) == 11 and phone.startswith('1'):
            return f"+1 ({phone[1:4]}) {phone[4:7]}-{phone[7:]}"
        elif len(phone) == 10:
            return f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"
        
        # Format international numbers (basic)
        if len(phone) > 11:
            return f"+{phone}"
        
        # Return as-is for other formats
        return phone
    
    def get_stats(self) -> dict:
        """Get extractor statistics"""
        stats = self.stats.copy()
        
        # Calculate success rate
        if stats['total_extractions'] > 0:
            stats['success_rate'] = (stats['successful_extractions'] / 
                                    stats['total_extractions']) * 100
            stats['cache_hit_rate'] = (stats['cache_hits'] / 
                                      stats['total_extractions']) * 100
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
        
        return stats
    
    def reset_cache(self):
        """Clear the cache"""
        with self.cache_lock:
            self.cache.clear()
        self.logger.info("Phone extractor cache cleared")