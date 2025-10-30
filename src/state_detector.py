#!/usr/bin/env python3
"""
US State Detection Module for SIP Bot
- Detects US state based on phone number area codes
- Thread-safe operations with caching
- Handles edge cases and international numbers
"""

import re
import logging
import threading
from typing import Optional, Dict
from functools import lru_cache

class StateDetector:
    """
    Detects US state from phone number area codes
    """
    
    # US State to Area Code mappings
    STATE_AREA_CODES = {
        "Alaska": ["907"],
        "Alabama": ["205", "251", "256", "334"],
        "Arkansas": ["479", "501", "870"],
        "Arizona": ["480", "520", "602", "623", "928"],
        "California": ["209", "213", "310", "323", "408", "415", "510", "530", "559", "562",
                       "619", "626", "650", "661", "707", "714", "760", "805", "818", "831",
                       "858", "909", "916", "925", "949", "951"],
        "Colorado": ["303", "719", "970"],
        "Connecticut": ["203", "860"],
        "District of Columbia": ["202"],
        "Delaware": ["302"],
        "Florida": ["239", "305", "321", "352", "386", "407", "561", "727", "772",
                    "813", "850", "863", "904", "941", "954"],
        "Georgia": ["229", "404", "478", "706", "770", "912"],
        "Hawaii": ["808"],
        "Iowa": ["319", "515", "563", "641", "712"],
        "Idaho": ["208"],
        "Illinois": ["217", "309", "312", "618", "630", "708", "773", "815", "847"],
        "Indiana": ["219", "260", "317", "574", "765", "812"],
        "Kansas": ["316", "620", "785", "913"],
        "Kentucky": ["270", "502", "606", "859"],
        "Louisiana": ["225", "318", "337", "504", "985"],
        "Massachusetts": ["413", "508", "617", "781", "978"],
        "Maryland": ["301", "410"],
        "Maine": ["207"],
        "Michigan": ["231", "248", "269", "313", "517", "586", "616", "734", "810", "906", "989"],
        "Minnesota": ["218", "320", "507", "612", "651", "763", "952"],
        "Missouri": ["314", "417", "573", "636", "660", "816"],
        "Mississippi": ["228", "601", "662"],
        "Montana": ["406"],
        "North Carolina": ["252", "336", "704", "828", "910", "919"],
        "North Dakota": ["701"],
        "Nebraska": ["308", "402"],
        "New Hampshire": ["603"],
        "New Jersey": ["201", "609", "732", "856", "908", "973"],
        "New Mexico": ["505", "575"],
        "Nevada": ["702", "775"],
        "New York": ["212", "315", "516", "518", "585", "607", "631", "716", "718", "845", "914"],
        "Ohio": ["216", "330", "419", "440", "513", "614", "740", "937"],
        "Oklahoma": ["405", "580", "918"],
        "Oregon": ["503", "541"],
        "Pennsylvania": ["215", "412", "570", "610", "717", "724", "814"],
        "Rhode Island": ["401"],
        "South Carolina": ["803", "843", "864"],
        "South Dakota": ["605"],
        "Tennessee": ["423", "615", "731", "865", "901", "931"],
        "Texas": ["210", "214", "254", "281", "325", "361", "409", "432", "512", "713",
                  "806", "817", "830", "903", "915", "936", "940", "956", "972", "979"],
        "Utah": ["435", "801"],
        "Virginia": ["276", "434", "540", "703", "757", "804"],
        "Vermont": ["802"],
        "Washington": ["206", "253", "360", "425", "509"],
        "Wisconsin": ["262", "414", "608", "715", "920"],
        "West Virginia": ["304"],
        "Wyoming": ["307"]
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.lock = threading.Lock()
        
        # Create reverse mapping: area_code -> state for O(1) lookup
        self._area_code_to_state = {}
        for state, area_codes in self.STATE_AREA_CODES.items():
            for area_code in area_codes:
                self._area_code_to_state[area_code] = state
        
        # Stats
        self.stats = {
            'total_lookups': 0,
            'successful_detections': 0,
            'unknown_area_codes': 0,
            'invalid_numbers': 0
        }
        
        self.logger.info(f"State detector initialized with {len(self._area_code_to_state)} area codes")
    
    def detect_state(self, phone_number: str) -> Optional[str]:
        """
        Detect US state from phone number
        
        Args:
            phone_number: Phone number (can be formatted or raw)
            
        Returns:
            State name or None if not found/not US number
        """
        with self.lock:
            self.stats['total_lookups'] += 1
        
        try:
            # Extract area code from phone number
            area_code = self._extract_area_code(phone_number)
            if not area_code:
                with self.lock:
                    self.stats['invalid_numbers'] += 1
                return None
            
            # Look up state
            state = self._lookup_state_by_area_code(area_code)
            if state:
                with self.lock:
                    self.stats['successful_detections'] += 1
                self.logger.debug(f"Detected state: {state} for area code {area_code}")
                return state
            else:
                with self.lock:
                    self.stats['unknown_area_codes'] += 1
                self.logger.debug(f"Unknown area code: {area_code}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error detecting state for {phone_number}: {e}")
            with self.lock:
                self.stats['invalid_numbers'] += 1
            return None
    
    def _extract_area_code(self, phone_number: str) -> Optional[str]:
        """
        Extract area code from phone number
        
        Args:
            phone_number: Phone number in various formats
            
        Returns:
            3-digit area code or None
        """
        if not phone_number:
            return None
        
        # Remove all non-digits
        digits_only = re.sub(r'\D', '', phone_number)
        
        # Handle different number formats
        if len(digits_only) == 11 and digits_only.startswith('1'):
            # US number with country code: 1AAANNNNNNN
            return digits_only[1:4]
        elif len(digits_only) == 10:
            # US number without country code: AAANNNNNNN
            return digits_only[:3]
        else:
            # Not a US number format
            return None
    
    @lru_cache(maxsize=1000)
    def _lookup_state_by_area_code(self, area_code: str) -> Optional[str]:
        """
        Look up state by area code with caching
        
        Args:
            area_code: 3-digit area code
            
        Returns:
            State name or None
        """
        return self._area_code_to_state.get(area_code)
    
    def get_area_codes_for_state(self, state: str) -> list:
        """
        Get all area codes for a given state
        
        Args:
            state: State name
            
        Returns:
            List of area codes for the state
        """
        return self.STATE_AREA_CODES.get(state, [])
    
    def get_all_states(self) -> list:
        """Get list of all supported states"""
        return list(self.STATE_AREA_CODES.keys())
    
    def get_stats(self) -> Dict[str, int]:
        """Get detection statistics"""
        with self.lock:
            stats = self.stats.copy()
        
        # Calculate success rate
        if stats['total_lookups'] > 0:
            stats['success_rate'] = (stats['successful_detections'] / stats['total_lookups']) * 100
        else:
            stats['success_rate'] = 0.0
        
        return stats
    
    def is_us_number(self, phone_number: str) -> bool:
        """
        Check if phone number is a US number
        
        Args:
            phone_number: Phone number to check
            
        Returns:
            True if US number format, False otherwise
        """
        area_code = self._extract_area_code(phone_number)
        return area_code is not None
    
    def format_state_info(self, phone_number: str, state: str = None) -> str:
        """
        Format state information for display
        
        Args:
            phone_number: The phone number
            state: Detected state (if None, will detect)
            
        Returns:
            Formatted string with phone and state info
        """
        if state is None:
            state = self.detect_state(phone_number)
        
        if state:
            area_code = self._extract_area_code(phone_number)
            return f"{phone_number} ({area_code} - {state})"
        else:
            return f"{phone_number} (State: Unknown)"
    
    def clear_cache(self):
        """Clear the LRU cache"""
        self._lookup_state_by_area_code.cache_clear()
        self.logger.info("State detector cache cleared")