"""
Timezone utilities for determining caller's local time based on US state.
Used for time-based greeting selection in SIP bot calls.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple
import time

# State to primary timezone mapping (UTC offset in hours)
# For states with multiple timezones, uses the most populous area's timezone
STATE_TIMEZONE_MAP = {
    # Eastern Standard Time (UTC-5)
    'CT': -5, 'DE': -5, 'DC': -5, 'GA': -5, 'ME': -5, 'MD': -5,
    'MA': -5, 'NH': -5, 'NJ': -5, 'NY': -5, 'NC': -5, 'OH': -5,
    'PA': -5, 'RI': -5, 'SC': -5, 'VT': -5, 'VA': -5, 'WV': -5,
    'FL': -5,  # Most of Florida
    'IN': -5,  # Most of Indiana
    'KY': -5,  # Eastern Kentucky (more populous)
    'MI': -5,  # Most of Michigan
    'TN': -5,  # East Tennessee (more populous)
    
    # Central Standard Time (UTC-6)
    'AL': -6, 'AR': -6, 'IL': -6, 'IA': -6, 'LA': -6, 'MN': -6,
    'MS': -6, 'MO': -6, 'OK': -6, 'WI': -6,
    'KS': -6,  # Most of Kansas
    'ND': -6,  # Most of North Dakota
    'NE': -6,  # Most of Nebraska
    'SD': -6,  # Eastern South Dakota (more populous)
    'TX': -6,  # Most of Texas
    
    # Mountain Standard Time (UTC-7)
    'AZ': -7, 'CO': -7, 'MT': -7, 'NM': -7, 'UT': -7, 'WY': -7,
    'ID': -7,  # Most of Idaho
    'NV': -7,  # Small part, but West Wendover
    
    # Pacific Standard Time (UTC-8)
    'CA': -8, 'WA': -8,
    'OR': -8,  # Most of Oregon
    
    # Alaska Standard Time (UTC-9)
    'AK': -9,  # Most of Alaska
    
    # Hawaii-Aleutian Standard Time (UTC-10)
    'HI': -10,
    
    # Atlantic Standard Time (UTC-4)
    'PR': -4,  # Puerto Rico
    'VI': -4,  # US Virgin Islands
}

# State full name to 2-letter code mapping (state detector returns full names)
STATE_NAME_TO_CODE = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
    'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL',
    'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA',
    'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN',
    'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT',
    'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
}

# Daylight Saving Time states (most US states observe DST)
DST_STATES = set(STATE_TIMEZONE_MAP.keys()) - {'AZ', 'HI', 'PR', 'VI'}

class TimezoneUtils:
    """Utility class for timezone and time period operations."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_timezone_offset(self, state: str) -> Optional[int]:
        """
        Get the timezone offset for a given US state.
        
        Args:
            state: State name (full name like 'Hawaii' or 2-letter code like 'HI')
            
        Returns:
            UTC offset in hours (negative for US timezones), or None if unknown
        """
        if not state:
            return None
        
        # Convert full state name to 2-letter code if needed
        state_code = STATE_NAME_TO_CODE.get(state, state.upper() if len(state) <= 2 else None)
        
        if not state_code:
            self.logger.debug(f"Unknown state format: {state}")
            return None
        
        offset = STATE_TIMEZONE_MAP.get(state_code)
        
        if offset is None:
            self.logger.debug(f"Unknown state code for timezone lookup: {state_code}")
            return None
        
        # Apply DST if applicable (March to November)
        if self._is_dst_active() and state_code in DST_STATES:
            offset += 1  # DST adds 1 hour
        
        return offset
    
    def get_current_time_for_state(self, state: str) -> Optional[datetime]:
        """
        Get current local time for a given state.
        
        Args:
            state: Two-letter state abbreviation
            
        Returns:
            Local datetime for the state, or None if state unknown
        """
        offset = self.get_timezone_offset(state)
        if offset is None:
            return None
        
        # Get current UTC time and apply offset
        from datetime import timezone
        utc_now = datetime.now(timezone.utc).replace(tzinfo=None)  # Remove timezone info for offset calculation
        local_time = utc_now + timedelta(hours=offset)
        
        return local_time

    
    def _normalize_state_input(self, state: str) -> str:
        """
        Convert state input to two-letter code.
        
        Args:
            state: Either full state name (e.g., "Illinois") or two-letter code (e.g., "IL")
            
        Returns:
            Two-letter state code, or original input if already a code or unknown
        """
        if not state:
            return state
        
        # If already a two-letter code, return as-is (uppercase)
        if len(state) == 2:
            return state.upper()
        
        # Try to convert full state name to code
        state_code = STATE_NAME_TO_CODE.get(state, None)
        if state_code:
            self.logger.debug(f"Converted state name: {state} -> {state_code}")
            return state_code
        
        # Unknown state - return original
        self.logger.debug(f"Unknown state: {state}")
        return state
    
    def get_time_period(self, state: str) -> str:
        """
        Determine the time period (morning/afternoon/evening/night) for a state.
        
        Args:
            state: Two-letter state abbreviation or full state name
            
        Returns:
            Time period string: 'morning', 'afternoon', 'evening', or 'night'
            Returns 'morning' as default if state is unknown
        """
        # Convert full state name to two-letter code if needed
        state_code = self._normalize_state_input(state)
        local_time = self.get_current_time_for_state(state_code)
        
        if local_time is None:
            self.logger.debug(f"Unknown state '{state}', defaulting to morning")
            return 'morning'
        
        hour = local_time.hour
        
        # Time period definitions
        if 5 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 17:
            return 'afternoon'
        elif 17 <= hour < 22:
            return 'evening'
        else:  # 22-5
            return 'night'
    
    def _is_dst_active(self) -> bool:
        """
        Check if Daylight Saving Time is currently active.
        DST typically runs from second Sunday in March to first Sunday in November.
        
        Returns:
            True if DST is active, False otherwise
        """
        now = datetime.now()
        year = now.year
        
        # Calculate DST start (second Sunday in March)
        march_first = datetime(year, 3, 1)
        days_to_second_sunday = (13 - march_first.weekday()) % 7
        if days_to_second_sunday == 0:
            days_to_second_sunday = 7
        dst_start = datetime(year, 3, days_to_second_sunday + 7)
        
        # Calculate DST end (first Sunday in November)
        november_first = datetime(year, 11, 1)
        days_to_first_sunday = (6 - november_first.weekday()) % 7
        dst_end = datetime(year, 11, days_to_first_sunday + 1)
        
        return dst_start <= now < dst_end
    
    def get_time_info_for_state(self, state: str) -> dict:
        """
        Get comprehensive time information for a state.
        
        Args:
            state: Two-letter state abbreviation
            
        Returns:
            Dictionary with timezone offset, local time, and time period
        """
        offset = self.get_timezone_offset(state)
        local_time = self.get_current_time_for_state(state)
        time_period = self.get_time_period(state)
        
        return {
            'state': state,
            'timezone_offset': offset,
            'local_time': local_time,
            'time_period': time_period,
            'dst_active': self._is_dst_active() if state and state.upper() in DST_STATES else False
        }

# Convenience functions for direct use
_timezone_utils = TimezoneUtils()

def get_timezone_offset(state: str) -> Optional[int]:
    """Get timezone offset for a state."""
    return _timezone_utils.get_timezone_offset(state)

def get_current_time_for_state(state: str) -> Optional[datetime]:
    """Get current local time for a state."""
    return _timezone_utils.get_current_time_for_state(state)

def get_time_period(state: str) -> str:
    """Get time period (morning/afternoon/evening/night) for a state."""
    return _timezone_utils.get_time_period(state)

def get_time_info_for_state(state: str) -> dict:
    """Get comprehensive time information for a state."""
    return _timezone_utils.get_time_info_for_state(state)

# Testing function
def test_timezone_utils():
    """Test the timezone utilities with various states."""
    test_states = ['CA', 'NY', 'TX', 'FL', 'AK', 'HI', 'UNKNOWN']
    
    print("Testing timezone utilities:")
    print("-" * 50)
    
    for state in test_states:
        info = get_time_info_for_state(state)
        if info['local_time']:
            time_str = info['local_time'].strftime('%Y-%m-%d %H:%M:%S')
        else:
            time_str = "Unknown"
        
        print(f"State: {state}")
        print(f"  Timezone Offset: {info['timezone_offset']}")
        print(f"  Local Time: {time_str}")
        print(f"  Time Period: {info['time_period']}")
        print(f"  DST Active: {info['dst_active']}")
        print()

if __name__ == "__main__":
    test_timezone_utils()