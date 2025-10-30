"""
Framework-agnostic ringing detection components
Reusable across pjsua2, FreeSwitch, or any other platform
"""

import numpy as np
import time
from typing import List, Tuple, Optional


class GoertzelDetector:
    """Goertzel algorithm implementation for frequency detection"""

    def __init__(self, sample_rate: int = 8000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

    def detect_frequency(self, samples: np.ndarray, target_freq: float) -> float:
        """Compute power at target frequency using Goertzel algorithm"""
        n = len(samples)
        if n == 0:
            return 0.0

        k = int(0.5 + ((n * target_freq) / self.sample_rate))
        omega = (2.0 * np.pi * k) / n
        coeff = 2.0 * np.cos(omega)

        q0 = q1 = q2 = 0.0
        for sample in samples:
            q0 = coeff * q1 - q2 + float(sample)
            q2, q1 = q1, q0

        return q1**2 + q2**2 - q1*q2*coeff


class RingbackPatternMatcher:
    """Pattern matcher for different regional ringback patterns"""

    def __init__(self):
        # Regional patterns (on_time, off_time in seconds)
        self.patterns = {
            'us_canada': {'on': 2.0, 'off': 4.0, 'frequencies': [440.0, 480.0]},
            'uk': {'on': 0.4, 'off': 0.2, 'frequencies': [400.0, 450.0]},  # Simplified
            'europe': {'on': 1.0, 'off': 4.0, 'frequencies': [425.0]},
            'australia': {'on': 0.4, 'off': 0.2, 'frequencies': [400.0, 425.0]}
        }

        # Use US/Canada pattern by default
        self.current_pattern = self.patterns['us_canada']

    def is_ringing_pattern(self, detections: list, timestamps: list) -> bool:
        """Check if detection pattern matches ringback timing"""
        if len(detections) < 10:  # Need minimum data
            return False

        # Find on/off transitions
        transitions = []
        current_state = detections[0]

        for i, detection in enumerate(detections[1:], 1):
            if detection != current_state:
                transitions.append((timestamps[i], detection))
                current_state = detection

        if len(transitions) < 4:  # Need at least 2 complete cycles
            return False

        # Check timing patterns
        on_times = []
        off_times = []

        for i in range(0, len(transitions) - 1, 2):
            if i + 1 < len(transitions):
                if transitions[i][1] == 1:  # Start of ON period
                    on_duration = transitions[i + 1][0] - transitions[i][0]
                    on_times.append(on_duration)
                if i + 2 < len(transitions):
                    off_duration = transitions[i + 2][0] - transitions[i + 1][0]
                    off_times.append(off_duration)

        # Check if timing matches expected pattern (with tolerance)
        expected_on = self.current_pattern['on']
        expected_off = self.current_pattern['off']
        tolerance = 0.8  # 800ms tolerance for more flexibility

        valid_on = sum(1 for t in on_times if abs(t - expected_on) < tolerance)
        valid_off = sum(1 for t in off_times if abs(t - expected_off) < tolerance)

        # At least 60% of timings should match (more permissive for real-world conditions)
        on_match_rate = valid_on / len(on_times) if on_times else 0
        off_match_rate = valid_off / len(off_times) if off_times else 0

        # Need at least 2 complete cycles and reasonable match rates
        has_enough_cycles = len(on_times) >= 2 and len(off_times) >= 1
        pattern_matches = on_match_rate >= 0.6 and off_match_rate >= 0.6

        return has_enough_cycles and pattern_matches


class RingCycleTracker:
    """
    Track ring cycles and detect when required number of rings completed
    A complete ring cycle = ON phase + OFF phase
    """

    def __init__(self, required_rings: int = 2):
        self.required_rings = max(1, min(required_rings, 10))  # Clamp 1-10

        # Ring tracking state
        self.ring_cycles = []
        self.current_ring_state = False
        self.last_state_change = None
        self.completed_rings = 0

    def update(self, is_ringing: bool, current_time: float) -> bool:
        """
        Update state with current ringing detection
        Returns True if required number of rings completed
        """
        # Check for state change
        if is_ringing != self.current_ring_state:
            if self.last_state_change is not None:
                # Calculate duration of previous state
                duration = current_time - self.last_state_change

                # Record the cycle transition
                cycle_info = {
                    'from_state': 'ringing' if self.current_ring_state else 'silence',
                    'to_state': 'ringing' if is_ringing else 'silence',
                    'duration': duration,
                    'timestamp': current_time
                }
                self.ring_cycles.append(cycle_info)

                # Check if we completed a ring (ringing -> silence transition)
                if self.current_ring_state and not is_ringing:
                    self.completed_rings += 1

                    # Check if we have enough rings
                    if self.completed_rings >= self.required_rings:
                        return True

            # Update state
            self.current_ring_state = is_ringing
            self.last_state_change = current_time

        return False

    def reset(self):
        """Reset tracker state"""
        self.ring_cycles.clear()
        self.current_ring_state = False
        self.last_state_change = None
        self.completed_rings = 0

    def get_stats(self):
        """Get current tracking statistics"""
        return {
            'completed_rings': self.completed_rings,
            'required_rings': self.required_rings,
            'cycle_count': len(self.ring_cycles),
            'current_state': 'ringing' if self.current_ring_state else 'silence'
        }


class DetectionValidator:
    """
    Validates ringing detection to filter false positives
    """

    def __init__(self,
                 relative_threshold: float = 3.0,
                 max_strength_threshold: float = 100.0,
                 frequency_balance_ratio: float = 6.5,
                 min_energy: float = 1e5,
                 required_consecutive: int = 2):
        self.relative_threshold = relative_threshold
        self.max_strength_threshold = max_strength_threshold
        self.frequency_balance_ratio = frequency_balance_ratio
        self.min_energy = min_energy
        self.required_consecutive = required_consecutive

        # State
        self.consecutive_detections = 0
        self.high_strength_buffer = []
        self.high_strength_window = 5

    def validate(self, energy_440: float, energy_480: float,
                 relative_strength: float, audio_array: Optional[np.ndarray] = None) -> bool:
        """
        Enhanced validation to filter false positives
        Returns True if detection is valid
        """
        # Check 1: Basic threshold
        if relative_strength <= self.relative_threshold:
            self.consecutive_detections = 0
            return False

        # Check 2: Pattern-based validation for extremely high strength
        if relative_strength > self.max_strength_threshold:
            # Track high-strength detections in buffer
            self.high_strength_buffer.append(relative_strength)
            if len(self.high_strength_buffer) > self.high_strength_window:
                self.high_strength_buffer.pop(0)

            # If we have sustained high strength (3+ out of last 5 chunks), it's likely real ringing
            high_count = sum(1 for s in self.high_strength_buffer if s > self.max_strength_threshold)
            if high_count < 3 and len(self.high_strength_buffer) >= 3:
                self.consecutive_detections = 0
                return False
        else:
            # Normal strength, clear high strength buffer
            self.high_strength_buffer.clear()

        # Check 3: Frequency balance
        freq_ratio = max(energy_440, energy_480) / (min(energy_440, energy_480) + 1e-9)
        if freq_ratio > self.frequency_balance_ratio:
            self.consecutive_detections = 0
            return False

        # Check 4: Minimum energy threshold
        min_target = min(energy_440, energy_480)
        if min_target < self.min_energy:
            self.consecutive_detections = 0
            return False

        # Check 5: Consecutive detection requirement
        self.consecutive_detections += 1
        if self.consecutive_detections < self.required_consecutive:
            return False

        # All checks passed
        return True

    def reset(self):
        """Reset validator state"""
        self.consecutive_detections = 0
        self.high_strength_buffer.clear()
