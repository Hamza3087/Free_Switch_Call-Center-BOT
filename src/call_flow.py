# src/call_flow.py

#!/usr/bin/env python3
"""
Call Flow Utilities for SuiteCRM Integration
- Parses call flow from a JSON string from the database.
- Constructs audio file paths based on agent-specific directories.
"""

import json
import os
from typing import Optional, Dict, Any

def parse_call_flow_from_string(script_content: str) -> Optional[Dict[str, Any]]:
    """
    Parses a call flow from a JSON string provided by SuiteCRM.
    
    Args:
        script_content: The JSON string from the script's description field.

    Returns:
        A dictionary representing the call flow, or None if parsing fails.
    """
    if not script_content:
        print("[Flow] Error: Script content is empty.")
        return None
    try:
        flow_data = json.loads(script_content)
        # Basic validation to ensure it's a valid flow structure
        if 'name' in flow_data and 'steps' in flow_data:
            # Check for valid entry points (more flexible than requiring 'start')
            valid_entry_points = ['start', 'hello', 'introduction', 'introduction_and_main_question']
            has_entry_point = any(entry in flow_data['steps'] for entry in valid_entry_points)
            
            if has_entry_point:
                print(f"[Flow] Loaded call flow from string: {flow_data['name']}")
                return flow_data
            else:
                print(f"[Flow] Error: No valid entry point found. Expected one of: {valid_entry_points}")
                return None
        else:
            print("[Flow] Error: Parsed JSON is missing required keys ('name', 'steps').")
            return None
    except json.JSONDecodeError as e:
        print(f"[Flow] Error parsing call flow JSON string: {e}")
        return None

def get_audio_path_for_agent(audio_file: str, voice_location: str, 
                            greetings: bool = False, us_states: bool = False,
                            time_period: str = None, state_code: str = None) -> Optional[str]:
    """
    Constructs the full path to an agent's specific audio file with fallback logic.
    
    Tries files in this order:
    1. Full modified name (state + time prefixes)
    2. State prefix only
    3. Time prefix only  
    4. Base file name

    Args:
        audio_file: The name of the audio file (e.g., "start.wav").
        voice_location: The base directory for the agent's voice files, provided by SuiteCRM.
                        (e.g., /var/www/html/public/legacy/custom/audio_files/AGENT_ID)
        greetings: Whether to try time-based prefix (gm_, ga_, ge_)
        us_states: Whether to try state-based prefix (NY_, CA_, etc.)
        time_period: Time period (morning, afternoon, evening, night) for greetings
        state_code: Two-letter state code for us_states

    Returns:
        The full, absolute path to the audio file, or None if inputs are invalid.
    """
    if not audio_file or not voice_location:
        return None
    
    # If no special flags are set, use the original behavior
    if not greetings and not us_states:
        return os.path.join(voice_location, audio_file)
    
    # List of file names to try in order of preference
    files_to_try = []
    
    # 1. Try full modified name (both state and time prefixes if applicable)
    if greetings and us_states and time_period and state_code:
        full_modified = get_dynamic_audio_file_name(audio_file, True, True, time_period, state_code)
        files_to_try.append(full_modified)
    
    # 2. Try state prefix only
    if us_states and state_code:
        state_only = get_dynamic_audio_file_name(audio_file, False, True, None, state_code)
        files_to_try.append(state_only)
    
    # 3. Try time prefix only
    if greetings and time_period:
        time_only = get_dynamic_audio_file_name(audio_file, True, False, time_period, None)
        files_to_try.append(time_only)
    
    # 4. Always include the base file as fallback
    files_to_try.append(audio_file)
    
    # Remove duplicates while preserving order
    unique_files = []
    for file in files_to_try:
        if file not in unique_files:
            unique_files.append(file)
    
    # Try each file in order until we find one that exists
    for filename in unique_files:
        file_path = os.path.join(voice_location, filename)
        if os.path.exists(file_path):
            return file_path
    
    # If no files exist, return the path to the base file anyway
    # The calling code will handle the file not found error
    return os.path.join(voice_location, audio_file)

def get_special_audio_path_for_agent(audio_file: str, voice_location: str) -> Optional[str]:
    """
    Constructs the full path for special audio files (e.g., "silence.wav").
    For this architecture, it's assumed they are in the same agent-specific directory.

    Args:
        audio_file: The name of the special audio file (e.g., "silence.wav").
        voice_location: The agent's voice file directory.

    Returns:
        The full, absolute path to the special audio file.
    """
    return get_audio_path_for_agent(audio_file, voice_location)

def get_dynamic_audio_file_name(audio_file: str, greetings: bool = False, us_states: bool = False, 
                                time_period: str = None, state_code: str = None) -> str:
    """
    Constructs the dynamic audio file name based on greetings and us_states flags.
    
    Args:
        audio_file: The base audio file name (e.g., "medicare_introduction.wav")
        greetings: Whether to add time-based prefix (gm_, ga_, ge_)
        us_states: Whether to add state-based prefix (NY_, CA_, etc.)
        time_period: Time period (morning, afternoon, evening, night) for greetings
        state_code: Two-letter state code for us_states
        
    Returns:
        Modified audio file name with appropriate prefixes
    """
    if not audio_file:
        return audio_file
    
    # Start with the base file name
    modified_name = audio_file
    
    # Add state prefix if us_states flag is set and state_code is provided
    if us_states and state_code:
        modified_name = f"{state_code}_{modified_name}"
    
    # Add time prefix if greetings flag is set and time_period is provided
    if greetings and time_period:
        # Map time period to prefix
        time_prefixes = {
            'morning': 'gm_',
            'afternoon': 'ga_', 
            'evening': 'ge_',
            'night': ''  # No prefix for night calls
        }
        prefix = time_prefixes.get(time_period, '')  # Default to no prefix
        if prefix:  # Only add prefix if not empty
            modified_name = f"{prefix}{modified_name}"
    
    return modified_name

def validate_voice_selection():
    """
    Dummy function to satisfy import in bot_manager.py.
    The actual validation happens implicitly by checking file paths at runtime.
    """
    print("[Voice] Validation for fallback voice is handled at runtime.")
    return True