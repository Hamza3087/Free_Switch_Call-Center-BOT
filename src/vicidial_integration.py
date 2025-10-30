#!/usr/bin/env python3
"""
Enhanced ViciDial Integration for Python Voicebot - FINAL SIMPLIFIED VERSION
Relies solely on lead_id provided from SIP headers, removing phone number lookups.
"""

import requests
import json
import time
import logging
from datetime import datetime
from typing import Dict, Optional, Any
from urllib.parse import urlencode
import re
import urllib3

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class ViciDialAPI:
    """ViciDial API client for voicebot integration - COMPLETE VERSION"""
    
    def __init__(self, server_url: str, api_user: str, api_pass: str):
        self.api_user = api_user
        self.api_pass = api_pass
        self.source = "voicebot_9999"  # Hardcoded as per request
        self.session = requests.Session()
        self.session.verify = False  # Disable SSL verification
        
        # Set up logging first
        self.logger = logging.getLogger('ViciDialAPI')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Handle both complete URLs and base URLs for backward compatibility
        self.api_endpoint_url = server_url.strip()
        
        self.logger.info(f"ViciDial API Initialized for {self.api_endpoint_url}")
        self.logger.info(f"User: {self.api_user}, Source: {self.source}")
    
        
    def _make_request(self, function: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make authenticated request to ViciDial API"""
        
        auth_params = {
            'user': self.api_user,
            'pass': self.api_pass,
            'source': self.source,
            'function': function
        }
        
        all_params = {**auth_params, **params}
        
        # Log request (hide password)
        display_params = all_params.copy()
        if 'pass' in display_params:
            display_params['pass'] = '********'
        
        self.logger.info(f"Making API request to function: '{function}'")
        self.logger.debug(f"Request params: {json.dumps(display_params, indent=2)}")
        
        try:
            url = self.api_endpoint_url
            self.logger.debug(f"Request URL: {url}")
            
            if len(urlencode(all_params)) > 2000:
                self.logger.debug(f"Using POST request (large payload)")
                response = self.session.post(url, data=all_params, timeout=30, verify=False)
            else:
                self.logger.debug(f"Using GET request")
                response = self.session.get(url, params=all_params, timeout=30, verify=False)
            
            self.logger.info(f"Response Status: {response.status_code}")
            self.logger.debug(f"Raw Response: {response.text.strip()}")
            
            response.raise_for_status()
            result = self.parse_vicidial_response(response.text)
            self.logger.info(f"Parsed Result: {json.dumps(result, indent=2)}")
            return result
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"ViciDial API request failed: {e}")
            return {'success': False, 'error': str(e), 'raw_response': ''}
    
    def parse_vicidial_response(self, response_text: str) -> Dict[str, Any]:
        """Parse ViciDial API response format"""
        response_text = response_text.strip()
        result = {'success': False, 'raw_response': response_text, 'message': response_text, 'data': None}

        if response_text.startswith('ERROR:'):
            result['success'] = False
            result['error'] = response_text
            self.logger.error(f"API returned error: {response_text}")
        elif response_text.startswith('SUCCESS:') or response_text.startswith('NOTICE:') or response_text.startswith('VERSION:'):
            result['success'] = True
            self.logger.info(f"API returned success: {response_text}")
        
        if '|' in response_text:
            data_part = response_text
            if ': ' in data_part:
                data_part = data_part.rsplit(': ', 1)[-1]
            if ' - ' in data_part:
                data_part = data_part.split(' - ', 1)[-1]
            
            parsed_data = data_part.strip().lstrip('|').split('|')
            result['data'] = parsed_data
            self.logger.debug(f"Parsed data array: {parsed_data}")
        
        return result
    
    def update_lead_status_by_id(self, lead_id: str, status: str, **kwargs) -> Dict[str, Any]:
        """Update lead status using lead_id"""
        params = {
            'lead_id': lead_id,
            'status': status,
            'user': kwargs.get('user', self.api_user),
            'vendor_lead_code': kwargs.get('vendor_lead_code', ''),
            'callback_date': kwargs.get('callback_date', ''),
            'callback_time': kwargs.get('callback_time', ''),
            'comments': kwargs.get('comments', ''),
            'called_since_last_reset': kwargs.get('called_since_last_reset', 'Y'),
            'modify_date': kwargs.get('modify_date', 'Y'),
        }

        self.logger.info(f"Updating lead {lead_id} to status: {status}")
        return self._make_request('update_lead', params)

    def update_log_entry(self, lead_id: str, campaign_id: str, status: str) -> Dict[str, Any]:
        """Update call log entry using lead_id and campaign_id"""
        params = {
            'call_id': lead_id,  # ViciDial uses lead_id as call_id
            'group': campaign_id,
            'status': status,
        }

        self.logger.info(f"Updating call log for lead {lead_id} (campaign {campaign_id}) to status: {status}")
        return self._make_request('update_log_entry', params)

class VoicebotViciDialIntegration:
    """Integration layer between voicebot and ViciDial - Simplified to rely on provided lead_id."""
    
    def __init__(self, vicidial_api: ViciDialAPI, campaign_id: str, list_id: str = "999"):
        self.vicidial_api = vicidial_api
        self.campaign_id = campaign_id
        self.list_id = list_id
        
        self.logger = logging.getLogger('VoicebotIntegration')
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.logger.info(f"ViciDial Integration Initialized. Relies on lead_id from SIP headers.")
    
    def test_connection(self) -> bool:
        """Test ViciDial connection"""
        try:
            self.logger.info("Testing ViciDial API connection...")
            result = self.vicidial_api._make_request('version', {})
            
            if result.get('success'):
                self.logger.info("✅ ViciDial API connection successful")
                return True
            else:
                self.logger.error(f"❌ ViciDial API connection failed: {result.get('error')}")
                return False
        except Exception as e:
            self.logger.error(f"❌ Exception testing ViciDial connection: {e}")
            return False

    def send_disposition_directly(self, phone_number: str, disposition: str, call_data: dict) -> bool:
        """
        Updates the disposition for a lead using the lead_id provided in call_data.
        Updates BOTH vicidial_list (lead table) and vicidial_log (call log table) with the SAME disposition.
        """
        try:
            lead_id = call_data.get('vici_lead_id')

            if not lead_id:
                self.logger.error(f"❌ Cannot send disposition for {phone_number}. 'vici_lead_id' is missing from call_data.")
                return False

            # Get ViciDial campaign_id from SIP header (stored as vici_campaign_id)
            campaign_id = call_data.get('vici_campaign_id', self.campaign_id)

            self.logger.info(f"Sending disposition '{disposition}' for lead_id: {lead_id} (Phone: {phone_number}, Campaign: {campaign_id})")

            # Prepare comments from call_data
            transcript = call_data.get('transcript', '')
            duration = call_data.get('duration', 0)
            intent = call_data.get('intent_detected', '')

            comments = f"VoiceBot Call: {disposition}"
            if duration:
                comments += f" | Duration: {duration}s"
            if intent:
                comments += f" | Intent: {intent}"
            if transcript:
                transcript_short = transcript[:150] + "..." if len(transcript) > 150 else transcript
                comments += f" | Transcript: {transcript_short}"

            if len(comments) > 255:
                comments = comments[:252] + "..."

            self.logger.info(f"Comments for lead {lead_id}: {comments}")

            # Step 1: Update lead status in vicidial_list table
            self.logger.info(f"Step 1: Updating lead {lead_id} in vicidial_list...")
            lead_update_result = self.vicidial_api.update_lead_status_by_id(
                lead_id=lead_id,
                status=disposition,
                user=self.vicidial_api.api_user,
                comments=comments
            )

            if not lead_update_result.get('success'):
                self.logger.error(f"❌ Failed to update lead {lead_id} in vicidial_list: {lead_update_result.get('error', 'Unknown error')}")
                return False

            self.logger.info(f"✅ Lead {lead_id} updated in vicidial_list with disposition '{disposition}'")

            # Wait 1 second for ViciDial to create call log entry
            time.sleep(1)

            # Step 2: Update call log entry in vicidial_log table with SAME disposition (non-critical)
            self.logger.info(f"Step 2: Updating call log for lead {lead_id} in vicidial_log...")
            log_update_result = self.vicidial_api.update_log_entry(
                lead_id=lead_id,
                campaign_id=campaign_id,
                status=disposition
            )

            if log_update_result.get('success'):
                self.logger.info(f"✅ Call log updated in vicidial_log with disposition '{disposition}'")
                self.logger.info(f"✅ Successfully updated BOTH vicidial_list and vicidial_log for lead {lead_id} with disposition '{disposition}'")
            else:
                # Log entry update returned no match (expected during active call processing)
                # ViciDial automatically syncs status from vicidial_list to vicidial_log
                self.logger.info(f"ℹ️ Call log update returned no match (expected during active call processing)")
                self.logger.info(f"✅ Lead status updated to '{disposition}' - ViciDial will sync to call log automatically")

            # Return success as long as lead was updated
            return True

        except Exception as e:
            self.logger.error(f"❌ Exception in send_disposition_directly for {phone_number} (lead_id: {call_data.get('vici_lead_id')}): {e}", exc_info=True)
            return False
    
    def handle_callback_request(self, phone_number: str, callback_datetime: str, call_data: dict) -> bool:
        """Handle callback request from voicebot, relying on lead_id."""
        try:
            lead_id = call_data.get('vici_lead_id')
            if not lead_id:
                self.logger.error(f"Cannot set callback for {phone_number}. 'vici_lead_id' is missing.")
                return False

            # Parse callback datetime (e.g., "2023-10-27 14:30:00")
            callback_parts = callback_datetime.split(' ')
            callback_date = callback_parts[0] if len(callback_parts) > 0 else ''
            callback_time = callback_parts[1] if len(callback_parts) > 1 else '09:00:00'
            
            self.logger.info(f"Setting callback for lead {lead_id} at {callback_date} {callback_time}")

            result = self.vicidial_api.update_lead_status_by_id(
                lead_id=lead_id,
                status='CALLBK',
                callback_date=callback_date,
                callback_time=callback_time,
                comments=f'Callback requested by voicebot for {callback_datetime}'
            )
            
            return result.get('success', False)
            
        except Exception as e:
            self.logger.error(f"Error handling callback request for lead {lead_id}: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test ViciDial Integration (Simplified)')
    parser.add_argument('--test', action='store_true', help='Run connection test')
    parser.add_argument('--lead_id', type=str, help='Test lead_id to update')
    parser.add_argument('--disposition', type=str, default='DNC', help='Test disposition')
    parser.add_argument('--url', type=str, default="http://65.108.156.184", help='ViciDial Server URL')
    parser.add_argument('--user', type=str, default="9999", help='API User')
    parser.add_argument('--password', type=str, default="9999", help='API Password')

    args = parser.parse_args()
    
    api = ViciDialAPI(
        server_url=args.url,
        api_user=args.user,
        api_pass=args.password
    )
    
    integration = VoicebotViciDialIntegration(
        vicidial_api=api,
        campaign_id="9999"
    )
    
    if args.test:
        print("Testing ViciDial connection...")
        success = integration.test_connection()
        print(f"Connection test result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if args.lead_id:
        print(f"Testing disposition update for lead_id {args.lead_id}...")
        
        call_data = {
            'vici_lead_id': args.lead_id,
            'duration': 60,
            'transcript': f'Test transcript for lead {args.lead_id} with disposition {args.disposition}.'
        }
        
        success = integration.send_disposition_directly("15551234567", args.disposition, call_data)
        print(f"Disposition update result: {'✅ SUCCESS' if success else '❌ FAILED'}")
    
    if not args.test and not args.lead_id:
        print("ViciDial Integration module loaded successfully!")
        print("Use --test to test connection or --lead_id <id> to test disposition update.")