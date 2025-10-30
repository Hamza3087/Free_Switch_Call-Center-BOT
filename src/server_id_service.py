#!/usr/bin/env python3
"""
Server ID Retrieval Service
Handles communication with centralized API to get server ID based on IP address
"""

import requests
import logging
import subprocess
import threading
import time
from typing import Optional
from config.centralized_db_config import SERVER_INFO_API_URL

class ServerIDService:
    """Service to retrieve and cache server ID from centralized API"""
    
    _instance = None
    _lock = threading.Lock()
    _server_id = None
    _server_ip = None
    _logger = logging.getLogger(__name__)
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ServerIDService, cls).__new__(cls)
        return cls._instance
    
    def get_server_ip(self) -> Optional[str]:
        """Get the current server's public IP address"""
        if self._server_ip is not None:
            return self._server_ip
            
        try:
            # Try multiple methods to get public IP
            methods = [
                ['curl', '-4', '--connect-timeout', '5', 'ifconfig.me'],
                ['curl', '-4', '--connect-timeout', '5', 'ipinfo.io/ip'],
                ['curl', '-4', '--connect-timeout', '5', 'api.ipify.org']
            ]
            
            for method in methods:
                try:
                    result = subprocess.run(
                        method,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        ip = result.stdout.strip()
                        if self._is_valid_ip(ip):
                            self._server_ip = ip
                            self._logger.info(f"✅ Server IP detected: {ip}")
                            return ip
                except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                    continue
                    
            self._logger.error("❌ Failed to detect server IP address")
            return None
            
        except Exception as e:
            self._logger.error(f"Error detecting server IP: {e}")
            return None
    
    def _is_valid_ip(self, ip: str) -> bool:
        """Basic IP address validation"""
        try:
            parts = ip.split('.')
            return (
                len(parts) == 4 and 
                all(0 <= int(part) <= 255 for part in parts) and
                not ip.startswith('127.') and
                not ip.startswith('192.168.') and
                not ip.startswith('10.') and
                not ip.startswith('169.254.')
            )
        except (ValueError, AttributeError):
            return False
    
    def retrieve_server_id(self, max_retries: int = 3) -> Optional[str]:
        """
        Retrieve server ID from centralized API
        Returns cached ID if already retrieved
        """
        if self._server_id is not None:
            return self._server_id
        
        server_ip = self.get_server_ip()
        if not server_ip:
            self._logger.error("Cannot retrieve server ID: No valid IP address")
            return None
        
        for attempt in range(max_retries):
            try:
                self._logger.info(f"Retrieving server ID for IP {server_ip} (attempt {attempt + 1})")
                
                # Prepare POST request
                payload = {'ip_address': server_ip}
                headers = {
                    'Content-Type': 'application/x-www-form-urlencoded',
                    'User-Agent': 'SIP-Bot-Server/1.0'
                }
                
                # Make API request
                response = requests.post(
                    SERVER_INFO_API_URL,
                    data=payload,
                    headers=headers,
                    timeout=10
                )
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        if data.get('status') == 'success':
                            server_data = data.get('data', {}).get('server', {})
                            server_id = server_data.get('id')
                            
                            if server_id:
                                self._server_id = server_id
                                self._logger.info(f"✅ Server ID retrieved: {server_id}")
                                
                                # Log additional server info
                                auto_start = server_data.get('auto_start_enabled', 'unknown')
                                self._logger.info(f"Server info - IP: {server_ip}, Auto-start: {auto_start}")
                                
                                return server_id
                            else:
                                self._logger.warning("Server ID not found in API response")
                        else:
                            self._logger.warning(f"API returned status: {data.get('status')}")
                            
                    except ValueError as e:
                        self._logger.error(f"Invalid JSON response: {e}")
                        self._logger.debug(f"Response content: {response.text[:500]}")
                        
                else:
                    self._logger.warning(f"API request failed with status {response.status_code}")
                    self._logger.debug(f"Response: {response.text[:200]}")
                    
            except requests.exceptions.Timeout:
                self._logger.warning(f"API request timeout (attempt {attempt + 1})")
                
            except requests.exceptions.ConnectionError as e:
                self._logger.warning(f"Connection error (attempt {attempt + 1}): {e}")
                
            except Exception as e:
                self._logger.error(f"Unexpected error retrieving server ID: {e}")
            
            # Wait before retry
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                self._logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        self._logger.error("❌ Failed to retrieve server ID after all attempts")
        return None
    
    def get_server_id(self) -> Optional[str]:
        """Get cached server ID or retrieve if not cached"""
        if self._server_id is None:
            return self.retrieve_server_id()
        return self._server_id
    
    def reset_cache(self):
        """Reset cached server ID and IP (for testing)"""
        with self._lock:
            self._server_id = None
            self._server_ip = None
            self._logger.info("Server ID cache reset")

# Global instance
server_id_service = ServerIDService()

def get_server_id() -> Optional[str]:
    """Convenience function to get server ID"""
    return server_id_service.get_server_id()

def get_server_ip() -> Optional[str]:
    """Convenience function to get server IP"""
    return server_id_service.get_server_ip()