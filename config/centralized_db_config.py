#!/usr/bin/env python3
"""
Centralized Database Configuration
Client bot database credentials for centralized logging
"""

CENTRALIZED_DB_CONFIG = {
    'user': 'bots_portal_client',
    'password': 'Pineapple@321',
    'host': '138.199.162.238',
    'port': 3306,
    'database': 'bots_portal',
    'raise_on_warnings': True
}

# Server information API endpoint
SERVER_INFO_API_URL = 'http://138.199.162.238/index.php?entryPoint=server_information'