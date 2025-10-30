#!/usr/bin/env python3
"""
Centralized Database Configuration
All operations now use centralized server for agents and call logging
"""

from config.centralized_db_config import CENTRALIZED_DB_CONFIG

# Use centralized database for all operations
DB_CONFIG = CENTRALIZED_DB_CONFIG