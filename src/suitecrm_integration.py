#!/usr/bin/env python3
"""
Production-Grade SuiteCRM Integration Module
- Thread-safe database operations with connection pooling
- Automatic reconnection on failures
- Comprehensive error handling
"""

import mysql.connector
from mysql.connector import pooling
import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
import threading
import time
import queue
import os
from contextlib import contextmanager

# Import DB configurations and server ID service
from config.suitecrm_config import DB_CONFIG
from config.centralized_db_config import CENTRALIZED_DB_CONFIG
from src.server_id_service import get_server_id

class SuiteCRMAgentConfig:
    """Data class for virtual agent configuration"""
    def __init__(self, agent_id: str, agent_name: str, voice_location: str, noise_location: str,
                 script_content: str, campaign_id: str, vicidial_campaign_id: str, server_ip: str,
                 server_url: str, username: str, password: str, enable_voicemail_detection: str,
                 background_noise_volume: float, max_silence_retries: int, max_clarification_retries: int,
                 did_transfer_qualified: str, did_transfer_hangup: str, honey_pot_sentences: List[str],
                 energy_threshold: str = None, rnnt_confidence_threshold: str = None, white_list_command: str = None):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.voice_location = voice_location
        self.noise_location = noise_location
        self.script_content = script_content
        self.campaign_id = campaign_id
        self.vicidial_campaign_id = vicidial_campaign_id
        self.server_ip = server_ip

        self.server_url = server_url
        self.username = username
        self.password = password
        self.enable_voicemail_detection = enable_voicemail_detection
        self.background_noise_volume = float(background_noise_volume) if background_noise_volume is not None else 1.0
        self.max_silence_retries = int(max_silence_retries) if max_silence_retries is not None else 2
        self.max_clarification_retries = int(max_clarification_retries) if max_clarification_retries is not None else 2
        self.did_transfer_qualified = str(did_transfer_qualified) if did_transfer_qualified is not None else '9998'
        self.did_transfer_hangup = str(did_transfer_hangup) if did_transfer_hangup is not None else '9997'
        self.honey_pot_sentences = honey_pot_sentences if honey_pot_sentences is not None else []
        self.white_list_command = white_list_command  # Command to execute at startup (optional)

        # Parse threshold values with defaults
        try:
            self.energy_threshold = float(energy_threshold) if energy_threshold is not None else 0.045
        except (ValueError, TypeError):
            self.energy_threshold = 0.045

        try:
            self.rnnt_confidence_threshold = float(rnnt_confidence_threshold) if rnnt_confidence_threshold is not None else 0.5
        except (ValueError, TypeError):
            self.rnnt_confidence_threshold = 0.5

        self.noise_volume = self.background_noise_volume

class SuiteCRMDBManager:
    """Production-grade database manager with connection pooling"""
    
    _pool = None
    _pool_lock = threading.Lock()
    _pool_name = "suitecrm_pool"
    _pool_size = 32  # MySQL connector maximum allowed
    _max_overflow = 0  # Overflow not supported by MySQL connector
    _logger = logging.getLogger(__name__)
    
    @classmethod
    def _create_pool(cls):
        """Create connection pool"""
        try:
            if cls._pool is None:
                pool_config = DB_CONFIG.copy()
                pool_config['pool_name'] = cls._pool_name
                pool_config['pool_size'] = cls._pool_size
                pool_config['pool_reset_session'] = True
                
                cls._pool = pooling.MySQLConnectionPool(**pool_config)
                cls._logger.info(f"✅ Database connection pool created (size={cls._pool_size})")
        except mysql.connector.Error as err:
            cls._logger.error(f"❌ Failed to create connection pool: {err}")
            cls._pool = None
            raise
    
    @classmethod
    @contextmanager
    def get_connection(cls):
        """Get connection from pool with automatic management"""
        connection = None
        retry_count = 3
        retry_delay = 1
        
        for attempt in range(retry_count):
            try:
                with cls._pool_lock:
                    if cls._pool is None:
                        cls._create_pool()
                
                connection = cls._pool.get_connection()
                
                # Test connection
                connection.ping(reconnect=True, attempts=3, delay=1)
                
                yield connection
                
                # Commit any pending transactions
                if connection.in_transaction:
                    connection.commit()
                
                return
                
            except mysql.connector.PoolError as err:
                cls._logger.warning(f"Pool error (attempt {attempt + 1}): {err}")
                
                if attempt < retry_count - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    
                    # Try to recreate pool
                    with cls._pool_lock:
                        cls._pool = None
                else:
                    raise
                    
            except mysql.connector.Error as err:
                cls._logger.error(f"Database error: {err}")
                
                if connection and connection.is_connected():
                    connection.rollback()
                
                if attempt < retry_count - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    raise
                    
            finally:
                # Always return connection to pool
                if connection and connection.is_connected():
                    try:
                        connection.close()
                    except:
                        pass
    
    @classmethod
    def execute_query(cls, query: str, params: tuple = None, fetch: str = None, 
                     retry: bool = True, context: str = None, logger: logging.Logger = None) -> Any:
        """
        Execute query with automatic retry and error handling
        Now includes query latency tracking for remote database monitoring
        
        Args:
            query: SQL query
            params: Query parameters
            fetch: 'one', 'all', or None
            retry: Whether to retry on failure
            context: Optional context for query identification in logs
            logger: Optional logger to use instead of module logger (for call record integration)
        """
        max_retries = 3 if retry else 1
        query_start_time = time.time()
        
        # Use provided logger or fall back to module logger
        active_logger = logger if logger else cls._logger
        
        # Extract query type and table for logging
        query_type = query.strip().split()[0].upper()
        query_preview = query.replace('\n', ' ').replace('\t', ' ')[:100] + "..."
        context_info = f"[{context.upper()}]" if context else "[AGENT_CONFIG]"
        
        for attempt in range(max_retries):
            attempt_start_time = time.time()
            try:
                with cls.get_connection() as conn:
                    cursor = conn.cursor(dictionary=True, buffered=True)
                    
                    try:
                        cursor.execute(query, params)
                        
                        result = None
                        row_count = 0
                        
                        if fetch == 'one':
                            result = cursor.fetchone()
                            row_count = 1 if result else 0
                        elif fetch == 'all':
                            result = cursor.fetchall()
                            row_count = len(result) if result else 0
                        else:
                            conn.commit()
                            result = cursor.lastrowid or cursor.rowcount
                            row_count = result if result else 0
                        
                        # Calculate and log latency
                        latency_ms = (time.time() - query_start_time) * 1000
                        
                        if latency_ms > 1000:  # Slow query warning
                            active_logger.warning(f"⚠️ {context_info} SLOW QUERY: {latency_ms:.2f}ms | {query_type} | Rows: {row_count}")
                        else:
                            active_logger.info(f"🔍 {context_info} Query: {latency_ms:.2f}ms | {query_type} | Rows: {row_count}")
                        
                        # Debug log for very detailed tracking
                        active_logger.debug(f"🔍 {context_info} Query details: {query_preview}")
                        
                        return result
                            
                    finally:
                        cursor.close()
                        
            except mysql.connector.Error as err:
                attempt_latency_ms = (time.time() - attempt_start_time) * 1000
                active_logger.error(f"❌ {context_info} Query error (attempt {attempt + 1}): {err} | {attempt_latency_ms:.2f}ms")
                
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                else:
                    total_latency_ms = (time.time() - query_start_time) * 1000
                    active_logger.error(f"❌ {context_info} Query failed after {max_retries} attempts | Total: {total_latency_ms:.2f}ms")
                    return None
                    
            except Exception as e:
                attempt_latency_ms = (time.time() - attempt_start_time) * 1000
                active_logger.error(f"❌ {context_info} Unexpected error (attempt {attempt + 1}): {e} | {attempt_latency_ms:.2f}ms", exc_info=True)
                if attempt >= max_retries - 1:
                    return None
        
        return None
    
    @classmethod
    def get_pool_stats(cls) -> Dict[str, Any]:
        """Get detailed connection pool statistics"""
        stats = {
            'pool_name': cls._pool_name,
            'pool_size': cls._pool_size,
            'max_overflow': cls._max_overflow,
            'available_connections': 0,
            'checked_out_connections': 0,
            'overflow_connections': 0,
            'total_capacity': cls._pool_size + cls._max_overflow,
            'utilization_percent': 0,
            'pool_healthy': False,
            'last_check': datetime.now().isoformat()
        }
        
        try:
            with cls._pool_lock:
                if cls._pool is not None:
                    # Get pool statistics (handle different pool implementations)
                    try:
                        if hasattr(cls._pool, '_pool') and hasattr(cls._pool._pool, 'qsize'):
                            stats['available_connections'] = cls._pool._pool.qsize()
                        else:
                            # Fallback for newer versions of mysql-connector-python
                            stats['available_connections'] = cls._pool_size  # Assume healthy for now
                    except:
                        stats['available_connections'] = cls._pool_size  # Safe fallback
                    
                    stats['checked_out_connections'] = cls._pool_size - stats['available_connections']
                    stats['overflow_connections'] = max(0, stats['checked_out_connections'] - cls._pool_size)
                    
                    # Calculate utilization
                    if stats['total_capacity'] > 0:
                        stats['utilization_percent'] = (stats['checked_out_connections'] / stats['total_capacity']) * 100
                    
                    # Health check
                    stats['pool_healthy'] = (
                        stats['utilization_percent'] < 90 and  # Less than 90% utilized
                        stats['available_connections'] > 0  # At least one connection available
                    )
                    
                    # Log warning if pool is heavily utilized
                    if stats['utilization_percent'] > 80:
                        cls._logger.warning(f"⚠️ SuiteCRM pool high utilization: {stats['utilization_percent']:.1f}% "
                                          f"({stats['checked_out_connections']}/{stats['total_capacity']} connections)")
                        
        except Exception as e:
            cls._logger.error(f"Error getting pool stats: {e}")
            stats['error'] = str(e)
        
        return stats
    
    @classmethod
    def health_check(cls) -> bool:
        """Perform health check on connection pool"""
        try:
            with cls._pool_lock:
                if cls._pool is None:
                    cls._logger.warning("SuiteCRM connection pool not initialized")
                    return False
            
            # Try to get a connection briefly
            try:
                with cls.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    cursor.close()
                    
                    if result:
                        cls._logger.debug("SuiteCRM pool health check passed")
                        return True
                    else:
                        cls._logger.error("SuiteCRM pool health check failed: No result")
                        return False
                        
            except Exception as e:
                cls._logger.error(f"SuiteCRM pool health check failed: {e}")
                return False
                
        except Exception as e:
            cls._logger.error(f"SuiteCRM pool health check error: {e}")
            return False
    
    @classmethod
    def monitor_pool_health(cls, alert_threshold: float = 85.0):
        """Monitor pool health and log alerts if needed"""
        try:
            stats = cls.get_pool_stats()
            
            # Alert on high utilization
            if stats['utilization_percent'] > alert_threshold:
                cls._logger.warning(f"🚨 SuiteCRM POOL ALERT: {stats['utilization_percent']:.1f}% utilization "
                                  f"({stats['checked_out_connections']}/{stats['total_capacity']} connections used)")
            
            # Alert on pool exhaustion
            if stats['available_connections'] == 0:
                cls._logger.error(f"🚨 SuiteCRM POOL EXHAUSTED: No available connections! "
                                f"All {stats['total_capacity']} connections in use")
            
            # Alert on unhealthy pool
            if not stats['pool_healthy']:
                cls._logger.error(f"🚨 SuiteCRM POOL UNHEALTHY: Health check failed")
            
            # Log stats at debug level
            cls._logger.debug(f"SuiteCRM Pool Stats: {stats['checked_out_connections']}/{stats['total_capacity']} "
                            f"used ({stats['utilization_percent']:.1f}%), {stats['available_connections']} available")
            
            return stats
            
        except Exception as e:
            cls._logger.error(f"Error monitoring SuiteCRM pool health: {e}")
            return None
    
    @classmethod
    def close_pool(cls):
        """Close connection pool"""
        with cls._pool_lock:
            if cls._pool:
                try:
                    # Close all connections in pool
                    cls._pool._remove_connections()
                    cls._pool = None
                    cls._logger.info("Database connection pool closed")
                except:
                    pass

def fetch_active_agent_configs() -> List[SuiteCRMAgentConfig]:
    """
    Fetch all active agent configurations with comprehensive error handling
    Now uses centralized database with dynamic server ID filtering
    """
    # Get server ID dynamically
    server_id = get_server_id()
    if not server_id:
        print("⚠️ Server ID not available - loading all agents without server filtering")
        server_filter = ""
        params = ()
    else:
        print(f"🔗 Loading agents for server ID: {server_id}")
        server_filter = "AND agent.sh_ep_servers_id = %s"
        params = (server_id,)
    
    query = f"""
    SELECT 
        agent.id as agent_id,
        agent.name as agent_name,
        campaigns.*,
        CONCAT('/var/www/audio_files/', agent.id) as voice_location,
        CONCAT('/var/www/noises/', noise.id) as noise_location,
        scripts.description as script_content
    FROM e_agent agent
    JOIN (
        SELECT c.e_agent_e_scriptse_agent_idb as agent_id, s.description 
        FROM e_agent_e_scripts_c c 
        JOIN e_scripts s ON c.e_agent_e_scriptse_scripts_ida = s.id AND s.deleted = 0 
        WHERE c.deleted = 0
    ) as scripts ON scripts.agent_id = agent.id
    LEFT JOIN (
        SELECT c.e_agent_e_noisee_agent_idb as agent_id, n.id
        FROM e_agent_e_noise_c c 
        JOIN e_noise n ON c.e_agent_e_noisee_noise_ida = n.id AND n.deleted = 0 
        WHERE c.deleted = 0
    ) as noise ON noise.agent_id = agent.id
    JOIN (
        SELECT
            camp_rel.e_campaigns_e_agente_agent_idb as agent_id,
            camp.id as suitecrm_campaign_id,
            camp.vicidialer_campaign_id as vicidial_campaign_id,
            camp.server_ip,
            camp.server_url,
            camp.username,
            camp.password,
            camp.enable_voicemail_detection,
            camp.background_noise_volume,
            camp.max_silence_retries,
            camp.max_clarification_retries,
            camp.did_transfer_qualified,
            camp.did_transfer_hangup,
            camp.honey_pot_sentences,
            camp.energy_threshold,
            camp.rnnt_confidence_threshold,
            camp.white_list_command
        FROM e_campaigns_e_agent_c camp_rel
        JOIN e_campaigns camp ON camp_rel.e_campaigns_e_agente_campaigns_ida = camp.id AND camp.deleted = 0
        WHERE camp_rel.deleted = 0
    ) as campaigns ON campaigns.agent_id = agent.id
    WHERE agent.status = 'Active' 
      AND agent.deleted = 0
      AND agent.voices_generated = 1
      {server_filter};
    """
    
    configs = []
    results = SuiteCRMDBManager.execute_query(query, params, fetch='all', context="fetch_agents")
    
    if results:
        for row in results:
            try:
                # Validate JSON script
                json.loads(row['script_content'])
                
                # Validate required fields
                if not all([row.get('suitecrm_campaign_id'), 
                          row.get('vicidial_campaign_id'), 
                          row.get('server_ip')]):
                    print(f"⚠️ Skipping agent {row.get('agent_id')} - missing required fields")
                    continue
                
                # Handle VMD setting
                vmd_setting_raw = row.get('enable_voicemail_detection')
                enable_vmd = 'Yes' if vmd_setting_raw == '^1^' else 'No'
                
                # Parse honeypot sentences - split by newlines and filter empty lines
                honeypot_raw = row.get('honey_pot_sentences')
                honeypot_phrases = []
                if honeypot_raw:
                    honeypot_phrases = [phrase.strip() for phrase in honeypot_raw.split('\n') if phrase.strip()]
                
                config = SuiteCRMAgentConfig(
                    agent_id=row['agent_id'],
                    agent_name=row['agent_name'],
                    voice_location=row['voice_location'],
                    noise_location=row.get('noise_location'),
                    script_content=row['script_content'],
                    campaign_id=row['suitecrm_campaign_id'],
                    vicidial_campaign_id=row['vicidial_campaign_id'],
                    server_ip=row['server_ip'],
                    server_url=row.get('server_url'),
                    username=row.get('username'),
                    password=row.get('password'),
                    enable_voicemail_detection=enable_vmd,
                    background_noise_volume=row.get('background_noise_volume'),
                    max_silence_retries=row.get('max_silence_retries'),
                    max_clarification_retries=row.get('max_clarification_retries'),
                    did_transfer_qualified=row.get('did_transfer_qualified'),
                    did_transfer_hangup=row.get('did_transfer_hangup'),
                    honey_pot_sentences=honeypot_phrases,
                    energy_threshold=row.get('energy_threshold'),
                    rnnt_confidence_threshold=row.get('rnnt_confidence_threshold'),
                    white_list_command=row.get('white_list_command')
                )
                
                configs.append(config)
                
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"⚠️ Skipping agent {row.get('agent_id')} due to invalid config: {e}")
    
    # Log unique campaign honeypot phrases
    campaign_phrases = {}
    for config in configs:
        if config.vicidial_campaign_id not in campaign_phrases:
            campaign_phrases[config.vicidial_campaign_id] = config.honey_pot_sentences
    
    for campaign_id, phrases in campaign_phrases.items():
        if phrases:
            print(f"🍯 Campaign {campaign_id} loaded {len(phrases)} honeypot phrases: {phrases}")
        else:
            print(f"🍯 Campaign {campaign_id} has no honeypot phrases")
    
    print(f"✅ Loaded {len(configs)} active agent configurations")
    return configs

class DatabaseLogger:
    """Custom logger wrapper that routes database logs through CallLogCollector"""
    
    def __init__(self, bot_instance):
        self.bot_instance = bot_instance
    
    def info(self, message: str):
        """Route info logs through _log_and_collect"""
        if hasattr(self.bot_instance, '_log_and_collect'):
            self.bot_instance._log_and_collect('info', message)
        else:
            # Fallback to standard logger if _log_and_collect not available
            self.bot_instance.logger.info(message)
    
    def warning(self, message: str):
        """Route warning logs through _log_and_collect"""
        if hasattr(self.bot_instance, '_log_and_collect'):
            self.bot_instance._log_and_collect('warning', message)
        else:
            self.bot_instance.logger.warning(message)
    
    def error(self, message: str):
        """Route error logs through _log_and_collect"""
        if hasattr(self.bot_instance, '_log_and_collect'):
            self.bot_instance._log_and_collect('error', message)
        else:
            self.bot_instance.logger.error(message)
    
    def debug(self, message: str):
        """Route debug logs through _log_and_collect"""
        if hasattr(self.bot_instance, '_log_and_collect'):
            self.bot_instance._log_and_collect('debug', message)
        else:
            self.bot_instance.logger.debug(message)

class AsyncCallLogger:
    """
    Asynchronous call logger that prevents database operations from blocking audio pipeline
    Uses background worker threads and queuing to ensure call flow is never delayed by database latency
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self, num_workers=5):
        self.num_workers = num_workers
        self.work_queue = queue.Queue(maxsize=1000)  # Prevent memory exhaustion
        self.workers = []
        self.shutdown_event = threading.Event()
        self.stats = {
            'queued_operations': 0,
            'completed_operations': 0,
            'failed_operations': 0,
            'queue_full_errors': 0
        }
        self.stats_lock = threading.Lock()
        self._logger = logging.getLogger(__name__)
        
        # Fallback file logging when database unavailable
        self.fallback_log_dir = "/var/log/sip-bot/database_fallback"
        os.makedirs(self.fallback_log_dir, exist_ok=True)
        
        self._start_workers()
    
    @classmethod
    def get_instance(cls):
        """Singleton pattern for global async logger"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _start_workers(self):
        """Start background worker threads"""
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"AsyncCallLogger-Worker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        self._logger.info(f"✅ Started {self.num_workers} async database workers")
    
    def _worker_loop(self):
        """Background worker thread loop"""
        while not self.shutdown_event.is_set():
            try:
                # Get work item with timeout
                work_item = self.work_queue.get(timeout=1.0)
                if work_item is None:  # Shutdown signal
                    break
                
                operation, args = work_item
                try:
                    if operation == 'call_start':
                        self._execute_call_start(*args)
                    elif operation == 'call_end':
                        self._execute_call_end(*args)
                    
                    with self.stats_lock:
                        self.stats['completed_operations'] += 1
                        
                except Exception as e:
                    with self.stats_lock:
                        self.stats['failed_operations'] += 1
                    self._logger.error(f"Database operation failed: {e}", exc_info=True)
                    
                    # Fallback to file logging
                    self._fallback_log(operation, args, str(e))
                
                finally:
                    self.work_queue.task_done()
                    
            except queue.Empty:
                continue  # Timeout, check shutdown event
            except Exception as e:
                self._logger.error(f"Worker loop error: {e}", exc_info=True)
    
    def _execute_call_start(self, call_id, call_data, agent_config, db_logger):
        """Execute call start database operation"""
        query = """
        INSERT INTO e_call (
            id, name, date_entered, date_modified, disposition, 
            vici_lead_id, vici_list_id, e_campaigns_id, e_agent_id, 
            start_date_time, state, sh_ep_servers_id, vicidial_unique_id
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        );
        """
        
        now_utc = datetime.utcnow()
        start_time_utc = datetime.utcfromtimestamp(call_data['start_time'])
        server_id = get_server_id()
        
        params = (
            call_id,
            f"{call_data['phone_number']}",
            now_utc.strftime('%Y-%m-%d %H:%M:%S'),
            now_utc.strftime('%Y-%m-%d %H:%M:%S'),
            'INITIATED',
            call_data.get('vici_lead_id'),
            call_data.get('vici_list_id'),
            agent_config.campaign_id,
            agent_config.agent_id,
            start_time_utc.strftime('%Y-%m-%d %H:%M:%S'),
            call_data.get('caller_state'),
            server_id,
            call_data.get('vicidial_unique_id')
        )
        
        result = CentralizedDBManager.execute_query(query, params, context="async_call_start", logger=db_logger)
        
        if result is not None:
            self._logger.debug(f"Async: Call {call_id} logged to centralized database")
        else:
            raise Exception(f"Failed to log call start for {call_data['phone_number']}")
    
    def _execute_call_end(self, call_data, db_logger):
        """Execute call end database operation"""
        call_id = call_data.get('id')
        if not call_id:
            raise Exception("Cannot log call end: No call ID")
        
        query = """
        UPDATE e_call 
        SET date_modified = %s, 
            disposition = %s, 
            description = %s, 
            end_date_time = %s,
            filename = %s,
            file_mime_type = %s,
            call_drop_step = %s,
            call_logs = %s,
            state = %s
        WHERE id = %s;
        """
        
        now_utc = datetime.utcnow()
        end_time_utc = None
        if call_data.get('end_time'):
            end_time_utc = datetime.utcfromtimestamp(call_data['end_time'])
        
        # Prepare call logs with transfer information
        call_logs = call_data.get('call_logs', '')
        if call_data.get('transfer_target') or call_data.get('transfer_status'):
            transfer_info = []
            if call_data.get('transfer_target'):
                transfer_info.append(f"Target: {call_data.get('transfer_target')}")
            if call_data.get('transfer_status'):
                transfer_info.append(f"Status: {call_data.get('transfer_status')}")
            if call_data.get('transfer_response_code'):
                transfer_info.append(f"Response: {call_data.get('transfer_response_code')}")
            if call_data.get('transfer_timestamp'):
                transfer_info.append(f"Time: {call_data.get('transfer_timestamp')}")
            if call_data.get('transfer_reason'):
                transfer_info.append(f"Reason: {call_data.get('transfer_reason')}")
            
            transfer_log = f"TRANSFER: {', '.join(transfer_info)}"
            if call_logs:
                call_logs += f"\n{transfer_log}"
            else:
                call_logs = transfer_log
        
        params = (
            now_utc.strftime('%Y-%m-%d %H:%M:%S'),
            call_data.get('disposition', 'UNKNOWN'),
            call_data.get('transcript', ''),
            end_time_utc.strftime('%Y-%m-%d %H:%M:%S') if end_time_utc else None,
            call_data.get('filename'),
            call_data.get('file_mime_type'),
            call_data.get('call_drop_step'),
            call_logs,
            call_data.get('caller_state'),
            call_id
        )
        
        result = CentralizedDBManager.execute_query(query, params, context="async_call_end", logger=db_logger)
        
        if result is not None:
            self._logger.debug(f"Async: Call {call_id} end logged to centralized database")
        else:
            raise Exception(f"Failed to log call end for {call_id}")
    
    def _fallback_log(self, operation, args, error):
        """Write to fallback file when database unavailable"""
        try:
            fallback_file = os.path.join(self.fallback_log_dir, f"failed_database_ops_{datetime.now().strftime('%Y%m%d')}.log")
            with open(fallback_file, 'a') as f:
                timestamp = datetime.now().isoformat()
                f.write(f"{timestamp} | {operation} | ERROR: {error} | ARGS: {args}\n")
        except Exception as e:
            self._logger.error(f"Failed to write fallback log: {e}")
    
    def queue_call_start(self, call_data, agent_config, db_logger) -> str:
        """
        Queue call start operation for background processing
        Returns call ID immediately without waiting for database
        """
        call_id = str(uuid.uuid4())
        
        try:
            work_item = ('call_start', (call_id, call_data, agent_config, db_logger))
            self.work_queue.put(work_item, block=False)
            
            with self.stats_lock:
                self.stats['queued_operations'] += 1
            
            return call_id
            
        except queue.Full:
            with self.stats_lock:
                self.stats['queue_full_errors'] += 1
            
            self._logger.warning("Async database queue full - falling back to file logging")
            self._fallback_log('call_start', (call_id, call_data, agent_config), "Queue full")
            return call_id
    
    def queue_call_end(self, call_data, db_logger):
        """
        Queue call end operation for background processing
        Returns immediately without waiting for database
        """
        try:
            work_item = ('call_end', (call_data, db_logger))
            self.work_queue.put(work_item, block=False)
            
            with self.stats_lock:
                self.stats['queued_operations'] += 1
                
        except queue.Full:
            with self.stats_lock:
                self.stats['queue_full_errors'] += 1
            
            self._logger.warning("Async database queue full - falling back to file logging")
            self._fallback_log('call_end', (call_data,), "Queue full")
    
    def get_stats(self):
        """Get async logger statistics"""
        with self.stats_lock:
            return self.stats.copy()
    
    def shutdown(self, timeout=10):
        """Gracefully shutdown async logger"""
        self._logger.info("Shutting down async call logger...")
        
        # Signal shutdown and add None items to wake up workers
        self.shutdown_event.set()
        for _ in range(self.num_workers):
            try:
                self.work_queue.put(None, timeout=1)
            except queue.Full:
                pass
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout/self.num_workers)
        
        # Wait for remaining queue items
        try:
            self.work_queue.join()
        except:
            pass
        
        stats = self.get_stats()
        self._logger.info(f"Async logger shutdown complete. Final stats: {stats}")

class SuiteCRMLogger:
    """Thread-safe call logger for SuiteCRM"""

    def __init__(self, agent_config: SuiteCRMAgentConfig, logger: logging.LoggerAdapter, bot_instance=None):
        self.agent_config = agent_config
        self.logger = logger
        self.bot_instance = bot_instance
        self.lock = threading.Lock()

        # Create database logger wrapper if bot instance is available
        if bot_instance:
            self.db_logger = DatabaseLogger(bot_instance)
        else:
            self.db_logger = logger

        # Get async logger instance for non-blocking database operations
        self.async_logger = AsyncCallLogger.get_instance()

        # Get HP cache instance for fast disposition checks
        try:
            from src.hp_disposition_cache import get_hp_cache
            self.hp_cache = get_hp_cache()
        except ImportError:
            self.logger.warning("HP Disposition Cache not available")
            self.hp_cache = None
    
    def log_call_start(self, call_data: Dict[str, Any]) -> Optional[str]:
        """
        Queue call start record for async database processing
        Returns call ID immediately without waiting for database
        """
        try:
            # Use async logger to queue database operation - returns immediately
            call_id = self.async_logger.queue_call_start(call_data, self.agent_config, self.db_logger)
            
            self.logger.info(f"Call {call_id} queued for async database logging (phone: {call_data.get('phone_number')})")
            return call_id
            
        except Exception as e:
            self.logger.error(f"Error queueing call start: {e}", exc_info=True)
            # Fallback: generate call ID even if logging fails
            fallback_id = str(uuid.uuid4())
            self.logger.warning(f"Using fallback call ID {fallback_id} for {call_data.get('phone_number')}")
            return fallback_id
    
    def log_call_end(self, call_data: Dict[str, Any]):
        """
        Queue call end record for async database processing
        Returns immediately without waiting for database
        """
        call_id = call_data.get('id')
        if not call_id:
            self.logger.warning("Cannot log call end: No call ID")
            return

        try:
            # Use async logger to queue database operation - returns immediately
            self.async_logger.queue_call_end(call_data, self.db_logger)

            self.logger.info(f"Call {call_id} end queued for async database logging (disposition: {call_data.get('disposition', 'UNKNOWN')})")

        except Exception as e:
            self.logger.error(f"Error queueing call end for {call_id}: {e}", exc_info=True)

    def check_hp_disposition(self, phone_number: str) -> bool:
        """
        Check if phone number has HP (Hangup Prior) disposition
        Uses lightning-fast memory cache (non-blocking, <1ms)

        Args:
            phone_number: Phone number to check

        Returns:
            True if HP record exists (should hangup), False otherwise
        """
        if not phone_number:
            return False

        # Use cache if available (non-blocking memory lookup)
        if self.hp_cache:
            try:
                return self.hp_cache.check_hp_disposition(phone_number)
            except Exception as e:
                self.logger.error(f"Error checking HP cache for {phone_number}: {e}", exc_info=True)
                return False  # Conservative: allow call on error
        else:
            # Fallback: No cache available, allow call (conservative approach)
            self.logger.debug(f"HP cache not available for {phone_number}, allowing call")
            return False

class CentralizedDBManager:
    """Database manager for centralized call logging only"""
    
    _pool = None
    _pool_lock = threading.Lock()
    _pool_name = "centralized_pool"
    _pool_size = 30  # Under MySQL connector maximum with safety margin
    _max_overflow = 0  # Overflow not supported by MySQL connector
    _logger = logging.getLogger(__name__)
    
    @classmethod
    def _create_pool(cls):
        """Create centralized connection pool"""
        try:
            if cls._pool is None:
                pool_config = CENTRALIZED_DB_CONFIG.copy()
                pool_config['pool_name'] = cls._pool_name
                pool_config['pool_size'] = cls._pool_size
                pool_config['pool_reset_session'] = True
                
                cls._pool = pooling.MySQLConnectionPool(**pool_config)
                cls._logger.info(f"✅ Centralized database connection pool created (size={cls._pool_size})")
        except mysql.connector.Error as err:
            cls._logger.error(f"❌ Failed to create centralized connection pool: {err}")
            cls._pool = None
            raise
    
    @classmethod
    @contextmanager
    def get_connection(cls):
        """Get centralized connection with automatic management"""
        connection = None
        retry_count = 3
        retry_delay = 1
        
        for attempt in range(retry_count):
            try:
                with cls._pool_lock:
                    if cls._pool is None:
                        cls._create_pool()
                
                connection = cls._pool.get_connection()
                connection.ping(reconnect=True, attempts=3, delay=1)
                
                yield connection
                
                if connection.in_transaction:
                    connection.commit()
                
                return
                
            except mysql.connector.PoolError as err:
                cls._logger.warning(f"Centralized pool error (attempt {attempt + 1}): {err}")
                
                if attempt < retry_count - 1:
                    time.sleep(retry_delay * (attempt + 1))
                    with cls._pool_lock:
                        cls._pool = None
                else:
                    raise
                    
            except mysql.connector.Error as err:
                cls._logger.error(f"Centralized database error: {err}")
                
                if connection and connection.is_connected():
                    connection.rollback()
                
                if attempt < retry_count - 1:
                    time.sleep(retry_delay * (attempt + 1))
                else:
                    raise
                    
            finally:
                if connection and connection.is_connected():
                    try:
                        connection.close()
                    except:
                        pass
    
    @classmethod
    def execute_query(cls, query: str, params: tuple = None, fetch: str = None, 
                     retry: bool = True, context: str = None, logger: logging.Logger = None) -> Any:
        """
        Execute query on centralized database with latency tracking
        
        Args:
            query: SQL query
            params: Query parameters
            fetch: 'one', 'all', or None
            retry: Whether to retry on failure
            context: Optional context for query identification in logs
            logger: Optional logger to use instead of module logger (for call record integration)
        """
        max_retries = 3 if retry else 1
        query_start_time = time.time()
        
        # Use provided logger or fall back to module logger
        active_logger = logger if logger else cls._logger
        
        # Extract query type and table for logging
        query_type = query.strip().split()[0].upper()
        query_preview = query.replace('\n', ' ').replace('\t', ' ')[:100] + "..."
        context_info = f"[{context.upper()}]" if context else "[CALL_LOGGING]"
        
        for attempt in range(max_retries):
            attempt_start_time = time.time()
            try:
                with cls.get_connection() as conn:
                    cursor = conn.cursor(dictionary=True, buffered=True)
                    
                    try:
                        cursor.execute(query, params)
                        
                        result = None
                        row_count = 0
                        
                        if fetch == 'one':
                            result = cursor.fetchone()
                            row_count = 1 if result else 0
                        elif fetch == 'all':
                            result = cursor.fetchall()
                            row_count = len(result) if result else 0
                        else:
                            conn.commit()
                            result = cursor.lastrowid or cursor.rowcount
                            row_count = result if result else 0
                        
                        # Calculate and log latency
                        latency_ms = (time.time() - query_start_time) * 1000
                        
                        if latency_ms > 1000:  # Slow query warning
                            active_logger.warning(f"⚠️ 🌐 {context_info} SLOW QUERY: {latency_ms:.2f}ms | {query_type} | Rows: {row_count}")
                        else:
                            active_logger.info(f"🌐 {context_info} Query: {latency_ms:.2f}ms | {query_type} | Rows: {row_count}")
                        
                        # Debug log for very detailed tracking
                        active_logger.debug(f"🌐 {context_info} Query details: {query_preview}")
                        
                        return result
                            
                    finally:
                        cursor.close()
                        
            except mysql.connector.Error as err:
                attempt_latency_ms = (time.time() - attempt_start_time) * 1000
                active_logger.error(f"❌ 🌐 {context_info} Query error (attempt {attempt + 1}): {err} | {attempt_latency_ms:.2f}ms")
                
                if attempt < max_retries - 1:
                    time.sleep(1 * (attempt + 1))
                else:
                    total_latency_ms = (time.time() - query_start_time) * 1000
                    active_logger.error(f"❌ 🌐 {context_info} Query failed after {max_retries} attempts | Total: {total_latency_ms:.2f}ms")
                    return None
                    
            except Exception as e:
                attempt_latency_ms = (time.time() - attempt_start_time) * 1000
                active_logger.error(f"❌ 🌐 {context_info} Unexpected error (attempt {attempt + 1}): {e} | {attempt_latency_ms:.2f}ms", exc_info=True)
                if attempt >= max_retries - 1:
                    return None
        
        return None
    
    @classmethod
    def get_pool_stats(cls) -> Dict[str, Any]:
        """Get detailed centralized connection pool statistics"""
        stats = {
            'pool_name': cls._pool_name,
            'pool_size': cls._pool_size,
            'max_overflow': cls._max_overflow,
            'available_connections': 0,
            'checked_out_connections': 0,
            'overflow_connections': 0,
            'total_capacity': cls._pool_size + cls._max_overflow,
            'utilization_percent': 0,
            'pool_healthy': False,
            'last_check': datetime.now().isoformat()
        }
        
        try:
            with cls._pool_lock:
                if cls._pool is not None:
                    # Get pool statistics (handle different pool implementations)
                    try:
                        if hasattr(cls._pool, '_pool') and hasattr(cls._pool._pool, 'qsize'):
                            stats['available_connections'] = cls._pool._pool.qsize()
                        else:
                            # Fallback for newer versions of mysql-connector-python
                            stats['available_connections'] = cls._pool_size  # Assume healthy for now
                    except:
                        stats['available_connections'] = cls._pool_size  # Safe fallback
                    
                    stats['checked_out_connections'] = cls._pool_size - stats['available_connections']
                    stats['overflow_connections'] = max(0, stats['checked_out_connections'] - cls._pool_size)
                    
                    # Calculate utilization
                    if stats['total_capacity'] > 0:
                        stats['utilization_percent'] = (stats['checked_out_connections'] / stats['total_capacity']) * 100
                    
                    # Health check
                    stats['pool_healthy'] = (
                        stats['utilization_percent'] < 90 and  # Less than 90% utilized
                        stats['available_connections'] > 0  # At least one connection available
                    )
                    
                    # Log warning if pool is heavily utilized
                    if stats['utilization_percent'] > 80:
                        cls._logger.warning(f"⚠️ Centralized pool high utilization: {stats['utilization_percent']:.1f}% "
                                          f"({stats['checked_out_connections']}/{stats['total_capacity']} connections)")
                        
        except Exception as e:
            cls._logger.error(f"Error getting centralized pool stats: {e}")
            stats['error'] = str(e)
        
        return stats
    
    @classmethod
    def health_check(cls) -> bool:
        """Perform health check on centralized connection pool"""
        try:
            with cls._pool_lock:
                if cls._pool is None:
                    cls._logger.warning("Centralized connection pool not initialized")
                    return False
            
            # Try to get a connection briefly
            try:
                with cls.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1")
                    result = cursor.fetchone()
                    cursor.close()
                    
                    if result:
                        cls._logger.debug("Centralized pool health check passed")
                        return True
                    else:
                        cls._logger.error("Centralized pool health check failed: No result")
                        return False
                        
            except Exception as e:
                cls._logger.error(f"Centralized pool health check failed: {e}")
                return False
                
        except Exception as e:
            cls._logger.error(f"Centralized pool health check error: {e}")
            return False
    
    @classmethod
    def monitor_pool_health(cls, alert_threshold: float = 85.0):
        """Monitor centralized pool health and log alerts if needed"""
        try:
            stats = cls.get_pool_stats()
            
            # Alert on high utilization
            if stats['utilization_percent'] > alert_threshold:
                cls._logger.warning(f"🚨 CENTRALIZED POOL ALERT: {stats['utilization_percent']:.1f}% utilization "
                                  f"({stats['checked_out_connections']}/{stats['total_capacity']} connections used)")
            
            # Alert on pool exhaustion
            if stats['available_connections'] == 0:
                cls._logger.error(f"🚨 CENTRALIZED POOL EXHAUSTED: No available connections! "
                                f"All {stats['total_capacity']} connections in use")
            
            # Alert on unhealthy pool
            if not stats['pool_healthy']:
                cls._logger.error(f"🚨 CENTRALIZED POOL UNHEALTHY: Health check failed")
            
            # Log stats at debug level
            cls._logger.debug(f"Centralized Pool Stats: {stats['checked_out_connections']}/{stats['total_capacity']} "
                            f"used ({stats['utilization_percent']:.1f}%), {stats['available_connections']} available")
            
            return stats
            
        except Exception as e:
            cls._logger.error(f"Error monitoring centralized pool health: {e}")
            return None
    
    @classmethod
    def close_pool(cls):
        """Close centralized connection pool"""
        with cls._pool_lock:
            if cls._pool:
                try:
                    cls._pool._remove_connections()
                    cls._pool = None
                    cls._logger.info("Centralized database connection pool closed")
                except:
                    pass

def monitor_all_pools() -> Dict[str, Any]:
    """
    Monitor health of all database connection pools
    Returns combined statistics and health status
    """
    monitoring_results = {
        'timestamp': datetime.now().isoformat(),
        'pools': {},
        'overall_healthy': True,
        'alerts': []
    }
    
    # Monitor SuiteCRM pool
    try:
        suitecrm_stats = SuiteCRMDBManager.monitor_pool_health()
        monitoring_results['pools']['suitecrm'] = suitecrm_stats
        
        if suitecrm_stats and not suitecrm_stats.get('pool_healthy', False):
            monitoring_results['overall_healthy'] = False
            monitoring_results['alerts'].append('SuiteCRM pool unhealthy')
            
    except Exception as e:
        monitoring_results['pools']['suitecrm'] = {'error': str(e)}
        monitoring_results['overall_healthy'] = False
        monitoring_results['alerts'].append(f'SuiteCRM pool monitoring failed: {e}')
    
    # Monitor Centralized pool
    try:
        centralized_stats = CentralizedDBManager.monitor_pool_health()
        monitoring_results['pools']['centralized'] = centralized_stats
        
        if centralized_stats and not centralized_stats.get('pool_healthy', False):
            monitoring_results['overall_healthy'] = False
            monitoring_results['alerts'].append('Centralized pool unhealthy')
            
    except Exception as e:
        monitoring_results['pools']['centralized'] = {'error': str(e)}
        monitoring_results['overall_healthy'] = False
        monitoring_results['alerts'].append(f'Centralized pool monitoring failed: {e}')
    
    # Monitor AsyncCallLogger if available
    try:
        async_logger = AsyncCallLogger.get_instance()
        async_stats = async_logger.get_stats()
        monitoring_results['async_logger'] = async_stats
        
        # Check for high failure rate
        if async_stats['queued_operations'] > 0:
            failure_rate = (async_stats['failed_operations'] / async_stats['queued_operations']) * 100
            if failure_rate > 10:  # More than 10% failure rate
                monitoring_results['alerts'].append(f'High async logger failure rate: {failure_rate:.1f}%')
        
        # Check for queue full errors
        if async_stats['queue_full_errors'] > 0:
            monitoring_results['alerts'].append(f'Async logger queue full errors: {async_stats["queue_full_errors"]}')
            
    except Exception as e:
        monitoring_results['async_logger'] = {'error': str(e)}
        monitoring_results['alerts'].append(f'Async logger monitoring failed: {e}')
    
    return monitoring_results

def periodic_health_monitor(interval_seconds: int = 300):
    """
    Start periodic health monitoring of all database components
    Should be called from a background thread
    """
    import threading
    import time
    
    logger = logging.getLogger(__name__)
    
    def monitor_loop():
        while True:
            try:
                results = monitor_all_pools()
                
                # Log summary
                if results['overall_healthy']:
                    logger.info(f"✅ Database health check passed - All pools healthy")
                else:
                    logger.warning(f"⚠️ Database health issues detected: {', '.join(results['alerts'])}")
                
                # Log detailed stats at debug level
                logger.debug(f"Database Pool Monitoring Results: {results}")
                
            except Exception as e:
                logger.error(f"Error in periodic health monitor: {e}", exc_info=True)
            
            time.sleep(interval_seconds)
    
    monitor_thread = threading.Thread(target=monitor_loop, name="DatabaseHealthMonitor", daemon=True)
    monitor_thread.start()
    logger.info(f"✅ Started periodic database health monitoring (interval: {interval_seconds}s)")
    
    return monitor_thread

# Cleanup on module exit
import atexit

def cleanup_all_database_resources():
    """Cleanup all database resources including async logger"""
    try:
        # Shutdown async logger first
        async_logger = AsyncCallLogger.get_instance()
        if async_logger:
            async_logger.shutdown(timeout=5)
    except:
        pass
    
    # Close connection pools
    try:
        SuiteCRMDBManager.close_pool()
    except:
        pass
    
    try:
        CentralizedDBManager.close_pool()
    except:
        pass

atexit.register(cleanup_all_database_resources)