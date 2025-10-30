#!/usr/bin/env python3
"""
HP Disposition Cache - Lightning-fast phone number lookups
Eliminates blocking database queries during PJSIP call setup

Features:
- Pre-loads ALL historical HP dispositions at startup
- Thread-safe singleton pattern
- Background refresh (full + incremental)
- Zero PJSIP thread blocking (memory-only lookups)
- Graceful degradation on cache misses
"""

import threading
import time
import logging
from typing import Optional, Set, Dict, Any
from datetime import datetime

# Import database manager (will be available when integrated)
try:
    from src.suitecrm_integration import CentralizedDBManager
except ImportError:
    # Fallback for testing
    CentralizedDBManager = None


class HPDispositionCache:
    """
    Thread-safe singleton cache for HP (Hangup Prior) phone number dispositions

    Architecture:
    - Singleton pattern for global access across all call instances
    - Set-based storage for O(1) lookup performance
    - Background refresh threads for data freshness
    - cache_ready event for graceful startup handling
    """

    _instance = None
    _lock = threading.RLock()  # Reentrant lock for nested calls
    _logger = logging.getLogger(__name__)

    def __init__(self):
        """Initialize cache (called once by singleton pattern)"""
        # Cache storage - using set for O(1) membership testing
        self._hp_cache: Set[str] = set()

        # Thread synchronization
        self._cache_lock = threading.RLock()
        self._cache_ready = threading.Event()
        self._shutdown_event = threading.Event()

        # Refresh threads
        self._refresh_thread = None
        self._running = False

        # Refresh configuration (seconds)
        self._full_refresh_interval = 6 * 3600  # 6 hours
        self._incremental_refresh_interval = 5 * 60  # 5 minutes
        self._last_full_refresh = 0
        self._last_incremental_refresh = 0

        # Statistics
        self._stats = {
            'cache_size': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'full_refreshes': 0,
            'incremental_refreshes': 0,
            'last_refresh_time': None,
            'last_refresh_duration_ms': 0,
            'total_loaded_numbers': 0,
            'errors': 0
        }
        self._stats_lock = threading.Lock()

        self._logger.info("🔄 HP Disposition Cache initialized")

    @classmethod
    def get_instance(cls):
        """Get singleton instance (thread-safe)"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def start_preload(self, blocking=False, shutdown_event=None):
        """
        Start cache pre-loading

        Args:
            blocking: If True, wait for initial load to complete
            shutdown_event: External shutdown event to monitor (from main server)
        """
        if self._running:
            self._logger.warning("Cache preload already running")
            return

        self._running = True

        # Start background refresh thread
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop,
            name="HPCacheRefresh",
            args=(shutdown_event,),
            daemon=True
        )
        self._refresh_thread.start()

        self._logger.info("🚀 HP Cache preload started (background thread)")

        # Wait for initial load if blocking requested
        if blocking:
            self._logger.info("⏳ Waiting for initial HP cache load...")
            if self._cache_ready.wait(timeout=60):
                self._logger.info("✅ HP Cache ready")
            else:
                self._logger.warning("⚠️ HP Cache load timeout - continuing anyway")

    def _refresh_loop(self, shutdown_event=None):
        """Background thread for cache refresh"""
        try:
            # Initial full load
            self._logger.info("📥 Starting initial HP cache load (all historical records)...")
            if self._load_all_hp_numbers():
                self._cache_ready.set()
                self._logger.info(f"✅ Initial HP cache load complete: {len(self._hp_cache)} numbers loaded")
            else:
                self._logger.error("❌ Initial HP cache load failed")
                self._cache_ready.set()  # Set ready anyway to avoid blocking

            # Periodic refresh loop
            while self._running and not self._shutdown_event.is_set():
                # Check external shutdown event if provided
                if shutdown_event and shutdown_event.is_set():
                    self._logger.info("External shutdown detected, stopping HP cache refresh")
                    break

                try:
                    current_time = time.time()

                    # Check if full refresh is needed
                    if current_time - self._last_full_refresh >= self._full_refresh_interval:
                        self._logger.info("🔄 Starting full HP cache refresh...")
                        if self._load_all_hp_numbers():
                            with self._stats_lock:
                                self._stats['full_refreshes'] += 1
                            self._logger.info(f"✅ Full refresh complete: {len(self._hp_cache)} numbers")
                        else:
                            self._logger.error("❌ Full refresh failed")

                    # Check if incremental refresh is needed
                    elif current_time - self._last_incremental_refresh >= self._incremental_refresh_interval:
                        self._logger.debug("🔄 Starting incremental HP cache refresh...")
                        new_count = self._load_recent_hp_numbers()
                        if new_count is not None:
                            with self._stats_lock:
                                self._stats['incremental_refreshes'] += 1
                            if new_count > 0:
                                self._logger.info(f"✅ Incremental refresh: {new_count} new HP numbers added")
                            else:
                                self._logger.debug("Incremental refresh: No new HP numbers")
                        else:
                            self._logger.error("❌ Incremental refresh failed")

                    # Sleep briefly before next check
                    time.sleep(10)  # Check every 10 seconds

                except Exception as e:
                    self._logger.error(f"Error in refresh loop: {e}", exc_info=True)
                    with self._stats_lock:
                        self._stats['errors'] += 1
                    time.sleep(60)  # Wait longer on error

        except Exception as e:
            self._logger.error(f"Fatal error in refresh loop: {e}", exc_info=True)
        finally:
            self._logger.info("HP Cache refresh loop stopped")

    def _load_all_hp_numbers(self) -> bool:
        """
        Load ALL historical HP disposition numbers from database
        Returns True on success, False on failure
        """
        if not CentralizedDBManager:
            self._logger.error("CentralizedDBManager not available")
            return False

        start_time = time.time()

        try:
            # Query: Load ALL HP dispositions (no time filter)
            query = """
            SELECT DISTINCT name
            FROM e_call
            WHERE disposition = 'HP'
              AND deleted = 0
              AND name IS NOT NULL
              AND name != ''
            """

            self._logger.debug("Executing HP cache query (all historical records)...")
            results = CentralizedDBManager.execute_query(
                query=query,
                fetch='all',
                context='hp_cache_full',
                logger=self._logger
            )

            if results is None:
                self._logger.error("Failed to query HP dispositions")
                return False

            # Load into set (atomic replacement for thread safety)
            new_cache = {row['name'] for row in results if row.get('name')}

            # Update cache atomically
            with self._cache_lock:
                self._hp_cache = new_cache
                cache_size = len(self._hp_cache)

            # Update statistics
            duration_ms = (time.time() - start_time) * 1000
            with self._stats_lock:
                self._stats['cache_size'] = cache_size
                self._stats['last_refresh_time'] = datetime.now().isoformat()
                self._stats['last_refresh_duration_ms'] = duration_ms
                self._stats['total_loaded_numbers'] = cache_size

            self._last_full_refresh = time.time()

            self._logger.info(f"📥 Loaded {cache_size} HP numbers in {duration_ms:.2f}ms")
            return True

        except Exception as e:
            self._logger.error(f"Error loading HP numbers: {e}", exc_info=True)
            with self._stats_lock:
                self._stats['errors'] += 1
            return False

    def _load_recent_hp_numbers(self) -> Optional[int]:
        """
        Load recent HP disposition numbers (incremental update)
        Returns count of new numbers added, or None on failure
        """
        if not CentralizedDBManager:
            return None

        try:
            # Query: Only HP dispositions from last 10 minutes (safety margin)
            query = """
            SELECT DISTINCT name
            FROM e_call
            WHERE disposition = 'HP'
              AND deleted = 0
              AND name IS NOT NULL
              AND name != ''
              AND date_modified >= DATE_SUB(NOW(), INTERVAL 10 MINUTE)
            """

            results = CentralizedDBManager.execute_query(
                query=query,
                fetch='all',
                context='hp_cache_incremental',
                logger=self._logger
            )

            if results is None:
                return None

            # Extract phone numbers
            new_numbers = {row['name'] for row in results if row.get('name')}

            # Add to cache (only new numbers)
            with self._cache_lock:
                before_size = len(self._hp_cache)
                self._hp_cache.update(new_numbers)
                after_size = len(self._hp_cache)
                new_count = after_size - before_size

            # Update statistics
            with self._stats_lock:
                self._stats['cache_size'] = after_size

            self._last_incremental_refresh = time.time()

            return new_count

        except Exception as e:
            self._logger.error(f"Error in incremental refresh: {e}", exc_info=True)
            with self._stats_lock:
                self._stats['errors'] += 1
            return None

    def check_hp_disposition(self, phone_number: str) -> bool:
        """
        Check if phone number has HP disposition (memory lookup - non-blocking)

        Args:
            phone_number: Phone number to check

        Returns:
            True if HP record exists (should hangup), False otherwise
        """
        if not phone_number:
            return False

        # Wait for cache to be ready (with timeout for safety)
        if not self._cache_ready.is_set():
            if not self._cache_ready.wait(timeout=0.1):
                # Cache not ready yet - conservative approach: allow call
                self._logger.debug(f"HP check for {phone_number}: cache not ready, allowing call")
                with self._stats_lock:
                    self._stats['cache_misses'] += 1
                return False

        # Fast memory lookup (O(1) with set)
        with self._cache_lock:
            is_hp = phone_number in self._hp_cache

        # Update statistics
        with self._stats_lock:
            if is_hp:
                self._stats['cache_hits'] += 1
            else:
                self._stats['cache_misses'] += 1

        if is_hp:
            self._logger.info(f"🚫 HP MATCH: {phone_number} found in cache")
        else:
            self._logger.debug(f"✅ HP check: {phone_number} not in cache (allow call)")

        return is_hp

    def is_ready(self) -> bool:
        """Check if cache is ready for lookups"""
        return self._cache_ready.is_set()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._stats_lock:
            stats = self._stats.copy()

        stats['cache_ready'] = self._cache_ready.is_set()
        stats['running'] = self._running

        # Calculate hit rate
        total_checks = stats['cache_hits'] + stats['cache_misses']
        if total_checks > 0:
            stats['hit_rate_percent'] = (stats['cache_hits'] / total_checks) * 100
        else:
            stats['hit_rate_percent'] = 0.0

        return stats

    def shutdown(self, timeout=10):
        """Gracefully shutdown cache refresh"""
        self._logger.info("Shutting down HP Disposition Cache...")

        self._running = False
        self._shutdown_event.set()

        # Wait for refresh thread
        if self._refresh_thread and self._refresh_thread.is_alive():
            self._refresh_thread.join(timeout=timeout)

        # Log final statistics
        stats = self.get_stats()
        self._logger.info(f"HP Cache shutdown complete. Final stats: {stats}")

    def force_refresh(self):
        """Force immediate full refresh (for testing/debugging)"""
        self._logger.info("Forcing immediate HP cache refresh...")
        success = self._load_all_hp_numbers()
        if success:
            self._logger.info("✅ Force refresh complete")
        else:
            self._logger.error("❌ Force refresh failed")
        return success


# Global instance accessor (convenience function)
def get_hp_cache() -> HPDispositionCache:
    """Get global HP cache instance"""
    return HPDispositionCache.get_instance()


# Cleanup on module exit
import atexit

def cleanup_hp_cache():
    """Cleanup HP cache on exit"""
    try:
        cache = HPDispositionCache.get_instance()
        if cache:
            cache.shutdown(timeout=5)
    except:
        pass

atexit.register(cleanup_hp_cache)
