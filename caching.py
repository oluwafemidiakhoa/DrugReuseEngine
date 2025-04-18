"""
Caching utilities for the Drug Repurposing Engine.

This module provides functionality to cache expensive operations and database queries,
helping to improve the performance and responsiveness of the application.
"""

import os
import json
import time
import hashlib
import logging
import pickle
from functools import wraps
from typing import Any, Dict, List, Optional, Callable, Union, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

class Cache:
    """Cache manager for storing and retrieving expensive computation results."""
    
    @staticmethod
    def get_cache_path(key: str) -> str:
        """Get the file path for a cache item"""
        # Create a hash of the key to ensure valid filename
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(CACHE_DIR, f"{hash_key}.cache")
    
    @staticmethod
    def exists(key: str) -> bool:
        """Check if a cache item exists"""
        cache_path = Cache.get_cache_path(key)
        return os.path.exists(cache_path)
    
    @staticmethod
    def get(key: str, max_age: Optional[int] = None) -> Any:
        """
        Get a cached item
        
        Args:
            key: The cache key
            max_age: Maximum age in seconds (None for no limit)
            
        Returns:
            The cached data or None if not found or expired
        """
        if not Cache.exists(key):
            return None
            
        cache_path = Cache.get_cache_path(key)
        
        # Check cache age if max_age is specified
        if max_age is not None:
            file_age = time.time() - os.path.getmtime(cache_path)
            if file_age > max_age:
                logger.info(f"Cache expired for {key} (age: {file_age}s, max: {max_age}s)")
                return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except (pickle.PickleError, IOError, EOFError) as e:
            logger.warning(f"Failed to load cache for {key}: {str(e)}")
            return None
    
    @staticmethod
    def set(key: str, data: Any) -> bool:
        """
        Set a cache item
        
        Args:
            key: The cache key
            data: The data to cache
            
        Returns:
            bool: True if successful, False otherwise
        """
        cache_path = Cache.get_cache_path(key)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Cache set for {key}")
            return True
        except (pickle.PickleError, IOError) as e:
            logger.warning(f"Failed to save cache for {key}: {str(e)}")
            return False
    
    @staticmethod
    def invalidate(key: str) -> bool:
        """
        Invalidate a cache item
        
        Args:
            key: The cache key
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not Cache.exists(key):
            return True
            
        cache_path = Cache.get_cache_path(key)
        
        try:
            os.remove(cache_path)
            logger.info(f"Cache invalidated for {key}")
            return True
        except IOError as e:
            logger.warning(f"Failed to invalidate cache for {key}: {str(e)}")
            return False
    
    @staticmethod
    def clear_all() -> int:
        """
        Clear all cached items
        
        Returns:
            int: Number of cleared cache files
        """
        count = 0
        for filename in os.listdir(CACHE_DIR):
            if filename.endswith(".cache"):
                try:
                    os.remove(os.path.join(CACHE_DIR, filename))
                    count += 1
                except IOError:
                    pass
        
        logger.info(f"Cleared {count} cache files")
        return count

def cached(max_age_seconds: int = 3600, key_prefix: str = ""):
    """
    Decorator to cache function results
    
    Args:
        max_age_seconds: Maximum age of cached result in seconds
        key_prefix: Prefix for cache keys
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create a cache key from function name, args, and kwargs
            key_parts = [key_prefix, func.__name__]
            
            # Add args to key
            for arg in args:
                try:
                    key_parts.append(str(arg))
                except:
                    key_parts.append("UNSTRINGABLE_ARG")
            
            # Add kwargs to key (sorted for consistency)
            sorted_kwargs = sorted(kwargs.items())
            for k, v in sorted_kwargs:
                try:
                    key_parts.append(f"{k}={v}")
                except:
                    key_parts.append(f"{k}=UNSTRINGABLE_VALUE")
            
            cache_key = ":".join(key_parts)
            
            # Check if result is cached and not expired
            cached_result = Cache.get(cache_key, max_age_seconds)
            if cached_result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_result
            
            # Call the function and cache the result
            result = func(*args, **kwargs)
            Cache.set(cache_key, result)
            
            return result
        
        return wrapper
    
    return decorator


# Streamlit session-based caching utilities
def get_session_cache_key(key_parts: List[Any]) -> str:
    """
    Generate a cache key for Streamlit session-based caching
    
    Args:
        key_parts: Parts to include in the cache key
        
    Returns:
        str: A concatenated cache key
    """
    key_strings = []
    for part in key_parts:
        if part is None:
            key_strings.append("None")
        elif isinstance(part, (str, int, float, bool)):
            key_strings.append(str(part))
        else:
            try:
                # Try to convert more complex objects to JSON strings
                key_strings.append(json.dumps(part))
            except:
                # Fall back to string representation with a hash
                hash_value = hashlib.md5(str(part).encode()).hexdigest()[:10]
                key_strings.append(f"complex_{hash_value}")
    
    return "_".join(key_strings)