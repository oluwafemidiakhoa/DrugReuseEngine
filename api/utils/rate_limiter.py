import time
from collections import defaultdict
from typing import Dict, List, Tuple
from fastapi import Request, HTTPException, status

# Storage for rate limiting information
# Key: IP address
# Value: List of timestamps of requests
request_history: Dict[str, List[float]] = defaultdict(list)

# Rate limiting configuration
RATE_LIMIT_WINDOW = 60  # Window in seconds
RATE_LIMIT_MAX_REQUESTS = 100  # Maximum requests per window
RATE_LIMIT_USER_WINDOW = 60  # Window in seconds for authenticated users
RATE_LIMIT_USER_MAX_REQUESTS = 200  # Maximum requests per window for authenticated users


def _clean_old_requests():
    """Remove requests older than the window from history"""
    current_time = time.time()
    for ip, timestamps in list(request_history.items()):
        # Keep only timestamps within the window
        new_timestamps = [ts for ts in timestamps if current_time - ts < RATE_LIMIT_WINDOW]
        if new_timestamps:
            request_history[ip] = new_timestamps
        else:
            # If all requests are old, remove the entry
            del request_history[ip]


async def rate_limit_middleware(request: Request, call_next):
    """Middleware for rate limiting requests"""
    # Clean up old requests first
    _clean_old_requests()
    
    # Get client IP
    client_ip = request.client.host if request.client else "unknown"
    
    # Get current time
    current_time = time.time()
    
    # Add request to history
    request_history[client_ip].append(current_time)
    
    # Check if rate limit is exceeded
    if len(request_history[client_ip]) > RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    
    # Process the request
    response = await call_next(request)
    return response


class RateLimiter:
    """Class for checking rate limits on specific endpoints"""
    
    def __init__(self, window_seconds: int = RATE_LIMIT_WINDOW, max_requests: int = RATE_LIMIT_MAX_REQUESTS):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.requests: Dict[str, List[float]] = defaultdict(list)
    
    def _clean_old_requests(self, key: str):
        """Remove requests older than the window for a specific key"""
        current_time = time.time()
        # Keep only timestamps within the window
        self.requests[key] = [
            ts for ts in self.requests[key] 
            if current_time - ts < self.window_seconds
        ]
    
    def is_rate_limited(self, key: str) -> Tuple[bool, int]:
        """
        Check if the key is rate limited
        
        Returns:
        - is_limited: Boolean indicating if rate limit exceeded
        - remaining: Number of requests remaining in window
        """
        self._clean_old_requests(key)
        current_time = time.time()
        self.requests[key].append(current_time)
        
        # Check if rate limit is exceeded
        requests_count = len(self.requests[key])
        is_limited = requests_count > self.max_requests
        remaining = max(0, self.max_requests - requests_count)
        
        return is_limited, remaining