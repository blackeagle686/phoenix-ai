import time
from typing import Optional

def current_timestamp() -> float:
    return time.time()

def calculate_expiry(ttl_seconds: Optional[int]) -> Optional[float]:
    if ttl_seconds is None:
        return None
    return current_timestamp() + ttl_seconds

def has_expired(expiry_timestamp: Optional[float]) -> bool:
    if expiry_timestamp is None:
        return False
    return current_timestamp() > expiry_timestamp

