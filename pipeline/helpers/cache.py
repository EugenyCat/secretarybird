from dogpile.cache import make_region
from dogpile.cache.api import NO_VALUE
import os

# 1. Configure cache region for Redis
cache_region = make_region().configure(
    backend='dogpile.cache.redis',
    expiration_time=3600,  # default expiration time — 1 hour
    arguments={
        'url': os.getenv("REDIS_URL"),  # redis service name from Docker Compose
        'distributed_lock': True,       # locks for concurrent recalculation
        'thread_local_lock': False,
    }
)

def set_cache(key: str, value, ttl: int = None):
    """
    Save to cache.
    :param key: string key
    :param value: serializable value
    :param ttl: time to live in seconds (default — expiration_time from configuration)
    """
    if ttl:
        cache_region.set(key, value, expiration_time=ttl)
    else:
        cache_region.set(key, value)

def get_cache(key: str):
    """
    Get from cache.
    :return: value or None if missing/expired
    """
    val = cache_region.get(key)
    return None if val is NO_VALUE else val

def delete_cache(key: str):
    """
    Invalidate cache entry.
    """
    cache_region.delete(key)