import json
try:
    import redis.asyncio as redis
except ImportError:
    redis = None
from typing import Any, Optional, List, Dict
from phoenix.services.cache.base import BaseCache
from phoenix.core.config import config
from phoenix.services.observability.logger import get_logger

logger = get_logger("Phoenix AI.Cache.Redis")

class RedisCache(BaseCache):
    def __init__(self):
        self.redis = None
        self._failed = False

    async def init(self):
        if redis is None:
            logger.warning("redis library is not installed. Caching disabled.")
            self._failed = True
            return
            
        try:
            self.redis = redis.from_url(config.REDIS_URL, decode_responses=True)
            # Ping to verify connection
            await self.redis.ping()
            self._failed = False
        except Exception as e:
            logger.warning(f"Failed to connect to Redis at {config.REDIS_URL}: {e}. Caching disabled.")
            self.redis = None
            self._failed = True

    async def get(self, key: str) -> Optional[Any]:
        if self._failed:
            return None
            
        if not self.redis:
            await self.init()
            if self._failed: return None
            
        try:
            value = await self.redis.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
        except Exception:
            self._failed = True
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if self._failed:
            return
            
        if not self.redis:
            await self.init()
            if self._failed: return
            
        try:
            serialized_value = json.dumps(value)
            kwargs = {}
            if ttl is not None:
                kwargs["ex"] = ttl
            await self.redis.set(key, serialized_value, **kwargs)
        except Exception:
            self._failed = True

    async def delete(self, key: str) -> None:
        if self._failed:
            return
            
        if not self.redis:
            await self.init()
            if self._failed: return
            
        try:
            await self.redis.delete(key)
        except Exception:
            self._failed = True

    async def keys(self, pattern: str = "*") -> List[str]:
        if self._failed:
            return []
            
        if not self.redis:
            await self.init()
            if self._failed: return []
            
        try:
            matched_keys = []
            async for key in self.redis.scan_iter(match=pattern):
                matched_keys.append(key)
            return matched_keys
        except Exception:
            self._failed = True
            return []

    async def get_all(self, pattern: str = "*") -> Dict[str, Any]:
        if self._failed:
            return {}
            
        if not self.redis:
            await self.init()
            if self._failed: return {}
            
        try:
            matched_keys = await self.keys(pattern)
            if not matched_keys:
                return {}
                
            results = {}
            values = await self.redis.mget(matched_keys)
            for key, val in zip(matched_keys, values):
                if val is not None:
                    try:
                        results[key] = json.loads(val)
                    except json.JSONDecodeError:
                        results[key] = val
            return results
        except Exception:
            self._failed = True
            return {}

    async def update(self, key: str, value: Dict[str, Any]) -> None:
        if self._failed:
            return
            
        if not self.redis:
            await self.init()
            if self._failed: return

        try:
            current = await self.get(key)
            if isinstance(current, dict):
                current.update(value)
                await self.set(key, current)
            else:
                await self.set(key, value)
        except Exception:
            self._failed = True

# sk-0b7a5da865e34b92b0f511b0af57b52d