from typing import Any, Optional
import hashlib

class CacheTool:
    """Tool for caching responses and classifications"""
    
    def __init__(self):
        self.cache = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value"""
        self.cache[key] = value
    
    def generate_key(self, prefix: str, content: str) -> str:
        """Generate cache key"""
        content_hash = hashlib.md5(content.lower().strip().encode()).hexdigest()[:8]
        return f"{prefix}_{content_hash}"
    
    def clear(self) -> None:
        """Clear all cache"""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        return len(self.cache)