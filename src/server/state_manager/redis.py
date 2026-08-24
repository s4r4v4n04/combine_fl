from pickle import PicklingError
from pickle import dumps as p_dumps
from pickle import loads as p_loads

from redis import Redis
from redis import exceptions as redis_exceptions

from utils.logger import FedLogger


class StateManager:
    def __init__(self, name: str, host: str = "localhost", port: int = 6379) -> None:
        self.logger = FedLogger("0", "STATE_MANAGER")
        self.redis = Redis(host=host, port=port)
        self.name = name
        # self.redis.flushdb()

    def get(self, key):
        """Get a value by key. Returns None if the key does not exist or on connection error."""
        try:
            value = self.redis.hget(self.name, key)
            if value is None:
                return None
            return p_loads(value)
        except redis_exceptions.ConnectionError as e:
            self.logger.error("server.redis.get.exception", str(e))
            return None

    def put(self, key, value):
        """Store a key-value pair. Logs and returns on serialization or connection error."""
        client_id = key.split(".")[0]
        try:
            serialized_value = p_dumps(value)
        except PicklingError as e:
            self.logger.error("server.redis.put.exception", f"Cannot pickle value for key {key}: {e}")
            return
        try:
            self.redis.hset(self.name, key, serialized_value)
            self.redis.sadd(f"keys_{self.name}", client_id)
        except redis_exceptions.ConnectionError as e:
            self.logger.error("server.redis.put.exception", str(e))
        except redis_exceptions.DataError as e:
            self.logger.error("server.redis.put.exception", f"Invalid input type: {e}")

    def keys(self):
        """Return list of client IDs. Returns empty list on connection error."""
        try:
            return [
                i.decode(encoding="utf-8")
                for i in self.redis.smembers(f"keys_{self.name}")
            ]
        except redis_exceptions.ConnectionError as e:
            self.logger.error("server.redis.keys.exception", str(e))
            return []

    def len(self):
        """Return number of stored client IDs. Returns 0 on connection error."""
        try:
            return self.redis.scard(f"keys_{self.name}")
        except redis_exceptions.ConnectionError as e:
            self.logger.error("server.redis.len.exception", str(e))
            return 0

    def clear(self):
        """Delete all keys. Logs on connection error."""
        try:
            self.redis.delete(f"keys_{self.name}")
            keys = [i.decode() for i in self.redis.hkeys(self.name)]
            for key in keys:
                self.redis.hdel(self.name, key)
        except redis_exceptions.ConnectionError as e:
            self.logger.error("server.redis.clear.exception", str(e))

    def deletebykey(self, key):
        """Delete a specific key. Logs on connection error."""
        try:
            self.redis.srem(f"keys_{self.name}", key)
            self.redis.hdel(self.name, key)
        except redis_exceptions.ConnectionError as e:
            self.logger.error("server.redis.deletebykey.exception", str(e))

    def getall(self):
        return self.redis.hgetall(self.name)

    def putall(self, data: dict):
        """Store all key-value pairs from a dict. Logs on connection or data error."""
        for key, value in data.items():
            if isinstance(key, bytes):
                key = key.decode(encoding="utf-8")
            client_id = key.split(".")[0]
            try:
                self.redis.hset(self.name, key, value)
                self.redis.sadd(f"keys_{self.name}", client_id)
            except redis_exceptions.ConnectionError as e:
                self.logger.error("server.redis.putall.exception", str(e))
            except redis_exceptions.DataError as e:
                self.logger.error("server.redis.putall.exception", str(e))
