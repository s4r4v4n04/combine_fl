import hashlib
import json
import os
import threading
from pickle import PicklingError
from pickle import dumps as p_dumps
from pickle import loads as p_loads

from ukv import rocksdb

from utils.logger import FedLogger

_db_lock = threading.Lock()
_db_instance = None
_db_path = None


def _get_db(db_dir: str):
    """
    Return a module-level singleton RocksDB handle.
    RocksDB holds an exclusive lock on the directory, so all
    StateManager instances in the same process must share one handle.
    """
    global _db_instance, _db_path
    with _db_lock:
        if _db_instance is not None and _db_path == db_dir:
            return _db_instance
        if _db_instance is not None:
            try:
                _db_instance.close()
            except Exception:
                pass
        os.makedirs(db_dir, exist_ok=True)
        config = json.dumps({"version": "1.0", "directory": db_dir})
        _db_instance = rocksdb.DataBase(config)
        _db_path = db_dir
        return _db_instance


def _str_to_int(key: str) -> int:
    """Deterministic string → int64 hash. Reserves 0 for the key registry."""
    h = hashlib.sha256(key.encode()).digest()[:8]
    val = int.from_bytes(h, "big") & 0x7FFFFFFFFFFFFFFF
    if val == 0:
        val = 1
    return val


_DEFAULT_DB_DIR = "/tmp/ukv_flotilla"
_REGISTRY_INT_KEY = 0


class StateManager:
    def __init__(self, name: str, host: str = None, port: int = None) -> None:
        self.logger = FedLogger("0", "STATE_MANAGER")
        self.name = name

        db_dir = host if (host and os.sep in host) else _DEFAULT_DB_DIR
        self.db = _get_db(db_dir)
        self._col = self.db[self.name]
        self._load_registry()

    def _load_registry(self):
        try:
            data = self._col[_REGISTRY_INT_KEY]
            if data is None:
                self._registry = {}
            else:
                self._registry = p_loads(bytes(data))
        except (KeyError, Exception):
            self._registry = {}

    def _save_registry(self):
        self._col[_REGISTRY_INT_KEY] = p_dumps(self._registry)

    def get(self, key):
        int_key = _str_to_int(key)
        try:
            value = self._col[int_key]
            if value is None:
                return None
            return p_loads(bytes(value))
        except KeyError:
            return None
        except Exception as e:
            self.logger.error("fedserver.ukv.get", str(e))
            return None

    def put(self, key, value):
        try:
            serialized = p_dumps(value)
        except PicklingError:
            self.logger.error("fedserver.ukv.put", f"{value} cannot be pickled")
            return
        try:
            int_key = _str_to_int(key)
            self._col[int_key] = serialized
            self._registry[key] = int_key
            self._save_registry()
        except Exception as e:
            self.logger.error("fedserver.ukv.put", str(e))

    def keys(self):
        return list(set(k.split(".")[0] for k in self._registry))

    def len(self):
        return len(set(k.split(".")[0] for k in self._registry))

    def clear(self):
        try:
            self._col.clear()
            self._registry.clear()
        except Exception as e:
            self.logger.error("fedserver.ukv.clear", str(e))
            self._registry.clear()

    def deletebykey(self, key):
        try:
            int_key = _str_to_int(key)
            try:
                del self._col[int_key]
            except KeyError:
                pass
            self._registry.pop(key, None)
            self._save_registry()
        except Exception as e:
            self.logger.error("fedserver.ukv.delete", str(e))

    def getall(self):
        result = {}
        for str_key, int_key in self._registry.items():
            try:
                value = self._col[int_key]
                if value is not None:
                    result[str_key] = bytes(value)
            except KeyError:
                pass
        return result

    def putall(self, data: dict):
        try:
            self._col.clear()
        except Exception:
            pass
        self._registry.clear()
        for key, value in data.items():
            if isinstance(key, bytes):
                key = key.decode("utf-8")
            int_key = _str_to_int(key)
            try:
                self._col[int_key] = value if isinstance(value, bytes) else p_dumps(value)
                self._registry[key] = int_key
            except Exception as e:
                self.logger.error("fedserver.ukv.putall", str(e))
        self._save_registry()
