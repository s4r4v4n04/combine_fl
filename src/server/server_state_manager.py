from importlib import import_module
from uuid import uuid4

from utils.logger import FedLogger

FALLBACK_ORDER = ["ukv", "redis", "inmemory"]


class StateManager:
    def __init__(
        self, loc: str, name: str, host: str, port: int, state_id: str = None
    ) -> None:
        self.state_id = state_id if state_id else str(uuid4())
        self.name = f"{name}_{self.state_id}"
        self.logger = FedLogger("0", "STATE_MANAGER")

        attempt_order = []
        if loc:
            attempt_order.append(loc)
        for backend in FALLBACK_ORDER:
            if backend not in attempt_order:
                attempt_order.append(backend)

        kvstore = None
        for backend in attempt_order:
            try:
                module = import_module(f"server.state_manager.{backend}")
                kvstore = module.StateManager(name=self.name, host=host, port=port)
                if backend != loc:
                    self.logger.warn(
                        "fedserver.state_manager",
                        f"Configured backend '{loc}' unavailable, using '{backend}' instead",
                    )
                break
            except Exception as e:
                self.logger.warn(
                    "fedserver.state_manager",
                    f"Backend '{backend}' failed to initialize: {e}",
                )

        if kvstore is None:
            raise RuntimeError("All storage backends failed to initialize")

        self.get = kvstore.get
        self.get_large = kvstore.get
        self.put = kvstore.put
        self.put_large = kvstore.put
        self.keys = kvstore.keys
        self.len = kvstore.len
        self.clear = kvstore.clear
        self.deletebykey = kvstore.deletebykey
        self.getall = kvstore.getall
        self.putall = kvstore.putall

    def get(self, key):
        raise NotImplementedError

    def get_large(self, key):
        raise NotImplementedError

    def put(self, key, value):
        raise NotImplementedError

    def put_large(self, key, value):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def len(self):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def deletebykey(self, key):
        raise NotImplementedError

    def getall(self):
        raise NotImplementedError

    def putall(self):
        raise NotImplementedError


class ReadOnlyState:
    def __init__(self, loc: str, name: str, host: str, port: int) -> None:
        self.state_id = str(uuid4())
        self.logger = FedLogger("0", "STATE_MANAGER")
        full_name = f"{name}_{self.state_id}"

        attempt_order = []
        if loc:
            attempt_order.append(loc)
        for backend in FALLBACK_ORDER:
            if backend not in attempt_order:
                attempt_order.append(backend)

        kvstore = None
        for backend in attempt_order:
            try:
                module = import_module(f"server.state_manager.{backend}")
                kvstore = module.StateManager(name=full_name, host=host, port=port)
                if backend != loc:
                    self.logger.warn(
                        "fedserver.state_manager",
                        f"Configured backend '{loc}' unavailable, using '{backend}' instead",
                    )
                break
            except Exception as e:
                self.logger.warn(
                    "fedserver.state_manager",
                    f"Backend '{backend}' failed to initialize: {e}",
                )

        if kvstore is None:
            raise RuntimeError("All storage backends failed to initialize")

        self.get = kvstore.get
        self.get_large = kvstore.get
        self.keys = kvstore.keys
        self.len = kvstore.len

    def get(self, key):
        raise NotImplementedError

    def get_large(self, key):
        raise NotImplementedError

    def keys(self):
        raise NotImplementedError

    def len(self):
        raise NotImplementedError
