# <span style="color: #94af08ff;">Developer Manual</span>

A comprehensive guide to the architecture and every Python module in the Flotilla framework. This reference is organized into four sections — **Entrypoints**, **Server**, **Client**, and **ML Frameworks** — to help developers and open-source contributors understand the internal mechanics and extend the system.

## <span style="color: #2AA5B8;">Entrypoints Overview</span>

The three top-level scripts in `src/` serve as the primary entry points into the Flotilla framework. Each one bootstraps a different role in the federated learning system: the coordination server, a training client, or a session launcher.

### `flo_server.py` (`src/flo_server.py`)
The **server entrypoint**. Boots a Flask + Waitress HTTP server that exposes REST endpoints for session management. Internally creates a `FlotillaServerManager` which handles client discovery and training orchestration.
- Parses `server_config.yaml` via `OpenYaML` and optional `--monitor` CLI flag.
- Exposes `POST /execute_command` to start/restore/revive federated sessions.
- Exposes `GET /get_status`, `GET /client_status`, `GET /get_latest_gm`, and `GET /query_past` for monitoring.
- Uses a threading `Event` to enforce single-session execution and tracks the current session ID.
- The `handle_request()` function creates an asyncio event loop, runs the session via `FlotillaServerManager.run()`, and clears state on completion.

### `flo_client.py` (`src/flo_client.py`)
The **client entrypoint**. Bootstraps an edge device that registers with the server, receives model weights, performs local training, and returns updates.
- Reads `client_config.yaml` and resolves or generates a persistent `client_id` (UUID) from `client_info.yaml`.
- Optionally starts a `Monitor` for CPU/RAM/Disk/Network telemetry via the `-m` flag.
- Instantiates `ClientManager` and calls `client.run()`, which starts the gRPC EdgeService and the discovery service, then blocks until interrupted.

### `flo_session.py` (`src/flo_session.py`)
The **CLI session launcher**. A command-line tool that sends HTTP requests to a running Flotilla server to start, query, or download models from federated learning sessions.
- Supports four tasks via `--task`: `start_session`, `get_status`, `query_past`, and `get_latest_gm`.
- **start_session**: Loads a YAML config, generates a session UUID, and POSTs to `/execute_command`. Supports `--restore`, `--revive`, and `--file` flags for session recovery.
- **get_status**: GETs `/get_status` to check if a session is running, idle, or not found.
- **query_past**: GETs `/query_past` to retrieve metrics and config from a completed session stored in the KV store.
- **get_latest_gm**: GETs `/get_latest_gm` and saves the serialized global model weights to disk as a `.pt` file.

## <span style="color: #E8712B;">Server Overview</span>

The `server/` package contains all modules responsible for orchestrating federated learning rounds, managing client state, aggregating model updates, and communicating with clients via gRPC or MQTT.

### `server_manager.py` (`server/server_manager.py`)
The top-level **FlotillaServerManager** class. Created once at startup by `flo_server.py` and lives for the entire server lifetime.
- Initializes a `StateManager` for `client_info` (tracks all registered clients).
- Starts a **discovery thread** — either gRPC (`start_discovery_server`) or MQTT (`ServerMQTTDiscovery`) — based on `discovery_type` config.
- `run()` creates a `FloSessionManager` for each training session, awaits its completion, and renames the log file.
- `get_active_clients()` filters `client_info` to return only clients with `is_active=True`.

### `server_session_manager.py` (`server/server_session_manager.py`)
The **heart of the framework** — the `FloSessionManager` class. This is the largest and most critical module (~1400 lines). It manages the entire lifecycle of a single federated learning session.
- **Initialization**: Creates four `StateManager` instances (`training_session`, `training_state`, `client_selection_state`, `aggregator_state`). Dynamically loads the aggregator, client selection, and termination condition strategies.
- **Session Recovery**: `restore()` resumes from in-memory state; `restore_from_file()` loads from a tarball checkpoint on disk.
- **Backend Resolution**: Resolves the ML framework from the model config via `get_backend_name()` and `get_backend()`, passing it to `ServerModelManager` and through gRPC to clients.
- **Model Management**: Creates a `ServerModelManager` to hold the global model, loss function, optimizer, and validation data.
- **Training Loop**: `start_session()` → `echo()` (connectivity test) → `train()` (multi-round federated training with async gRPC calls to clients).
- **gRPC Operations**: `grpc_send_model()` streams model files to clients with caching. `async_grpc_train()` sends model weights and receives trained updates. `grpc_train_callback()` triggers aggregation after each client response.
- **Checkpointing**: Periodically saves all state to a tarball under the checkpoint directory.

### `server_model_manager.py` (`server/server_model_manager.py`)
The **ServerModelManager** class. Manages the server-side copy of the global model for validation purposes. Supports all registered ML frameworks via the `BackendAdapter` interface.
- Accepts a `backend` parameter and resolves the appropriate framework adapter.
- Dynamically loads the model class from the model directory using `get_model_class()`, then builds it via `backend.build_model()`.
- Supports custom server dataloaders (`CustomDataLoader.get_server_dataloader()`) or falls back to the backend's default `get_server_dataloader()`.
- `validate_model()` delegates to `backend.default_validate_classifier()` and returns accuracy/loss metrics. Supports custom validators via `CustomModelTrainer.validate_model()`.
- `get_model_weights()` / `set_model_weights()` use the backend's `get_weights()` / `set_weights()` for framework-native serialization.
- For PyTorch, loss functions and optimizers are loaded dynamically via the loader modules. Non-PyTorch frameworks manage loss and optimizer internally.

### `server_state_manager.py` (`server/server_state_manager.py`)
The **StateManager** and **ReadOnlyState** classes. A pluggable key-value store abstraction used everywhere in the framework to persist training state.
- Dynamically imports a backend from `server/state_manager/` — supports `inmemory`, `redis`, and `ukv` backends.
- Falls back to `inmemory` if the configured backend fails to initialize.
- Exposes a uniform API: `get()`, `put()`, `keys()`, `len()`, `clear()`, `deletebykey()`, `getall()`, `putall()`.
- `ReadOnlyState` provides a read-only view with a multi-backend fallback chain.

### `server_file_manager.py` (`server/server_file_manager.py`)
Utility functions for file I/O, model loading, dataset discovery, and YAML parsing on the server side.
- `get_model_class()`: Dynamically imports a Python class from a model directory by scanning all `.py` files and matching by class name.
- `get_available_datasets()`: Scans a directory for dataset subdirectories, each containing a `dataset_config.yaml`.
- `get_model_dir_hash()`: Computes a composite SHA-256 hash of all files in a model directory for cache-invalidation during model transfer.
- `OpenYaML()`: Safe YAML loader with optional logger-based error reporting.
- `add_init_file_to_dir()`: Creates `__init__.py` in directories so they become importable Python packages.

### `server_grpc_discovery.py` (`server/server_grpc_discovery.py`)
The **gRPC-based discovery service**. Implements the `DiscoveryServiceServicer` for client registration and heartbeat monitoring.
- `Register()`: Stores client metadata (name, gRPC endpoint, hardware info, datasets, models) into the `client_info` state and signals that at least one client is available.
- `Heartbeat()`: Bidirectional streaming RPC. Receives periodic pings from clients and maintains a sliding window of timestamps.
- `Deregister()`: Marks a client as inactive.
- `heartbeat_alive_check()`: Background thread that periodically scans all clients and marks those that have exceeded the heartbeat miss threshold as inactive.
- `start_discovery_server()`: Entry-point function that starts the gRPC server, waits for at least one client to register, then signals readiness.

### `server_mqtt_discovery.py` (`server/server_mqtt_discovery.py`)
The **MQTT-based discovery service**. An alternative to gRPC discovery for environments where an MQTT broker is preferred.
- `mqtt_ad()`: Connects to the MQTT broker, publishes a server advertisement on `mqtt_server_topic`, and subscribes to `mqtt_client_topic` and `heartbeat`.
- Client advertisements are parsed from JSON payloads and stored in `client_info` state.
- Heartbeat messages are processed similarly to gRPC discovery — a sliding window timestamp check removes stale clients.
- Blocks until at least one client is discovered, then signals the `grpc_event` to allow sessions to proceed.

### Dynamic Loaders (`server/load_*.py`)
Four small loader modules that use `importlib` to dynamically resolve pluggable strategies at runtime:
- **load_aggregator.py**: Imports `server.aggregation.aggregator_{name}` — loads the federated aggregation strategy (e.g., FedAvg, FedProx). Aggregation strategies are implemented for all supported frameworks (e.g., `fedavg_torch`, `fedavg_tensorflow`, `fedavg_jax`, etc.).
- **load_client_selection.py**: Imports `server.clientselection.client_selection_{name}` — loads the client selection policy (e.g., random, round-robin).
- **load_loss.py**: Imports the loss function module. Supports custom losses from `server.loss.loss_function_{name}` or standard PyTorch modules.
- **load_optimizer.py**: Imports the optimizer module. Supports custom optimizers from `server.optimizer.optimizer_{name}` or standard PyTorch optimizers.
- **load_termination_condition.py**: Imports `server.termination.termination_{name}` — determines when to stop training (e.g., max_rounds, accuracy_threshold).
- **load_lifecycle.py**: Imports `server.lifecycle.lifecycle_{name}` — loads a session lifecycle module that can override the default training loop (e.g., `standard`, `odin`, `surgefl`).

All loaders return `None` on failure, allowing the caller to raise descriptive errors.

## <span style="color: #6366f1;">Client Overview</span>

The `client/` package implements the edge-side logic. Each client registers with the server, receives model architecture and weights, trains locally on its private data partition, and returns the updated weights and metrics.

### `client_manager.py` (`client/client_manager.py`)
The top-level **ClientManager** class. Orchestrates client startup, networking, and lifecycle management.
- Configures the PyTorch device (CPU/GPU), detects Docker vs bare-metal networking, and resolves the client's IP address.
- Allocates a gRPC port using `port_allocator()` and configures gRPC options (max message size, reuse settings).
- Scans for available datasets via `get_available_datasets()` and sets up temp directories.
- `discovery_init()`: Starts MQTT or gRPC discovery in a background thread based on `discovery_type`.
- `grpc_init()`: Starts the `ClientEdgeService` gRPC server that listens for training/benchmark/validation requests from the Flotilla server.
- `run()`: Starts both services and blocks on a stop event until `KeyboardInterrupt`.

### `client.py` (`client/client.py`)
The core **Client** class. Contains the high-level logic for benchmark, training, and validation operations.
- Resolves the ML framework per operation via `_backend_name()`, which reads the `backend` field from the model config or model directory. Passes the framework name to `ClientTrainer` and `DataLoader` so all operations use the correct adapter.
- `Benchmark()`: Runs a benchmark training pass to profile the client's hardware capabilities. Uses a timeout or mini-batch count to limit execution. Updates `client_info.yaml` with results.
- `Train()`: Loads a model checkpoint, creates a `ClientTrainer`, selects the appropriate dataloader (custom or default), and executes local training. Returns metrics and updated model weights.
- `Validate()`: Similar to Train but runs inference-only validation on the test set and returns accuracy/loss metrics.
- All three methods support **custom dataloaders** via `CustomDataLoader.get_client_dataloaders()` with automatic fallback to `get_train_test_dataset_loaders()`.
- Caches dataloaders across rounds — only reloads when the `dataset_id` changes.

### `client_edge_service.py` (`client/client_edge_service.py`)
The **ClientEdgeService** gRPC servicer. Implements the `EdgeServiceServicer` interface — the server-facing API that the Flotilla server calls to interact with this client.
- `Echo()`: Simple connectivity check. Returns the received text message back to the server.
- `StreamFile()`: Receives streamed model files from the server and saves them to the local model cache directory.
- `InitBench()`: Deserializes the benchmark request, delegates to `Client.Benchmark()`, and returns mini-batch count and duration.
- `StartTraining()`: Deserializes model weights, config, loss function, and optimizer from the gRPC request. Delegates to `Client.Train()` and returns the serialized updated weights and metrics.
- `StartValidation()`: Similar to training but runs `Client.Validate()` and returns only metrics (no weight updates).

### `client_trainer.py` (`client/client_trainer.py`)
The **ClientTrainer** class. Handles training and validation loops for any supported ML framework on the client.
- Accepts a `backend` parameter and delegates all framework-specific operations to the corresponding `BackendAdapter`.
- Dynamically loads the model class via `get_model_class()` and builds it via `backend.build_model()`.
- `train_model()`: Dispatches to either a **custom trainer** (`CustomModelTrainer.train_model()`) or the backend's `default_train_classifier()`.
- The default training loop is framework-specific (e.g., PyTorch forward-backward, TensorFlow GradientTape, JAX value_and_grad) with configurable termination: timeout, max mini-batches, or max epochs.
- `validate_model()`: Delegates to the backend's `default_validate_classifier()` and returns accuracy/loss metrics. Supports custom validator classes.
- `exit_check()`: Central predicate that evaluates all termination conditions each mini-batch.

### `client_dataset_loader.py` (`client/client_dataset_loader.py`)
The default **DataLoader** class. Delegates dataset loading to the configured ML framework's backend adapter.
- Accepts a `backend` parameter (defaults to `pytorch`) and forwards all loading to `backend.get_train_test_dataset_loaders()`.
- PyTorch loads `.pt` files via `torch.load()`. TensorFlow, JAX, Scikit-Learn, and ONNX load `.npz` files via `numpy.load()`.
- `get_train_loader()` / `get_test_loader()`: Convenience wrappers that return the train or test partition.
- `get_train_test_dataset_loaders()`: Loads a dataset and splits it (typically 95/5) into train and test subsets.

### `client_file_manager.py` (`client/client_file_manager.py`)
File I/O utilities for the client side — mirrors the server's file manager but with client-specific paths and conventions.
- `setup_dir()` / `setup_model_dir()`: Create necessary directories for temp storage and model caching.
- `get_model_class()`: Same dynamic import pattern as the server — scans `model_cache/{model_id}/` for a class matching the given name.
- `get_available_datasets()`: Scans the datasets directory for subdirectories containing `train_dataset_config.yaml`.
- `get_available_models()`: Lists cached models and computes SHA-256 hashes for change detection.
- `OpenYaML()` / `read_yaml()`: Safe YAML parsing with error handling.

### `client_grpc_discovery.py` (`client/client_grpc_discovery.py`)
The **ClientGRPCDiscovery** class. Handles client-side registration and heartbeat streaming when `discovery_type` is `grpc`.
- `run()`: Main loop — connects to the server's discovery endpoint, registers, and streams heartbeats. Automatically reconnects on failure.
- `_register()`: Sends a `ClientRegistration` RPC with client metadata (name, gRPC endpoint, hardware info, datasets, cached models, benchmark results). Retries every 3 seconds until accepted.
- `_heartbeat_loop()`: Generator-based bidirectional stream that sends periodic `HeartbeatPing` messages at the interval dictated by the server's registration response.

### `client_mqtt_discovery.py` (`client/client_mqtt_discovery.py`)
The **ClientMQTTDiscovery** class. Alternative discovery mechanism using MQTT pub/sub.
- `mqtt_sub()`: Connects to the MQTT broker, subscribes to the server's advertisement topic, and responds with the client's own advertisement payload (ID, gRPC endpoint, hardware info, datasets, models).
- After the server is discovered, enters a heartbeat loop that publishes periodic JSON heartbeat messages on the `heartbeat` topic until the stop event is set.
- The heartbeat interval is received from the server's initial advertisement message.

## <span style="color: #E8712B;">ML Frameworks Overview</span>

The `ml_backends/` package provides a framework-agnostic adapter layer that decouples all framework-specific operations (model building, weight serialization, training, validation, data loading) from the rest of Flotilla. This allows the server and clients to work with any supported ML framework without changing core orchestration code.

### `base.py` (`ml_backends/base.py`)
The **BackendAdapter** abstract base class. Defines the contract every framework adapter must implement:
- `get_weights()` / `set_weights()`: Extract or apply model weights as a `WeightPayload`.
- `default_train_classifier()`: Run a default training loop (forward pass, loss, backward pass, optimizer step) with timeout, epoch, and mini-batch termination conditions.
- `default_validate_classifier()`: Run inference and return `{accuracy, loss}` metrics.
- `resolve_device()`: Map device strings (e.g., `cuda`, `cpu`) to framework-specific device handles.
- `build_model()`: Instantiate a model class with flexible constructor signatures.
- `get_train_test_dataset_loaders()`: Load datasets from disk in the framework's native format.

### `weights.py` (`ml_backends/weights.py`)
The **WeightPayload** dataclass. A backend-tagged container that travels through transport (gRPC pickle) and aggregation:
- Fields: `backend` (str), `weights` (Any), `model_id` (optional), `metadata` (dict).
- Helper functions: `unwrap_weights()`, `wrap_weights()`, `get_payload_backend()`, `is_weight_payload()`.

### `registry.py` (`ml_backends/registry.py`)
The **framework registry**. Manages lazy registration and LRU-cached instantiation of adapters:
- `get_backend(name)`: Returns the singleton adapter for a given framework name.
- `get_backend_name(config, model_dir)`: Resolves the framework from training config → model config → model directory `config.yaml`, defaulting to `pytorch`.
- `register_backend(name, cls)`: Register a custom adapter at runtime.
- Built-in aliases: `pytorch`/`torch`, `tensorflow`/`tf`, `jax`, `sklearn`/`scikit_learn`, `onnx`.

### Framework Adapters
Five built-in adapters implement `BackendAdapter`:
- **`pytorch_backend.py`** — Uses `state_dict()`, `torch.no_grad()`, and standard PyTorch `DataLoader`. Datasets are loaded from `.pt` files.
- **`tensorflow_backend.py`** — Uses Keras `get_weights()`/`set_weights()`, `GradientTape` training, `tf.data.Dataset`. Datasets are loaded from `.npz` files.
- **`jax_backend.py`** — Uses `model.params`/`model.apply()`, `jax.value_and_grad()` training, NumPy batch loaders. Datasets are loaded from `.npz` files.
- **`sklearn_backend.py`** — Uses `coef_`/`intercept_`/`classes_` attributes, `partial_fit()` training, NumPy batch loaders. Datasets are loaded from `.npz` files.
- **`onnx_backend.py`** — Wraps `onnxruntime.InferenceSession` for inference or trainable sklearn-based wrappers for training. Datasets are loaded from `.npz` files.
