# <span style="color: #94af08ff;">User Guide</span>

A comprehensive guide for setting up, configuring, and running federated learning experiments using the Flotilla framework. This document covers installation, configuration files, dataset management, and command-line execution for both local and distributed environments.

### 1.1 Features

**Extensibility**

Flotilla is designed so you can swap or extend major components without rewriting the full stack:

- **Aggregation strategies** combine client updates into a single global model update. Built-in algorithms include FedAvg (synchronous averaging), FedAsync (asynchronous aggregation), and FedAT (asynchronous tiered aggregation). You can add more by placing modules under `src/server/aggregation/`.
- **Client selection strategies** choose which clients participate in each round. Many policies are included (for example all clients, random subsets, loss-based selection, TiFL, HACCS, and strategies matched to FedAsync/FedAT). You can add more under `src/server/clientselection/`.
- **Session termination conditions** decide when to stop training. Built-in strategies include a fixed round cap, convergence on validation loss, a target accuracy, a loss threshold, and an accuracy plateau. You can add custom logic under `src/server/termination/`.

| Strategy           | Description |
|--------------------|-------------|
| `max_rounds`       | Stop after `termination_condition_args.max_rounds`. |
| `convergence`      | Stop when validation loss has not improved for `patience` rounds. |
| `accuracy_target`  | Stop when validation accuracy reaches `target_accuracy`. |
| `loss_threshold`   | Stop when validation loss is at or below `loss_threshold`. |
| `accuracy_plateau` | Stop when accuracy has not improved for `patience` rounds. |

Example training configs for different termination settings live alongside `config/training_config.yaml` (for example `training_config_convergence.yaml`, `training_config_accuracy_target.yaml`).

- **State backends** persist or hold orchestration state on the server. You can use in-memory storage for development, or **Redis** / **UKV** for durability and features such as querying past session metadata (when configured).
- **Custom models, trainers, validators, and dataloaders** let you bring your own models in any supported framework (PyTorch, TensorFlow, JAX, Scikit-Learn, or ONNX) and optional training or data-loading code. Built-in models live under `models/` and are copied to `src/models/` on the server for training.
- **Checkpointing, restore, and revive** save progress at a configurable interval. **Restore** resumes a session from stored state in the chosen backend; **revive** re-initializes from stored configuration and continues; **file** can load from an on-disk checkpoint artifact. These are important for long runs and recovery after failures. Use `flo_session.py` with `--restore`, `--revive`, or `--file` together with `--session_id` (see [Checkpointing and recovery](#8-checkpointing-and-recovery)).
- **Communication backends** let you run client discovery and coordination over **gRPC** (default) or **MQTT** (for example Eclipse Mosquitto). Set `discovery_type` consistently in `server_config.yaml` and `client_config.yaml`.
- **Differential privacy** is illustrated in the repository: **DP-SCAFFOLD**-style training (`examples/dp-scaffold`) and **Opacus** on the client (`examples/dp-opacus`), configured via custom trainer arguments.
- **ML framework support** allows you to train and validate models using the framework of your choice. Flotilla ships with built-in support for **PyTorch**, **TensorFlow/Keras**, **JAX**, **Scikit-Learn**, and **ONNX Runtime**. Each framework is implemented as an adapter under `src/ml_backends/` and provides default training loops, validation loops, and dataset loaders. Select the framework by setting `backend` in your model's `config.yaml`. You can register custom frameworks via `register_backend()`.
- **Session lifecycle modules** control the high-level orchestration flow. The default lifecycle runs the standard round-based training loop, but pluggable alternatives like **Odin** and **SurgeFL** are included. Add custom lifecycles under `src/server/lifecycle/`.

**Ease of use**

Flotilla is easy to set up and use even if you are new to federated learning. It provides a simple workflow: configure YAML files, start the server and clients, then start a session with `flo_session.py`. The server exposes HTTP endpoints for starting runs and querying status; see [Running experiments](#7-running-experiments).

**Portability**

Flotilla can run on a wide range of devices, from Raspberry Pis to GPU workstations, including NVIDIA Jetson with a platform-appropriate PyTorch build. It can also run in Docker with separate server and client images and a compose file for MQTT and Redis.

**Framework-agnostic**

Flotilla is fully **framework-agnostic**. The ML framework adapter layer (`ml_backends/`) provides first-class support for **PyTorch**, **TensorFlow/Keras**, **JAX**, **Scikit-Learn**, and **ONNX Runtime**. The framework is selected per model via the `backend` key in the model's `config.yaml`, and all built-in strategies (aggregation, training, validation, data loading) work across every supported framework.

---

## <span style="color: #2AA5B8;">Installation</span>

### 2.1 Bare metal

For a distributed setup where the server and each client run on different machines, install Python and build tools (for example GCC) as needed, then:

1. Clone the repository on each machine that will run the server or a client:

    ```bash
    git clone https://github.com/dream-lab/fedml-ng.git
    cd fedml-ng
    ```

2. Install dependencies. Server and client use separate requirement files:

    ```bash
    pip install -r src/server/requirements.txt   # on the server machine
    pip install -r src/client/requirements.txt   # on each client machine
    ```

3. **PyTorch** is not pinned inside those requirement files. Install PyTorch for your platform and CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

    **PyTorch 2.7.1 is compatible with the dependency set in `src/server/requirements.txt` and `src/client/requirements.txt`** (for example `numpy==1.24.3`). Install it explicitly, for example:

    ```bash
    pip install torch==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu124
    ```

    Adjust the index URL or packages for CPU-only or other CUDA builds.

4. Install optional ML framework dependencies based on which frameworks your models use:

    - **TensorFlow** `2.14.x`: `pip install tensorflow==2.14.0`
    - **JAX** `0.4.28` with matching jaxlib: `pip install jax==0.4.28 jaxlib==0.4.28`
    - **Scikit-Learn**: `pip install scikit-learn`
    - **ONNX Runtime**: `pip install onnxruntime`

    Validate versions in an isolated environment; upgrading NumPy beyond what the requirements files pin may break pinned packages.

5. Regenerate gRPC Python stubs to match your Python version:

    ```bash
    cd src/proto && bash run.sh
    ```

For **NVIDIA Jetson**, install PyTorch and torchvision builds built for your JetPack from [NVIDIA’s PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048).

### 2.2 Docker

1. Clone the repository and enter the project root.

2. Build images:

    ```bash
    docker build -f docker/Dockerfile.server -t flotilla-server .
    docker build -f docker/Dockerfile.client -t flotilla-client .
    ```

3. Create a bridge network (name used below matches `docker-compose`):

    ```bash
    docker network create -d bridge flotilla-network
    ```

4. Start MQTT and Redis (adjust host ports in `docker/docker-compose.yml` if needed):

    ```bash
    cd docker && docker-compose up -d
    ```

5. **Client containers** — set `--network` (for example `flotilla-network`), bind mounts for client data to `/src/data` and logs to `/src/logs`, optional `--memory` and `--cpuset-cpus`, and environment variables for `MQTT_IP` / `MQTT_PORT` (and gRPC discovery if used). See `docker/sample_docker_client_run.sh` as a template.

6. **Server container** — set `--network`, environment variables `REDIS_IP`, `REDIS_PORT`, `MQTT_IP`, `MQTT_PORT` to match where those services run, publish REST (for example `-p 12345:12345`), and bind mounts for server logs (`/src/logs`), validation data (`/src/val_data`), checkpoints (`/src/checkpoints` or the path in `server_config.yaml`), and model definitions (`/src/models`). See `docker/sample_docker_server_run.sh`.

7. For clients across multiple physical hosts, use **Docker Swarm** (or equivalent) with an **overlay** network so containers can reach the broker and server discovery endpoint.

---

## <span style="color: #2AA5B8;">Configuration</span>

Templates and documentation live under `config/`. For a run, copy the server and client YAML plus logging config into `src/config/` on each host:

```bash
mkdir -p src/config
cp config/server_config.yaml src/config/    # on server host
cp config/client_config.yaml src/config/    # on client hosts
cp config/logger.conf src/config/            # on all relevant hosts
```

Edit the copies so broker addresses, discovery endpoints, dataset paths, and state backend match your deployment.

### 3.1 `training_config.yaml`

**Purpose:** Defines the federated learning session: aggregation and client selection, when to stop training, benchmarking, server and client training settings, and model-related flags.

**Main blocks:**

- **`session_config`** — `aggregator`, `client_selection`, optional `aggregator_args` / `client_selection_args`, `termination_condition` / `termination_condition_args`, `validation_round_interval`, `checkpoint_interval`, `generate_plots`, optional `wait_for_clients` (`min_clients`, `max_wait_s`, `poll_interval_s`), `use_gpu` for server-side work where needed, optional `session_lifecycle` (pluggable orchestration flow, e.g. `odin`, `surgefl`).
- **`benchmark_config`** — Whether to skip benchmarking (`skip_benchmark`), which model and dataset to use for device benchmarking, minibatch count, batch size, learning rate, and timeout.
- **`server_training_config`** — `model_dir`, optional `validation_dataset`, `global_model_validation_batch_size`.
- **`client_training_config`** — `model_id`, `model_class`, `dataset`, `epochs`, `batch_size`, `learning_rate`, `train_timeout_duration_s`, `loss_function`, `loss_function_custom`, `optimizer`, `optimizer_custom`.
- **`model_config`** — Toggles and arguments for custom client/server dataloaders, custom trainer, custom validator, and **`model_args`** (for example `num_classes`).

**Example** (see `config/training_config.yaml` in the repo for the canonical template):

```yaml
session_config:
  session_id: dev
  use_gpu: True
  aggregator: fedavg_torch
  aggregator_args: None
  client_selection: fedavg
  client_selection_args:
    client_fraction: 1.0
  termination_condition: max_rounds
  termination_condition_args:
    max_rounds: 20
  validation_round_interval: 1
  checkpoint_interval: 1
  generate_plots: False
  wait_for_clients:
    min_clients: 1
    max_wait_s: 3600
    poll_interval_s: 10

benchmark_config:
  skip_benchmark: True
  model_id: LeNet5
  model_dir: ../models/LeNet5
  model_class: LeNet5_class
  dataset: EMNIST_NONIID3
  bench_minibatch_count: 500
  batch_size: 16
  learning_rate: 0.001
  timeout_duration_s: 120

server_training_config:
  model_dir: ../models/LeNet5
  global_model_validation_batch_size: 100

client_training_config:
  model_id: LeNet5
  model_class: LeNet5_class
  epochs: 1
  dataset: EMNIST_NONIID3
  batch_size: 16
  learning_rate: 0.001
  train_timeout_duration_s: 300
  loss_function: crossentropy
  loss_function_custom: True
  optimizer: adam
  optimizer_custom: True

model_config:
  use_custom_client_dataloader: False
  custom_client_loader_args: None
  use_custom_server_dataloader: False
  custom_server_loader_args: None
  use_custom_trainer: False
  custom_trainer_args: None
  use_custom_validator: False
  custom_validator_args: None
  model_args:
    num_classes: 10
```

**Setup:** Point `model_dir` and `dataset` IDs at directories you placed under `src/models/` and `src/data/` / `src/val_data/`. Use the variant YAML files in `config/` for different termination policies or datasets.

### 3.2 `server_config.yaml`

**Purpose:** Communication mode (MQTT vs gRPC), discovery and heartbeat tuning, gRPC limits, REST API bind address, state backend, and paths for checkpoints, validation data, and scratch.

**Example structure:**

```yaml
comm_config:
  discovery_type: grpc  # "mqtt" or "grpc"
  mqtt_discovery:
    type: server
    broker_host: <broker_host>
    broker_port: 1884
    mqtt_sub_timeout_s: 10
    mqtt_server_topic: advert_server
    mqtt_client_topic: advert_client
    
  grpc_discovery:
    port: 50051
  grpc_runtime:
    max_message_length: 1048576000
    chunk_size_bytes: 1024
    timeout_s: 3600
  restful:
    host: 0.0.0.0
    port: 12345
state:
  state_location: redis  # "inmemory", "redis", or "ukv"
  state_hostname: <redis_host>
  state_port: 6379
checkpoint_dir_path: ./checkpoints
validation_data_dir_path: <path_to_val_data>
temp_dir_path: ./scratch
```

**Setup:** Set `discovery_type` and either MQTT broker fields or gRPC `grpc_discovery` / `grpc_runtime` sections. Ensure `restful` matches how you call `flo_session.py`. For production recovery and `query_past`, use **Redis** or **UKV** instead of `inmemory`.

### 3.3 `client_config.yaml`

**Purpose:** Must mirror the server’s `discovery_type`. Configures MQTT client settings or gRPC discovery endpoint, gRPC worker and port settings, dataset root on the client, and general cleanup and GPU options.

**Example structure:**

```yaml
comm_config:
  discovery_type: grpc
  mqtt_discovery:
    type: client
    client_name: <unique_client_name>
    broker_host: <broker_host>
    broker_port: 1884
    mqtt_sub_timeout_s: 30
    mqtt_server_topic: advert_server
    mqtt_client_topic: advert_client
    heartbeat_timeout_s: 15
  grpc_discovery:
    host: <server_host>
    port: 50051
  grpc_runtime:
    workers: 8
    sync_port: 50053
    async_port: 50052
dataset_config:
  datasets_dir_path: <path_to_client_data_root>
general_config:
  temp_dir_path: temp
  cleanup_model_cache_on_exit: False
  cleanup_temp_on_exit: False
  use_gpu: True
```

**Setup:** Set `server_discovery_endpoint` to the machine where the server’s discovery port listens. Point `datasets_dir_path` at the parent of per-dataset folders (each dataset folder contains data files and a `train_dataset_config.yaml` as described in [Datasets](#6-datasets)).

### 3.4 `logger.conf`

**Purpose:** Python `logging` configuration: named loggers for server and client components, handlers (file and stream), formatters, and log levels.

**Setup:** Copy `config/logger.conf` to `src/config/logger.conf`. Adjust levels or handler paths if you change log file locations. Individual loggers (for example `SESSION_MANAGER`, `SERVER_MANAGER`) route to file or console handlers defined in the same file.

---

## <span style="color: #2AA5B8;">Quick start</span>

After completing [Installation](#2-installation) and placing configs under `src/config/`:

1. Download and unpack sample data (or use your own per [Datasets](#6-datasets)):

    ```bash
    wget "https://www.dropbox.com/scl/fi/2oozukii5zvkt8643iohk/flotilla_quicksetup_data.zip?rlkey=1nkwzif43zpu3o9h46ismrxe2&st=7wa25qdm&dl=0" -O flotilla_quicksetup_data.zip
    unzip flotilla_quicksetup_data.zip
    mv data src/
    mv val_data src/
    ```

2. Copy the quickstart training config’s model to the server layout if needed (`cp -r models/FedAT_CNN src/models/`, etc.) and align `training_config` paths with your tree.

3. Start MQTT and Redis (if using MQTT mode):

    ```bash
    cd docker && docker-compose up -d
    ```

4. In one terminal, start a client:

    ```bash
    cd src && python flo_client.py
    ```

5. In another terminal, start the server:

    ```bash
    cd src && python flo_server.py
    ```

6. In a third terminal, start the session (replace the config path and endpoint with yours):

    ```bash
    cd src && python flo_session.py --task start_session --config_path ../config/flotilla_quicksetup_config.yaml --federated_server_endpoint localhost:12345
    ```

The server must see at least one active client before accepting `start_session`.

---

## <span style="color: #2AA5B8;">Models</span>

This folder contains various models that the server can use for training or benchmarking. Each model is organized in a separate directory named after its `model_id`, and it includes a Python file defining the `Model` class and a `config.yaml` file with model-specific details. If a model needs a custom `Training` or `Validation` function provide them in the `trainer.py` file. If a model needs a custom `DataLoader` function provide the function in the `loader.py` file.

#### For custom files use the provided naming convention. (It might work with other names, but it's not guaranteed).

### Model Directory Structure

The models in this folder follow the following directory structure:

```text
models/
|-- model_id/
| |-- model.py
| |-- config.yaml
|-- another_model_id/
| |-- model.py
| |-- trainer.py     # If custom train/validation functions are provided
| |-- loader.py      # If custom dataloader function is provided
| |-- config.yaml
|-- ...
```


### Model Configuration

Each model directory contains a `config.yaml` file, which provides specific details about the model and its training configuration.

### `config.yaml`

The `config.yaml` file contains the following information:

- `model_details`:
  - `model_id`: A unique identifier for the model.
  - `model_class`: Name of the Python class defining the model in `model.py` file.
  - `backend`: The ML framework to use for this model (e.g., `pytorch`, `tensorflow`, `jax`, `sklearn`, `onnx`). Defaults to `pytorch` if omitted.
  - `model_tags`: A list of tags associated with the model. These tags help identify the model's characteristics or type, e.g., CNN, RNN, etc.
  - `suitable_datasets`: A list of dataset IDs that are suitable for training or benchmarking with this model.

- `training_config`:
  - `use_custom_dataloader`: A boolean flag to specify if a custom implementation of DataLoader is provided. If `False` the framework will use the default implementation of the DataLoader
    - `custom_dataloader_args`: If `use_custom_dataloader` is set to `True`, provide all the arguments that the Training function takes.
  - `use_custom_trainer`: A boolean flag to specify if a custom implementation of Training function for the model is provided.
    - `custom_trainer_args`: If `use_custom_trainer` is set to `True`, provide all the arguments that the Training function takes.
  - `use_custom_validator`: A boolean flag to specify if a custom implementatin of Validation functon for the model is provided.
    - `custom_validator_args`: If `use_custom_validator` is set to `True`, provide all the arguments that the Validation function takes.
  - `model_args`: All the arguments needed to initialize the model, for example the `num_classes` for the number of classes to train on.

Below is an example of a [`config.yaml`](models/AlexNet/config.yaml) file for the model named AlexNet:

```yaml
  model_details:
  model_id: AlexNet
  model_class: AlexNet_class
  backend: pytorch
  model_tags: [CNN]
  suitable_datasets: [MNIST, FMNIST]

training_config:
  use_custom_dataloader: False
  custom_loader_args: None

  use_custom_trainer: False
  custom_trainer_args: None

  use_custom_validator: False
  custom_validator_args: None

  model_args:
    num_classes: 10
```

### Built-in model directories

Shipped under `models/` include: **AlexNet**, **AlexNet_MNIST**, **FedAT_CNN**, **LeNet5**, **LSTM**, **LSTM-B**, **MobileNet**, **ResNet18**, **VGG** (PyTorch), as well as models for other frameworks such as **JAX_MLP_MNIST**, **Keras_MLP_MNIST**, **ONNX_MLP**, and **Sklearn_LogReg**. Copy the folder you need to `src/models/` on the server and reference its `model_id` in your training configuration.

**Note:** Training YAML at the project level may use `use_custom_client_dataloader` / `use_custom_server_dataloader` in `model_config`; the per-model `config.yaml` inside `models/` may still use `use_custom_dataloader` — keep the two in sync when you add custom loaders.

---

## <span style="color: #2AA5B8;">Datasets</span>

### 6.1 Server validation data

Place validation data under `src/val_data/<validation_dataset_id>/`. Add a `dataset_config.yaml` beside the data file:

```yaml
dataset_details:
  data_filename: <path_to_the_data_file>
  dataset_id: <validation_dataset_id>
  dataset_tags: <dataset_tags or None>
  suitable_models: <list_of_model_ids or None>
metadata:
  label_distribution: <dictionary_of_label_distributions or None>
  num_items: <number_of_datapoints_in_validation_dataset>
```

### 6.2 Client training data

Place each client’s training shard under `src/data/<train_dataset_id>/` with a `train_dataset_config.yaml`:

```yaml
dataset_details:
  data_filename: <path_to_the_data_file>
  dataset_id: <train_dataset_id>
  dataset_tags: <dataset_tags or None>
  suitable_models: <list_of_model_ids or None>
metadata:
  label_distribution: <dictionary_of_label_distributions or None>
  num_items: <number_of_datapoints_in_train_dataset>
```

The `dataset_id` must match the `dataset` field in `client_training_config` (and any `validation_dataset` you set on the server).

### 6.3 Offline partitioning

Use `src/utils/data_partitioner.py` to build IID or non-IID partitions and `src/utils/get_data_summary.py` to inspect label distributions. Additional example training configs in `config/` target specific splits (for example EMNIST non-IID, CIFAR-10 variants).

### 6.4 Dataset formats by framework

PyTorch models expect `.pt` files loaded via `torch.load()`. Models using TensorFlow, JAX, Scikit-Learn, or ONNX expect `.npz` files (NumPy archive) containing arrays named `x`/`y` or `x_train`/`y_train`/`x_test`/`y_test`. The `dataset_config.yaml` structure is the same for all frameworks — only the `data_filename` extension differs.

---

## <span style="color: #2AA5B8;">Running experiments</span>

1. Ensure MQTT broker and Redis (if used) are running when using those backends (`cd docker && docker-compose up -d`).

2. Copy `server_config.yaml`, `client_config.yaml`, and `logger.conf` to `src/config/` on the appropriate hosts and edit paths and endpoints.

3. Place the model under `src/models/<model_id>/` on the server.

4. Place validation data under `src/val_data/` and client training data under `src/data/` as in [Datasets](#6-datasets).

5. Start the server:

    ```bash
    cd src && python flo_server.py
    ```

    Optional hardware monitoring:

    ```bash
    cd src && python flo_server.py --monitor
    ```

6. Start each client:

    ```bash
    cd src && python flo_client.py
    ```

    Optional monitoring:

    ```bash
    cd src && python flo_client.py -m
    ```

7. Start a training session from a machine that can reach the server REST port:

    ```bash
    cd src && python flo_session.py --task start_session --config_path ../config/training_config.yaml --federated_server_endpoint <server_host>:12345
    ```

8. Query status of a run:

    ```bash
    cd src && python flo_session.py --task get_status --session_id <uuid> --federated_server_endpoint <server_host>:12345
    ```

9. Query a completed session stored in the state backend (typically with Redis or UKV):

    ```bash
    cd src && python flo_session.py --task query_past --session_id <uuid> --federated_server_endpoint <server_host>:12345
    ```

10. Download the latest global model for an active session:

    ```bash
    cd src && python flo_session.py --task get_latest_gm --session_id <uuid> --save_path ./latest_gm --federated_server_endpoint <server_host>:12345
    ```

---

## <span style="color: #2AA5B8;">Checkpointing and Recovery</span>

- Checkpoints are written every `checkpoint_interval` rounds to `checkpoint_dir_path` from `server_config.yaml`.
- To resume or attach to a saved session, use `--session_id` with one of:

| Flag         | Role |
|--------------|------|
| `--restore`  | Restore session state from the configured state backend. |
| `--revive`   | Revive using stored session id and configuration semantics as implemented by the server. |
| `--file`     | Restore from an on-disk checkpoint file when the server supports that path. |

Example:

```bash
cd src && python flo_session.py --task start_session --config_path ../config/training_config.yaml \
  --restore --session_id <existing_session_uuid> --federated_server_endpoint <server_host>:12345
```

Use **`inmemory` state** only when you do not need persistence across server restarts; **`redis`** or **`ukv`** is appropriate for production and for querying past sessions.

---

## <span style="color: #2AA5B8;">Device benchmarking</span>

When `benchmark_config.skip_benchmark` is `False`, the server can run a short training benchmark on clients (model, dataset, minibatch count, batch size, learning rate, timeout defined in YAML). Results characterize device speed and can inform client selection policies. Set `skip_benchmark: True` to skip this phase.

---

## <span style="color: #2AA5B8;">Hardware monitoring</span>
Pass **`--monitor`** (or **`-m`** on the client) to log CPU, RAM, disk I/O, and network I/O during runs:

```bash
python flo_server.py --monitor
python flo_client.py -m
```

Monitoring is off by default.

---

## <span style="color: #2AA5B8;">Supported Platforms</span>

- **Python** 3.6.15 and above (see repository note in `README.md`); prefer a supported 3.10+ environment for current PyTorch builds.
- **PyTorch** 2.7.1 alongside the pinned requirements (install separately).
- **TensorFlow** 2.14.x, **JAX** 0.4.28, **Scikit-Learn**, and **ONNX Runtime** as optional ML framework dependencies.
- **Raspberry Pi**, **GPU workstations**, **NVIDIA Jetson** (Jetson needs a Jetson-built PyTorch).
- **Docker** on x86_64 (and analogous workflows on ARM where images exist).
- **MQTT** via Eclipse Mosquitto (**docker-compose** in `docker/`).
- **Redis** for durable session state (**docker-compose** in `docker/`).

