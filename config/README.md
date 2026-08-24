# Flotilla Configuration

This directory contains the configuration files and logger setup for Flotilla. These configuration files define various settings and parameters required for different aspects of the project. Below are detailed explanations of each configuration file:

## Templates

Reusable templates live under [`templates/`](templates/):

- [`server_config.template.yaml`](templates/server_config.template.yaml): server communication, state, checkpoint, validation data, and temp directory settings.
- [`client_config.template.yaml`](templates/client_config.template.yaml): client discovery, heartbeat, gRPC service, dataset root, GPU, and cleanup settings.
- [`session_config.template.yaml`](templates/session_config.template.yaml): federated session, benchmark, training, backend, model, and dataloader settings.

Copy a template, replace placeholder values such as `<path_to_model_dir>`, and pass the copied file to the relevant server, client, or session command.

## 1. [training_config.yaml](training_config.yaml)

This file contains the configuration settings for the training session, benchmark, and training process.

### `session_config`:

- `session_id`: A unique identifier for the training session, which helps track and manage different training runs.
- `aggregator`: The type of aggregator used during federated learning. Set to `None` for default aggregation.
- `client_selection`: The client selection method used in federated learning. This determines how clients are selected to participate in each training round. Possible values include 'default', 'random', or custom selection strategies.
- `percentage_client_selection`: The percentage of clients selected in each training round when using random client selection.
- `wait_for_clients` (optional): When the number of active clients is below the minimum, the server waits (with an event-driven callback) until enough clients are available or a timeout is reached. If omitted, defaults are applied (min_clients=1, no max_wait).
  - `min_clients`: Minimum number of active clients required to run a round (default: 1).
  - `max_wait_s`: Total time in seconds to wait for sufficient clients before aborting the session (default: null, i.e. wait indefinitely).
  - `poll_interval_s`: Interval in seconds for timeout checks while waiting (default: 10).
- `termination_condition`: Name of the strategy that decides when to stop training (e.g. `max_rounds`, `convergence`, `accuracy_target`, `loss_threshold`, `accuracy_plateau`). Default is `max_rounds`. See [config/examples/README_termination.md](examples/README_termination.md) and the example configs in `config/examples/` for templates.
- `termination_condition_args`: Optional dictionary of arguments for the termination condition (e.g. `max_rounds`, `patience`, `target_accuracy`, `loss_threshold`, `min_rounds`, `check_interval`). For `max_rounds`, this must include `max_rounds`.

### `benchmark_config`:

- `bench_model_id`: A unique identifier for the benchmark model.
- `bench_model_dir`: The directory path to the benchmark model files.
- `bench_model_class`: The class name of the benchmark model in the code.
- `bench_dataset_id`: A unique identifier for the benchmark dataset.
- `bench_minibatch_count`: The number of minibatches used for benchmarking the device's training performance.
- `bench_batch_size`: The batch size used during the benchmarking process.
- `learning_rate`: The learning rate used for benchmark training.
- `bench_timeout_duration_s`: The timeout duration in seconds for the benchmark training process.

### `train_config`:

- `model_id`: A unique identifier for the training model.
- `model_dir`: The directory path to store the model files.
- `model_class`: The class name of the model in the code.
- `dataset_id`: A unique identifier for the training dataset.
- `epochs`: The number of epochs to train the model on each client during a federated learning round.
- `batch_size`: The batch size used during federated learning training.
- `learning_rate`: The learning rate used during federated learning training.
- `train_timeout_duration_s`: The timeout duration in seconds for each federated learning training round.
- `loss_function`: The loss function used for model optimization during training.
- `optimizer`: The optimization algorithm used for model training.
- `validation_data_path`: The directory path to fetch the validation data.
- `validation_batch_size`: The batch size of the data used for evaluating the global model. The evaluation is done for 1 minibatch.

## 2. [server_config.yaml](server_config.yaml)

This file contains the communication configuration settings for the server.

### `comm_config`:

- `discovery_type`: Selects the discovery/heartbeat mechanism. Use `grpc` or `mqtt`.
- `common`: Shared heartbeat settings used by the selected discovery backend:
  - `heartbeat_interval_s`: Interval for server-side heartbeat checks and advertised heartbeat timing.
  - `num_heartbeats_timestamp_cached`: Number of recent heartbeat timestamps to keep per client.
  - `max_heartbeat_miss_threshold`: Number of missed heartbeat intervals before a client is marked inactive.
- `mqtt_discovery`: Configuration for MQTT (Message Queuing Telemetry Transport) discovery:
  - `type`: The type of MQTT configuration. Set to `server` to specify server-related MQTT settings.
  - `broker_host`: The IP address or hostname of the MQTT broker.
  - `broker_port`: The port number for the MQTT broker.
  - `mqtt_sub_timeout_s`: The timeout duration in seconds for MQTT subscriptions.
  - `mqtt_server_topic`: The topic name used by the server to publish messages.
  - `mqtt_client_topic`: The topic name used by clients to publish messages.
- `grpc_discovery`: Configuration for gRPC client discovery:
  - `port`: The port where the gRPC discovery service listens.
- `grpc`: Configuration for gRPC (Google Remote Procedure Call) communication protocol:
  - `max_message_length`: Maximum gRPC message size in bytes.
  - `chunk_size_bytes`: The chunk size in bytes used for data transmission.
  - `timeout_s`: The timeout duration in seconds for gRPC communication.
- `restful`: Configuration for the server REST API:
  - `host`: Host/IP where the REST API listens.
  - `port`: Port where the REST API listens.

### `temp_dir_path`:

The directory path where temporary files are stored on the server.

## 3. [client_config.yaml](client_config.yaml)

This file contains the communication configuration settings for the client.

### `comm_config`:

- `mqtt_discovery`: Configuration for MQTT discovery protocol:
  - `type`: The type of MQTT configuration. Set to `client` to specify client-related MQTT settings.
  - `broker_host`: The IP address or hostname of the MQTT broker.
  - `broker_port`: The port number for the MQTT broker.
  - `mqtt_sub_timeout_s`: The timeout duration in seconds for MQTT subscriptions.
  - `heartbeat_timeout_s`: The timeout duration in seconds for client heartbeat messages.

- `grpc_runtime`: Configuration for gRPC runtime communication protocol:
  - `workers`: The number of worker threads to handle gRPC communication.
  - `sync_port`: The port number for synchronous gRPC communication.
  - `async_port`: The port number for asynchronous gRPC communication.

- `grpc_discovery`: Configuration for gRPC discovery:
  - `host`: The server host/IP that runs discovery.
  - `port`: The server discovery port.

### `dataset_config`:

- `datasets_dir_path`: The directory path where datasets are stored on the client.

### `general_config`:

- `temp_dir_path`: The directory path where temporary files are stored on the client.
- `cleanup_model_cache_on_exit`: Set to `true` to remove `<temp_dir_path>/model_cache` when the client exits.
- `cleanup_temp_on_exit`: Set to `true` to remove the entire `temp_dir_path` when the client exits. This takes precedence over model cache cleanup.
- `use_gpu`: Set to `true` to use GPU for training if available; otherwise, set to `false`.

## 4. [logger.conf](logger.conf)

This file configures the loggers, handlers, and formatters for the project.

- `[loggers]`: Defines the names of different loggers used in the project.
- `[handlers]`: Specifies the names of the log handlers (`fileHandler` and `streamHandler`) used to handle log messages.
- `[formatters]`: Defines the names and formatting details of log formatters.
- `[logger_xxx]`: Each logger name (e.g., `logger_SERVER_MANAGER`) is associated with a specific log level (e.g., `DEBUG`) and a log handler (e.g., `fileHandler`).
- `[formatter_fileFormatter]`: Specifies the format and date format for log messages.
- `[handler_fileHandler]` and `[handler_streamHandler]`: Configures the log handlers, including their log levels, associated formatters, and any additional arguments.

Please note that these configurations are user specific and are used to set up various aspects of the training, server communication, and logging functionality. Make sure to adjust the values accordingly for your specific use case.

For more information on how to use and customize these configuration files, refer to the project documentation or relevant code comments.
