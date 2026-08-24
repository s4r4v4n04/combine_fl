# <span style="color: #94af08ff;">Integration Manual</span>

A guide for integrating custom components and interacting programmatically with the Flotilla framework. This document provides an API reference for the Flotilla session CLI and server endpoints. Use `flo_session.py` to submit federated learning jobs, and the REST API to interact with the server.

## <span style="color: #2AA5B8;">Overview</span>

Flotilla exposes its functionality through a CLI tool (`flo_session.py`) and a RESTful HTTP API served by the Flotilla server (`flo_server.py`). The CLI sends a POST request to the server's `/execute_command` endpoint with the session configuration parsed from a YAML file.

## <span style="color: #2AA5B8;">CLI Reference</span>

Submit a federated learning job to a running Flotilla server.

```shell
python flo_session.py <config_path> [options]
```

### Basic Usage

```shell
# Start a new session
python flo_session.py ../config/flotilla_config.yaml \
    --federated_server_endpoint localhost:12345

# Restore a previous session
python flo_session.py ../config/flotilla_config.yaml \
    --restore --session_id <uuid> \
    --federated_server_endpoint localhost:12345

# Revive a session with new clients
python flo_session.py ../config/flotilla_config.yaml \
    --revive --session_id <uuid> \
    --federated_server_endpoint localhost:12345

# Restore from checkpoint file
python flo_session.py ../config/flotilla_config.yaml \
    --file --session_id <uuid> \
    --federated_server_endpoint localhost:12345
```

## <span style="color: #2AA5B8;">config_path</span>

**Required** · `str` · Positional argument

Path to the Federated Learning configuration YAML file. This file defines the model, dataset, aggregation strategy, training rounds, and all session parameters. The CLI reads and parses this file, then sends it to the server as part of the request payload.

```python
parser.add_argument(
    "config_path", type=str,
    help="Path to the Federated Learning configuration file"
)

# How it's used: the config is loaded and sent to the server
with open(args.config_path) as file:
    federated_learning_config["federated_learning_config"] = yaml.safe_load(file)
```

```shell
python flo_session.py ../config/flotilla_config.yaml
```

## <span style="color: #2AA5B8;">--federated_server_endpoint</span>

**Optional** · `str` · Default: `10.24.24.31:12345`

The network address (`host:port`) of the running Flotilla server. The CLI constructs the REST API URL from this value and sends the session request to `http://{endpoint}/execute_command`.

```python
parser.add_argument(
    "--federated_server_endpoint",
    type=str,
    default="10.24.24.31:12345",
    help="Address of the Federated Learning Server",
)

# Constructs the API URL from this endpoint
api_url = f"http://{args.federated_server_endpoint}/execute_command"
```

```shell
python flo_session.py config.yaml --federated_server_endpoint localhost:12345
```

## <span style="color: #2AA5B8;">--restore</span>

**Optional** · `flag` · Default: `False`

Restore a previously interrupted session. Requires `--session_id`. The server will attempt to resume training with **all the original clients** from the previous session. If the original clients are not available, the request will fail. If both `--restore` and `--revive` are set, restore is attempted first.

```python
parser.add_argument(
    "--restore",
    action="store_true",
    default=False,
    help="Restore session by providing the session id.",
)

# When restore is set, the previous session_id is reused
if args.restore:
    federated_learning_config["session_id"] = args.session_id
    federated_learning_config["restore"] = True
```

```shell
python flo_session.py config.yaml --restore --session_id a1b2c3d4-...
```

## <span style="color: #2AA5B8;">--revive</span>

**Optional** · `flag` · Default: `False`

Revive a previously interrupted session with **whichever clients are currently available**. Unlike `--restore`, this does not require the original client set — the server will use whatever clients are connected. Requires `--session_id`.

```python
parser.add_argument(
    "--revive",
    action="store_true",
    default=False,
    help="Revive session by providing the session id.",
)

# When revive is set, the previous session_id is reused
if args.revive:
    federated_learning_config["session_id"] = args.session_id
    federated_learning_config["revive"] = True
```

```shell
python flo_session.py config.yaml --revive --session_id a1b2c3d4-...
```

## <span style="color: #2AA5B8;">--file</span>

**Optional** · `flag` · Default: `False`

Restore a session from a **checkpoint file** saved on the server. The server will attempt to load the model state from a previously saved checkpoint and resume training from that point. Requires `--session_id`.

```python
parser.add_argument(
    "--file",
    action="store_true",
    default=False,
    help="If file flag is provided server will try to restore from a checkpoint file.",
)

# When file is set, the checkpoint is loaded for the given session
if args.file:
    federated_learning_config["session_id"] = args.session_id
    federated_learning_config["file"] = True
```

```shell
python flo_session.py config.yaml --file --session_id a1b2c3d4-...
```

## <span style="color: #2AA5B8;">--session_id</span>

**Conditionally Required** · `str`

The UUID of the session to restore, revive, or load from file. This argument is **only required** when `--restore`, `--revive`, or `--file` is set. For new sessions, a UUID is automatically generated.

```python
# session_id is only added as a required arg when restore/revive/file is used
if "--restore" in sys.argv or "--revive" in sys.argv or "--file" in sys.argv:
    parser.add_argument(
        "--session_id",
        type=str,
        required=True,
        help="The id of the session to be restored/revived",
    )

# For new sessions, a UUID is auto-generated
federated_learning_config["session_id"] = str(uuid4())
```

## <span style="color: #E8712B;">POST /execute_command</span>

The server-side Flask endpoint that receives session requests from the CLI. It validates that no session is currently running and that at least one client is connected, then starts the federated training loop.

### Responses

| Status | Body | Condition |
|---|---|---|
| 200 | `{"message": "Session {id} finished"}` | Session completed successfully. |
| 400 | `{"message": "A session is already running"}` | Another session is in progress. |
| 400 | `{"message": "No active clients"}` | No clients are connected. |
| 400 | `{"message": "Invalid request"}` | Malformed or missing request body. |

```python
@app.route("/execute_command", methods=["POST"])
def execute_command():
    data = request.get_json()
    if session_running.is_set():
        return jsonify({"message": "A session is already running"}), 400
    elif len(flo_server.get_active_clients()) == 0:
        return jsonify({"message": "No active clients"}), 400
    elif data and "federated_learning_config" in data:
        session_config = data["federated_learning_config"]
        if (data["file"] or data["restore"] or data["revive"]) and data["session_id"]:
            session_id = handle_request(
                session_id=data["session_id"],
                session_config=session_config,
                restore=data["restore"],
                revive=data["revive"],
                file=data["file"],
            )
        elif data["session_id"]:
            session_id = data["session_id"]
            handle_request(session_id=session_id, session_config=session_config)
        return jsonify({"message": f"Session {session_id} finished"}), 200
    return jsonify({"message": "Invalid request"}), 400
```

> **Note:** The `aggregator` field in the training config should match the model's ML framework. For example, use `fedavg_torch` for PyTorch models or `fedavg_tensorflow` for TensorFlow models. See [Strategies](strategies.md) for a full list.

## <span style="color: #E8712B;">GET /get_status</span>

Checks the live status of a federated learning session.

### Parameters
- **`session_id`** (Query): The UUID of the session to check.

### Responses

| Status | Body | Condition |
|---|---|---|
| 200 | `{"status": "running", ...}` | The requested session is currently running. |
| 200 | `{"status": "not_found", ...}` | The session is not running (a different session is). |
| 200 | `{"status": "idle", ...}` | No session is running on the server. |
| 400 | `{"message": "Missing session_id"}` | The `session_id` parameter was not provided. |

```python
@app.route("/get_status", methods=["GET"])
def get_status():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"message": "Missing session_id"}), 400
    
    if session_running.is_set() and current_session_id == session_id:
        return jsonify({"status": "running"}), 200
    # ...
```

## <span style="color: #E8712B;">GET /client_status</span>

Returns information about all actively connected clients.

### Responses

| Status | Body | Condition |
|---|---|---|
| 200 | `{"clients": {"client_id": {...}}}` | Dictionary of active clients and their status. |

```python
@app.route("/client_status", methods=["GET"])
def client_status():
    try:
        active_clients = flo_server.get_active_clients()
        # ...
        return jsonify({"clients": clients}), 200
    except Exception as e:
        return jsonify({"clients": {}, "error": str(e)}), 200
```

## <span style="color: #E8712B;">GET /get_latest_gm</span>

Retrieves the latest serialized global model for the requested session. Returns a `.pt` file attachment.

### Parameters
- **`session_id`** (Query): The UUID of the session to fetch the model for.

### Responses

| Status | Body | Condition |
|---|---|---|
| 200 | `<Binary Data>` | Returns the `global_model_round_{num}.pt` file. |
| 200 | `{"status": "idle"}` | No session is running. |
| 200 | `{"status": "not_found"}` | The requested session is not currently running. |
| 400 | `{"message": "Missing session_id"}` | The `session_id` parameter was not provided. |
| 404 | `{"message": "Global model not available yet"}` | The session is running but hasn't completed round 1. |

```python
@app.route("/get_latest_gm", methods=["GET"])
def get_latest_gm():
    session_id = request.args.get("session_id")
    # ...
    # Get the actual global model weights from the running session
    session = flo_server.current_session
    global_model = session.training_session.get(f"{session_id}.global_model")
    
    buffer = io.BytesIO()
    torch.save(global_model, buffer)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/octet-stream",
        as_attachment=True,
        download_name=f"global_model_round_{current_round}.pt",
    )
```

> **Note:** For non-PyTorch models, the global model is serialized via `pickle` as a `WeightPayload` object rather than `torch.save()`. The response file attachment is still named `.pt` for compatibility, but the content format depends on the model's ML framework.

## <span style="color: #E8712B;">GET /query_past</span>

Retrieves the final validation metrics, start/end timestamps, and complete configuration of a completed session from the Redis/InMemory KV store.

### Parameters
- **`session_id`** (Query): The UUID of the past session.

### Responses

| Status | Body | Condition |
|---|---|---|
| 200 | Session metadata JSON | Returns the session details if found. |
| 400 | `{"message": "Missing session_id"}` | The `session_id` parameter was not provided. |
| 400 | `{"status": "running"}` | Reject if the session is currently running to enforce live checks. |
| 404 | `{"status": "not_found"}` | No state found in the KV store for the session. |

```python
@app.route("/query_past", methods=["GET"])
def query_past():
    session_id = request.args.get("session_id")
    
    # ...
    # Extract relevant metadata (exclude large objects like the model weights)
    last_round = training_session.get(f"{session_id}.last_round_number", "unknown")
    validation_metrics = training_session.get(f"{session_id}.global_validation_metrics", {})
    session_config = training_session.get(f"{session_id}.session_config", {})
    
    return jsonify({
        "session_id": session_id,
        "status": "completed",
        "last_checkpoint_round": last_round,
        "global_validation_metrics": validation_metrics,
        "session_config": session_config,
    }), 200
```
