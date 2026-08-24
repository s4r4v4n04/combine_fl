<div align="center">
  <img src="https://img.shields.io/badge/FLOTILLA-DASHBOARD-0077B6?style=for-the-badge" alt="Flotilla Dashboard">
  <h1>Interactive Dashboard Setup</h1>
  <p><i>Real-time metrics, telemetry, and architecture documentation.</i></p>
</div>

---

## Architecture Overview

We have structured the dashboard into a robust, scalable backend architecture for better separation of concerns:

* `client/`: Contains the frontend UI templates and static assets.
* `server/src/`: Contains the backend API.
  * `main.py`: The application entry point (registers Flask blueprints and starts services).
  * `database.py`: Placeholder configuration for potential future database connections.
  * `models.py`: Defines the shape of the global configuration and runtime state dicts.
  * `routes/`: All network endpoints (separated into `auth.py`, `fl_server.py`, `fl_session.py`, `metrics.py`, `files.py`, etc.).
  * `services/`: Any heavy lifting, like `process_service` for OS commands and `log_service` for background thread logic.

<br/>

## State-Driven Metrics System

The dashboard is designed to seamlessly pull its live metrics data from `fedml-ng`'s `StateManager` instead of relying entirely on parsed text logs. 

**Under the hood, the metrics endpoint:**
* Connects directly to the `StateManager` (`redis` or `inmemory`, configuring itself automatically based on your `server_config.yaml`).
* Fetches the exact list of `global_validation_metrics` and the `last_round_number`.
* Computes the explicit sequence of rounds using `validation_round_interval` stored in the `session_config`.
* Bypasses the need for log reading completely for plotting and dashboard state sync.

> [!NOTE]
> **Fallback Mechanism:** If the requested session's state no longer exists in `redis` (which could happen for very old sessions that were pruned from memory or restarted), the endpoint will flawlessly fall back to using the default log-based parser to ensure the dashboard continues functioning exactly as before.

> [!TIP]
> The dashboard displays training metrics from any ML framework. All built-in framework adapters return a standardised `{accuracy, loss}` metrics format, so the dashboard works identically regardless of whether your session uses PyTorch, TensorFlow, JAX, Scikit-Learn, or ONNX models.

<br/>

## How to Start the Dashboard

To launch the dashboard server, run the following commands in your terminal:

1. **Activate your environment:**
   ```bash
   conda activate <env_name>
   ```

2. **Navigate into the dashboard server directory:**
   ```bash
   cd ../fastapi_dash/server
   ```

3. **Install required dependencies:**
   ```bash
   pip install fastapi uvicorn python-multipart jinja2 itsdangerous starlette PyYAML python-dotenv SQLAlchemy psycopg2-binary
   ```

4. **Run the application as a module:**
   ```bash
   python -m src.main
   ```

> [!IMPORTANT]
> **Client Connectivity:** Once the dashboard is loaded and running, ensure that your client nodes are up and active so they can begin streaming real-time telemetry to the interface!

---

<div align="center">
  <b><a href="v1.0.md">&larr; Back to Release Notes</a></b>
</div>