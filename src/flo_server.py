"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import asyncio
import io
import os
import pickle
import tarfile
from argparse import ArgumentParser
from os import getpid
from threading import Event
from uuid import uuid4

from flask import Flask, jsonify, request, send_file
from waitress import serve

from ml_backends.registry import get_backend_from_payload
from server.server_file_manager import OpenYaML
from server.server_manager import FlotillaServerManager, normalize_server_config
from server.server_state_manager import StateManager
from utils.monitor import Monitor

app = Flask("flo_server")


process_id: int = getpid()
session_running = Event()
current_session_id = None  # Track the currently running session ID

parser = ArgumentParser()

parser.add_argument(
    "--server_config",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "config", "server_config.yaml"),
    help="Path to server_config.yaml (default: src/config/server_config.yaml)",
)
parser.add_argument(
    "--monitor",
    action="store_true",
    default=False,
    help="Monitor CPU/RAM/Disk/Network IO",
)
args = parser.parse_args()

server_config = normalize_server_config(OpenYaML(args.server_config))
is_monitoring = args.monitor
if is_monitoring:
    monitor = Monitor("0", process_id)



def handle_request(
    session_id,
    session_config,
    restore=False,
    revive=False,
    file=False,
):
    global current_session_id
    current_session_id = session_id
    session_running.set()
    if is_monitoring:
        monitor.set_session(session_id)
    print("Starting Session:", session_id)
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            asyncio.gather(
                flo_server.run(session_id, session_config, restore, revive, file)
            )
        )
    except asyncio.CancelledError:
        print("flo_server.handle_request.exception:: Session Cancelled")
    except KeyboardInterrupt:
        print("Received KeyboardInterrupt")
    except RuntimeError as e:
        print(f"flo_server.handle_request.exception:: RuntimeError in Gather loop- {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"flo_server.handle_request.unknown_exception:: Exception in Gather loop- {e}")
    finally:
        session_running.clear()
        current_session_id = None
        if is_monitoring:
            monitor.reset_session()
        return session_id


@app.route("/execute_command", methods=["POST"])
def execute_command():
    try:
        data = request.get_json()
        if session_running.is_set():
            print("Session Already Running")
            return jsonify({"message": "A session is already running"}), 400
        elif len(flo_server.get_active_clients()) == 0:
            print("No active clients")
            return jsonify({"message": "No active clients"}), 400
        elif data and "federated_learning_config" in data:
            session_config = data["federated_learning_config"]

            if (data["file"] or data["restore"] or data["revive"]) and data["session_id"]:
                restore_session_id = data["session_id"]
                session_id = handle_request(
                    session_id=restore_session_id,
                    session_config=session_config,
                    restore=data["restore"],
                    revive=data["revive"],
                    file=data["file"],
                )
            elif data["session_id"]:
                session_id = data["session_id"]
                handle_request(session_id=session_id,session_config=session_config)

            return jsonify({"message": f"Session {session_id} finished"}), 200
        else:
            print("Received Invalid Request")
            return jsonify({"message": "Invalid request"}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"message": str(e)}), 500


@app.route("/get_status", methods=["GET"])
def get_status():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"message": "Missing session_id"}), 400
    print(f"Status request for session: {session_id}")

    if session_running.is_set() and current_session_id == session_id:
        return jsonify({
            "session_id": session_id,
            "status": "running",
            "message": f"Session {session_id} is currently running"
        }), 200
    elif session_running.is_set() and current_session_id != session_id:
        return jsonify({
            "session_id": session_id,
            "status": "not_found",
            "message": f"Session {session_id} is not running. Current running session: {current_session_id}"
        }), 200
    else:
        return jsonify({
            "session_id": session_id,
            "status": "idle",
            "message": f"No session is currently running"
        }), 200


@app.route("/client_status", methods=["GET"])
def client_status():
    """Return information about connected clients."""
    try:
        active_clients = flo_server.get_active_clients()
        clients = {}
        for client_id in active_clients:
            client_type = flo_server.client_info.get(f"{client_id}.role")
            client_status = flo_server.client_info.get(f"{client_id}.status") or "idle"
            clients[client_id] = {
                "client_id": client_id,
                "client_type": client_type if client_type else "unknown",
                "status": client_status,
                "is_active": True,
            }
        return jsonify({"clients": clients}), 200
    except Exception as e:
        return jsonify({"clients": {}, "error": str(e)}), 200


@app.route("/get_latest_gm", methods=["GET"])
def get_latest_gm():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"message": "Missing session_id"}), 400
    print(f"Global model request for session: {session_id}")

    if not session_running.is_set():
        return jsonify({
            "session_id": session_id,
            "status": "idle",
            "message": "No session is currently running"
        }), 200
    elif current_session_id != session_id:
        return jsonify({
            "session_id": session_id,
            "status": "not_found",
            "message": f"Session {session_id} is not running. Current running session: {current_session_id}"
        }), 200

    # Get the actual global model weights from the running session
    try:
        session = flo_server.current_session
        if session is None:
            return jsonify({"message": "Session object not available"}), 500

        # Get the global model weights and current round number
        global_model = session.training_session.get(f"{session_id}.global_model")
        current_round = session.training_session.get(f"{session_id}.last_round_number")

        if global_model is None:
            return jsonify({"message": "Global model not available yet"}), 404

        buffer = io.BytesIO()
        backend = get_backend_from_payload(global_model)
        backend.save_weights(global_model, buffer)
        buffer.seek(0)
        extension = "pt" if backend.name == "pytorch" else "weights"

        return send_file(
            buffer,
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name=f"global_model_round_{current_round}.{extension}",
        )
    except Exception as e:
        print(f"Error retrieving global model: {e}")
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route("/query_past", methods=["GET"])
def query_past():
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"message": "Missing session_id"}), 400
    print(f"Past session query for: {session_id}")

    # Reject if the session is currently running — use /get_status instead
    if session_running.is_set() and current_session_id == session_id:
        return jsonify({
            "session_id": session_id,
            "status": "running",
            "message": f"Session {session_id} is currently running. Use /get_status to check its live status."
        }), 400

    try:
        # Construct the StateManager to connect to Redis/InMemory
        state_location = server_config["state"].get("state_location", "inmemory")
        state_hostname = server_config["state"].get("state_hostname", None)
        state_port = server_config["state"].get("state_port", None)

        training_session_state = StateManager(
            loc=state_location,
            name="training_session",
            host=state_hostname,
            port=state_port,
            state_id=session_id,
        )

        raw_state = training_session_state.getall()

        if not raw_state:
            return jsonify({
                "session_id": session_id,
                "status": "not_found",
                "message": f"No state found in Redis for session {session_id}"
            }), 404

        # Redis hgetall returns dictionary with bytes keys and pickled bytes values
        # We deserialize it safely. If it's already a dict with strings (inmemory), it bypasses.
        training_session = {
            k.decode("utf-8") if isinstance(k, bytes) else k: pickle.loads(v) if isinstance(v, bytes) else v
            for k, v in raw_state.items()
        }

        # Extract relevant metadata (exclude large objects like the model weights)
        last_round = training_session.get(f"{session_id}.last_round_number", "unknown")
        validation_metrics = training_session.get(f"{session_id}.global_validation_metrics", {})
        session_config = training_session.get(f"{session_id}.session_config", {})
        start_timestamp = training_session.get(f"{session_id}.start_timestamp", "unknown")
        end_timestamp = training_session.get(f"{session_id}.end_timestamp", "unknown")

        return jsonify({
            "session_id": session_id,
            "status": "completed",
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "last_checkpoint_round": last_round,
            "global_validation_metrics": validation_metrics,
            "session_config": session_config,
        }), 200

    except Exception as e:
        print(f"Error reading state from KV store: {e}")
        return jsonify({"message": f"Error reading state: {str(e)}"}), 500


def main():
    print("Starting FLo_Server")
    global flo_server
    flo_server = FlotillaServerManager(server_config)
    serve(
        app,
        host=server_config["comm_config"]["restful"]["rest_hostname"],
        port=server_config["comm_config"]["restful"]["rest_port"],
    )


if __name__ == "__main__":
    main()
