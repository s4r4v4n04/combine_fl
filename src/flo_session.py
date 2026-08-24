"""
Authors: Prince Modi, Roopkatha Banerjee, Yogesh Simmhan
Emails: princemodi@iisc.ac.in, roopkathab@iisc.ac.in, simmhan@iisc.ac.in
Copyright 2023 Indian Institute of Science
Licensed under the Apache License, Version 2.0, http://www.apache.org/licenses/LICENSE-2.0
"""

import argparse
import pprint
import sys
from uuid import uuid4
import os
import requests
import yaml

from utils.cli_parser import get_args
args = get_args()


def _fmt(response):
    """Return the response body as parsed JSON when possible, else the raw text.
    Prevents a crash when the server replies with a non-JSON body (e.g. an empty
    500), so the actual error is printed instead of a JSONDecodeError."""
    try:
        return response.json()
    except ValueError:
        return response.text or "<empty response body>"


## -- API TO START A SESSION -- ##
if args.task == 'start_session':
    api_url = f"http://{args.server_endpoint}/execute_command"

    federated_learning_config = dict()
    federated_learning_config["session_id"] = str(uuid4())

    with open(args.config) as file:
        federated_learning_config["federated_learning_config"] = yaml.safe_load(file)
        pass

    if args.restore:
        federated_learning_config["session_id"] = args.session_id
        federated_learning_config["restore"] = True
    else:
        federated_learning_config["restore"] = False
    if args.revive:
        federated_learning_config["session_id"] = args.session_id
        federated_learning_config["revive"] = True
    else:
        federated_learning_config["revive"] = False
    if args.file:
        federated_learning_config["session_id"] = args.session_id
        federated_learning_config["file"] = True
    else:
        federated_learning_config["file"] = False

    # send a POST request with the dictionary as JSON data
    try:
        pprint.pprint(federated_learning_config)
        response = requests.post(api_url, json=federated_learning_config)
        if response.status_code == 200:
            print("Request was successful!")
            print("Response JSON:", _fmt(response))
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response JSON:", _fmt(response))
    except requests.exceptions.ConnectionError:
        print("Server closed connection without response")

elif args.task == 'get_status':
    api_url = f"http://{args.server_endpoint}/get_status"

    # build the request payload
    request_params = {"session_id": args.session_id}

    # send a GET request with session_id as query parameter
    try:
        pprint.pprint(request_params)
        response = requests.get(api_url, params=request_params)
        if response.status_code == 200:
            print("Request was successful!")
            print("Session Status:", _fmt(response))
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response JSON:", _fmt(response))
    except requests.exceptions.ConnectionError:
        print("Server closed connection without response")

elif args.task == 'query_past':
    api_url = f"http://{args.server_endpoint}/query_past"

    # build the request payload
    request_params = {"session_id": args.session_id}

    # send a GET request with session_id as query parameter
    try:
        pprint.pprint(request_params)
        response = requests.get(api_url, params=request_params)
        if response.status_code == 200:
            print("Request was successful!")
            print("Session Status:", _fmt(response))
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response JSON:", _fmt(response))
    except requests.exceptions.ConnectionError:
        print("Server closed connection without response")

elif args.task == 'get_latest_gm':
    api_url = f"http://{args.server_endpoint}/get_latest_gm"

    os.makedirs(args.save_path, exist_ok=True)

    # Send a GET request with session_id as query parameter
    try:
        print(f"Requesting global model for session: {args.session_id}")
        response = requests.get(api_url, params={"session_id": args.session_id})

        if response.status_code == 200:
            # Check if we got a file (binary) or JSON error
            content_type = response.headers.get("Content-Type", "")
            if "application/octet-stream" in content_type:
                # Extract filename from Content-Disposition header
                filename = "global_model.pt"
                content_disp = response.headers.get("Content-Disposition", "")
                if "filename=" in content_disp:
                    filename = content_disp.split("filename=")[-1].strip()

                save_file = os.path.join(args.save_path, filename)
                with open(save_file, "wb") as f:
                    f.write(response.content)
                print(f"Global model weights saved to: {save_file}")
            else:
                print("Response:", _fmt(response))
        else:
            print(f"Request failed with status code {response.status_code}")
            print("Response:", _fmt(response))
    except requests.exceptions.ConnectionError:
        print("Server closed connection without response")
