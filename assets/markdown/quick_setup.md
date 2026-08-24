## Quick Setup
To test a sample Federated Learning (FL) workflow, follow these instructions for running the Flotilla server and one Flotilla client on the same machine to train the FACNN model on CIFAR10_IID. Flotilla supports multiple ML frameworks including PyTorch, TensorFlow, JAX, Scikit-Learn, and ONNX — see the [User Guide](user.md) for multi-framework setup instructions.

After cloning the GitHub repository, install the necessary requirements for both the server and the client. Optionally, regenerate the protocol buffers. Then, download and unzip the following file

```bash
wget "https://www.dropbox.com/scl/fi/020bgbj0esl4345jq5zqt/flotilla_quicksetup_data.zip?rlkey=css0ctcyanwcq19oj9ouf6lyh&st=f94j39vs&dl=1" -O flotilla_quicksetup_data.zip
unzip flotilla_quicksetup_data.zip
mv data src/
mv val_data src/
```

> **Note:** Before proceeding, ensure your `src/config/server_config.yaml` and `src/config/client_config.yaml` are correctly filled out, particularly the discovery mechanism (MQTT/gRPC) and ports.

To run the MQTT Broker, use the following command:
```bash
cd docker && docker-compose up -d
```

On two different terminals, run the following:
```bash
cd src && python flo_client.py
```
and
```bash
cd src && python flo_server.py
```

To start the training session, on a third terminal run:
```bash
cd src && python flo_session.py --config ../config/flotilla_quicksetup_config.yaml --server_endpoint localhost:12345
```