class KerasMLPMNIST:
    def __new__(cls, device="cpu", args: dict = None):
        import tensorflow as tf

        args = args or {}
        input_shape = tuple(args.get("input_shape", [28, 28, 1]))
        hidden_units = args.get("hidden_units", 128)
        num_classes = args.get("num_classes", 10)

        return tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Rescaling(1.0 / 255.0),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(hidden_units, activation="relu"),
                tf.keras.layers.Dense(num_classes),
            ],
            name="KerasMLPMNIST",
        )
