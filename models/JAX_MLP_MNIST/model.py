class JaxMLPMNIST:
    def __init__(self, device="cpu", args: dict = None):
        import jax
        import jax.numpy as jnp

        args = args or {}
        input_dim = args.get("input_dim", 784)
        hidden_units = args.get("hidden_units", 128)
        num_classes = args.get("num_classes", 10)
        seed = args.get("seed", 0)

        key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
        self.params = {
            "dense1": {
                "w": jax.random.normal(key1, (input_dim, hidden_units)) * 0.01,
                "b": jnp.zeros((hidden_units,)),
            },
            "dense2": {
                "w": jax.random.normal(key2, (hidden_units, num_classes)) * 0.01,
                "b": jnp.zeros((num_classes,)),
            },
        }

    def apply(self, params, x):
        import jax.numpy as jnp

        x = jnp.asarray(x, dtype=jnp.float32) / 255.0
        x = x.reshape((x.shape[0], -1))
        hidden = jnp.maximum(x @ params["dense1"]["w"] + params["dense1"]["b"], 0)
        return hidden @ params["dense2"]["w"] + params["dense2"]["b"]

    def get_weights(self):
        return self.params

    def set_weights(self, weights):
        self.params = weights
class JaxMLPMNIST:
    def __init__(self, device="cpu", args: dict = None):
        import jax
        import jax.numpy as jnp

        args = args or {}
        input_dim = args.get("input_dim", 784)
        hidden_units = args.get("hidden_units", 128)
        num_classes = args.get("num_classes", 10)
        seed = args.get("seed", 0)

        key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
        self.params = {
            "dense1": {
                "w": jax.random.normal(key1, (input_dim, hidden_units)) * 0.01,
                "b": jnp.zeros((hidden_units,)),
            },
            "dense2": {
                "w": jax.random.normal(key2, (hidden_units, num_classes)) * 0.01,
                "b": jnp.zeros((num_classes,)),
            },
        }

    def apply(self, params, x):
        import jax.numpy as jnp

        x = jnp.asarray(x, dtype=jnp.float32) / 255.0
        x = x.reshape((x.shape[0], -1))
        hidden = jnp.maximum(x @ params["dense1"]["w"] + params["dense1"]["b"], 0)
        return hidden @ params["dense2"]["w"] + params["dense2"]["b"]

    def get_weights(self):
        return self.params

    def set_weights(self, weights):
        self.params = weights
