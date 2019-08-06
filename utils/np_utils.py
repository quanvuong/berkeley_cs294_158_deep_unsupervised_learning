import jax.numpy as np


def one_hot(x, k, dtype=np.float32):
    # Create one hot encoding of x of size k

    return np.array(x[:, None] == np.arange(k), dtype=dtype)
