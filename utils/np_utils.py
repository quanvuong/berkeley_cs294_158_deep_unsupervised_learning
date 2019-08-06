import jax.numpy as np
import numpy as onp


def one_hot(x, k, dtype=np.float32):
    # Create one hot encoding of x of size k

    return np.array(x[:, None] == np.arange(k), dtype=dtype)


def exp_normalize(v):

    exp_v = np.exp(v)

    norm = exp_v.sum()

    return exp_v / norm


def test_exp_normalize():

    a = np.zeros((100, ), dtype=np.float32)

    normalized_a = exp_normalize(a)

    print(normalized_a)


if __name__ == "__main__":
    test_exp_normalize()
