import numpy as onp
import jax.numpy as np
from jax import jit, vmap, grad


from utils.np_utils import one_hot


def sample_data():
    count = 10000
    rand = onp.random.RandomState(0)

    # rand.randn: normal dist
    a = 0.3 + 0.1 * rand.randn(count)
    b = 0.8 + 0.05 * rand.randn(count)

    # rand.rand: uniform dist over [0, 1]
    mask = rand.rand(count) < 0.5

    samples = onp.clip(a * mask + b * (1 - mask), 0.0, 1.0)

    # linspace: return evenly spaced num over interval
    # digitize: return the idx of the bin to which
    # each value in the input belongs

    return onp.digitize(samples, onp.linspace(0.0, 1.0, 100))


def get_prob(x, theta):
    # x is the one-hot encoding of the index of the sample
    # theta is a 1-d array which parameterizes the model

    exp_theta = np.exp(theta)

    num = np.sum(x * exp_theta)
    denum = np.sum(exp_theta)

    return num / denum


def get_neglogprob(x, theta):
    p = get_prob(x, theta)
    return - np.log(p)


def update(x, theta):

    grads = grad(get_neglogprob, argnums=1)(x, theta)

    return theta - lr * grads


num_class = 100

data = sample_data()
data = one_hot(data, num_class)

theta = np.zeros((100), dtype=np.float32)
lr = 0.01

batched_get_prob = vmap(get_prob, in_axes=(0, None))

for idx, x in enumerate(data):
    theta = update(x, theta)

    data_prob = batched_get_prob(data, theta).sum()

    print(data_prob)

    if idx == 100:
        break
