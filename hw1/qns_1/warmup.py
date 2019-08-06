import os.path as osp
import time

import gtimer as gt
import jax.numpy as np
import matplotlib.pyplot as plt
import numpy as onp
from jax import device_put, grad, jit, random, vmap
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from utils.np_utils import exp_normalize, one_hot
from utils.tensorboard import add_plot


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


@jit
def get_prob(x, theta):
    # x is the one-hot encoding of the index of the sample
    # theta is a 1-d array which parameterizes the model

    exp_theta = np.exp(theta)

    num = np.sum(x * exp_theta)
    denum = np.sum(exp_theta)

    return num / denum


@jit
def get_neglogprob(x, theta):
    p = get_prob(x, theta)
    return - np.log(p)


@jit
def loss(x, theta):

    batch_neglogprob = vmap(get_neglogprob, in_axes=(0, None))

    neglogprob = batch_neglogprob(x, theta)

    return np.mean(neglogprob)


@jit
def get_logprob2(x, theta):
    p = get_prob(x, theta)
    return np.log2(p)


@jit
def update(x, theta):

    grads = grad(loss, argnums=1)(x, theta)

    return theta - lr * grads


num_class = 100
lr = 0.001
train_percent = 0.8
num_update = 200000
log_interval = 1000
batch_size = 32
tensorboard_logdir = osp.join(
    './tensorboard/warmup', time.strftime("%Y-%m-%d %H:%M:%S"))

# Debug value
# num_update = 10
# log_interval = 1

data = sample_data()
data = one_hot(data, num_class)
data = device_put(data)

num_train = int(train_percent * len(data))

train_data = data[:num_train]
val_data = data[num_train:]

writer = SummaryWriter(tensorboard_logdir)
theta = np.zeros((num_class), dtype=np.float32)

batch_get_logprob2 = jit(vmap(get_logprob2, in_axes=(0, None)))
batch_update = jit(vmap(update, in_axes=(0, None)))

# Plot the empirical data distribution

# data has shape (num_samples, num_class)
# where each row is an one-hot vector
data_dist = np.sum(data, axis=0)

data_dist = data_dist / data_dist.sum()

add_plot([np.arange(len(data_dist))], [data_dist], ['Empirical Distribution'],
         xlabel='Class', ylabel='Class Probability',
         writer=writer, tag='Empirical Distribution', global_step=0)

key = random.PRNGKey(0)

all_batch_idxes = random.randint(key, shape=(num_update,
                                             batch_size,),
                                 minval=0, maxval=len(train_data))

gt.start()

for update_idx in trange(num_update):

    batch_idxes = all_batch_idxes[update_idx]

    x = train_data[batch_idxes]

    gt.stamp('sample minibatch', unique=False)

    theta = update(x, theta)

    gt.stamp('update', unique=False)

    if update_idx % log_interval == 0:
        sample_logprob = batch_get_logprob2(x, theta).mean()
        writer.add_scalar('sample log prob', onp.asarray(
            sample_logprob), update_idx)

        gt.stamp('sample logprob', unique=False)

        val_data_logprob = batch_get_logprob2(val_data, theta).mean()

        writer.add_scalar('val set log prog', onp.asarray(
            val_data_logprob), update_idx)

        gt.stamp('val data logprob', unique=False)

        # Plot the model probabilities and empirical distribution
        normalized_theta = exp_normalize(theta)

        normalized_onp_theta = onp.asarray(normalized_theta)
        data_from_model = onp.random.multinomial(
            1000, pvals=normalized_onp_theta)
        data_from_model = data_from_model.astype(np.float32)
        data_from_model_dist = data_from_model / np.sum(data_from_model)

        add_plot(x_vals=None, y_vals=[normalized_theta, data_from_model_dist],
                 labels=['Model Probabilities',
                         'Empirical Dist of Data drawn from Model'],
                 xlabel='Class', ylabel='Class Probability',
                 writer=writer, tag='Model Probability', global_step=update_idx)

writer.close()

print(gt.report())
