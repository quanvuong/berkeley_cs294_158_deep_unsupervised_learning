from utils.tensorboard import add_plot
from utils.np_utils import exp_normalize, one_hot
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from jax import device_put, grad, jit, random, vmap
import matplotlib.pyplot as plt
import jax.numpy as np
import gtimer as gt
import time
import os
import os.path as osp
import numpy as onp
import scipy.stats as stats
from jax.experimental import optimizers

onp.random.seed(0)

distribution = onp.load('./hw1/qns_2/distribution.npy',
                        allow_pickle=True)


"""
You are provided with a 2-dimensional
array of floating point numbers
representing the joint distribution of x:
element (i, j) of this array is the
joint probability pdata(x1 = i, x2 = j).

Sample a dataset of 100,000 points from this distribution.
Treat the first 80% as a training set
and the remaining 20% as a test set.

pdata(x1, x2) = pdata(x1) pdata(x2|x1)
"""

# Computing pdata(x1)
pdata_x1 = onp.sum(distribution, axis=1)

# Sample x1
# multinomial returns
# an array of size (100000, 200)
# where each row is a one hot vector
x1s = onp.random.multinomial(1, pvals=pdata_x1, size=(100000, ))

# Sample x2
samples = onp.empty((100000, 2, 200), dtype=onp.float32)

for idx, x1 in enumerate(x1s):

    pdata_x2gx1 = distribution[onp.where(x1 == 1)[0][0]]

    pdata_x2gx1 = pdata_x2gx1 / onp.sum(pdata_x2gx1)

    x2 = onp.random.multinomial(1, pvals=pdata_x2gx1)

    samples[idx][0] = x1.astype(onp.float32)
    samples[idx][1] = x2.astype(onp.float32)


train_set = samples[:60000]
val_set = samples[60000:80000]
test_set = samples[80000:]

"""
Train pθ(x) = pθ(x1)pθ(x2|x1)
"""

# train p(x1) first
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


num_class = 200
lr = 0.0003

num_update = 100000
log_interval = 1000
batch_size = 32
tensorboard_logdir = osp.join(
    './tensorboard/qns_2_part_1_train_p(x1)', 'adam', time.strftime("%Y-%m-%d %H:%M:%S"))

# Debug value
# num_update = 10
# log_interval = 1

train_data = train_set[:, 0, :]
val_data = val_set[:, 0, :]
test_data = test_set[:, 0, :]

writer = SummaryWriter(tensorboard_logdir)
theta_x1 = np.zeros((num_class), dtype=np.float32)

# Use optimizers to set optimizer initialization and update functions
opt_init, opt_update, get_params = optimizers.adam(step_size=lr)

opt_state = opt_init(theta_x1)


@jit
def update(i, opt_state, x):

    theta = get_params(opt_state)

    grads = grad(loss, argnums=1)(x, theta)

    return opt_update(i, grads, opt_state)


batch_get_logprob2 = jit(vmap(get_logprob2, in_axes=(0, None)))
batch_update = jit(vmap(update, in_axes=(0, None)))

# Plot the empirical data distribution

# data has shape (num_samples, num_class)
# where each row is an one-hot vector
data_dist = onp.sum(train_data, axis=0)

data_dist = data_dist / data_dist.sum()

add_plot([onp.arange(len(data_dist))], [data_dist], ['Empirical Distribution'],
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

    opt_state = update(update_idx, opt_state, x)

    gt.stamp('update', unique=False)

    if update_idx % log_interval == 0:

        theta_x1 = get_params(opt_state)

        sample_logprob = batch_get_logprob2(x, theta_x1).mean()
        writer.add_scalar('sample log prob', onp.asarray(
            sample_logprob), update_idx)

        gt.stamp('sample logprob', unique=False)

        train_data_logprob = batch_get_logprob2(train_data, theta_x1).mean()

        writer.add_scalar('train set log prob', onp.asarray(
            train_data_logprob), update_idx)

        gt.stamp('val data logprob', unique=False)

        val_data_logprob = batch_get_logprob2(val_data, theta_x1).mean()

        writer.add_scalar('val set log prob', onp.asarray(
            val_data_logprob), update_idx)

        gt.stamp('val data logprob', unique=False)

        # Plot the model probabilities and empirical distribution
        normalized_theta = exp_normalize(theta_x1)

        normalized_onp_theta = onp.asarray(normalized_theta)
        data_from_model = onp.random.multinomial(
            1000, pvals=normalized_onp_theta)
        data_from_model = data_from_model.astype(np.float32)
        data_from_model_dist = data_from_model / onp.sum(data_from_model)

        add_plot(x_vals=None, y_vals=[normalized_theta, data_from_model_dist],
                 labels=['Model Probabilities',
                         'Empirical Dist of Data drawn from Model'],
                 xlabel='Class', ylabel='Class Probability',
                 writer=writer, tag='Model Probability', global_step=update_idx)


# Report test set performance
theta_x1 = get_params(opt_state)

test_data_neglogprob = - batch_get_logprob2(test_data, theta_x1).mean()

writer.add_scalar('test set neg log prob', onp.asarray(
    test_data_neglogprob))

writer.close()

print(gt.report())

# Save trained theta
os.makedirs('./hw1/qns_2/data', exist_ok=True)
np.save('./hw1/qns_2/data/part1_p(x1)', theta_x1)
