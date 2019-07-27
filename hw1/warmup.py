import numpy as np

def sample_data():
    count = 10000
    rand = np.random.RandomState(0)

    # rand.randn: normal dist
    a = 0.3 + 0.1 * rand.randn(count)
    b = 0.8 + 0.05 * rand.randn(count)

    # rand.rand: uniform dist over [0, 1]
    mask = rand.rand(count) < 0.5

    samples = np.clip(a * mask + b * (1 - mask), 0.0, 1.0)

    # linspace: return evenly spaced num over interval
    # digitize: return the idx of the bin to which
    # each value in the input belongs

    return np.digitize(samples, np.linspace(0.0, 1.0, 100))

print(sample_data())