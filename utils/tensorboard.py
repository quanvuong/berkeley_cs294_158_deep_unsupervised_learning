import matplotlib.pyplot as plt
import jax.numpy as np


def add_plot(x_vals, y_vals, labels, xlabel, ylabel,
             writer, tag, global_step):

    fig = plt.figure()

    ax = fig.add_subplot()

    if x_vals is None:
        tmp_xvals = []
        for y_val in y_vals:
            tmp_xvals.append(np.arange(len(y_val)))
        x_vals = tmp_xvals

    for x_val, y_val, label in zip(x_vals, y_vals, labels):

        ax.plot(x_val, y_val, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend()

    writer.add_figure(tag, fig, global_step)

    plt.close(fig)
