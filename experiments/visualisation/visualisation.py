import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def take_sorted(to_sort, sort_by):
    """Sort tensor according to sorted indices of the other tensor."""

    return np.take_along_axis(
        to_sort.numpy(), 
        np.flip(np.argsort(sort_by.numpy())), axis=-1)


def symmetry(symmetries, axs):
    epochs = [5*i for i in range(len(symmetries))]
    axs.plot(epochs, symmetries)
    axs.set(xlabel='Epoch', ylabel='Symmetry')
    axs.set_title('Assymetric measure')


def eigenspace_allignment(allignments, eigenvalues, axs):
    num_points = len(allignments)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_points))
    labels = [0, 5, 10, 15, 20, 100]
    
    for i in range(num_points):
        # Sort by sorted eigenvalue index.
        data = -take_sorted(allignments[i], eigenvalues[0])
        if i * 5 in labels:
            axs.plot(data, color=colors[i], label=f'Epoch {i*5}')
        else:
            axs.plot(data, color=colors[i])

    axs.set(xlabel='Sorted eigenvalue index', ylabel=r'Normalized correlation')
    axs.set_title('Eigenspace allignment')
    axs.legend()


def eigenvalues_Wp(eigv_wp, axs):
    num_points = len(eigv_wp)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_points))
    labels = [0, 5, 10, 15, 20, 50]
    
    for i in range(num_points):
        data = tf.sort(eigv_wp[i], direction='DESCENDING')
        if i * 5 in labels:
            axs.plot(data, color=colors[i], label=f'Epoch {i*5}')
        else:
            axs.plot(data, color=colors[i])

    axs.set_ylim((-0.5, 1))
    axs.set(xlabel='Sorted eigenvalue index', ylabel=r'Eigenvalue of $W_p$')
    axs.set_title(r'Evolvement of eigenvalues $p_j$ of $W_p$')
    axs.legend()


def eigenvalues_F(eigv_F, axs):

    num_points = len(eigv_F)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_points))
    labels = [0, 5, 10, 15, 20, 50]
    
    for i in range(num_points):
        data = tf.sort(tf.math.log(eigv_F[i]), direction='DESCENDING')
        if i * 5 in labels:
            axs.plot(data, color=colors[i], label=f'Epoch {i*5}')
        else:
            axs.plot(data, color=colors[i])
    axs.set(xlabel='Sorted eigenvalue index', ylabel=r'$\log$ eigenvalue of $F$')
    axs.set_title(r'Evolvement of eigenvalues $s_j$ of $F$')
    axs.legend()


