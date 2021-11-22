import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def take_sorted(to_sort, sort_by):
    """Sort tensor according to sorted indices of the other tensor."""

    return np.take_along_axis(
        to_sort.numpy(), 
        np.flip(np.argsort(sort_by.numpy())), axis=-1)


def symmetry(symmetries):
    epochs = [5*i for i in range(len(symmetries))]
    plt.plot(epochs, symmetries)
    plt.xlabel('Epoch')
    plt.ylabel('Symmetry')
    plt.show()


def eigenspace_allignment(allignments, eigenvalues):
    num_points = len(allignments)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_points))
    labels = [0, 5, 10, 15, 20, 50]
    
    for i in range(num_points):
        # Sort by sorted eigenvalue index.
        data = -take_sorted(allignments[i], eigenvalues[0])
        if i * 5 in labels:
            plt.plot(data, color=colors[i], label=f'Epoch {i*5}')
        else:
            plt.plot(data, color=colors[i])

    plt.xlabel('Sorted eigenvalue index')
    plt.ylabel('Normalized correlation')
    plt.title('Eigenspace allignment')
    plt.legend()
    plt.show()


def eigenvalues_Wp(eigv_wp):
    num_points = len(eigv_wp)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_points))
    labels = [0, 5, 10, 15, 20, 50]
    
    for i in range(num_points):
        data = tf.sort(eigv_wp[i], direction='DESCENDING')
        if i * 5 in labels:
            plt.plot(data, color=colors[i], label=f'Epoch {i*5}')
        else:
            plt.plot(data, color=colors[i])

    plt.ylim((-0.5, 1))
    plt.xlabel('Sorted eigenvalue index')
    plt.ylabel(r'Eigenvalue of $W_p$')
    plt.title(r'Evolvement of eigenvalues $p_j$ of $W_p$')
    plt.legend()
    plt.show()


def eigenvalues_F(eigv_F):

    num_points = len(eigv_F)
    colors = plt.cm.rainbow(np.linspace(0, 1, num_points))
    labels = [0, 5, 10, 15, 20, 50]
    
    for i in range(num_points):
        data = tf.sort(tf.math.log(eigv_F[i]), direction='DESCENDING')
        if i * 5 in labels:
            plt.plot(data, color=colors[i], label=f'Epoch {i*5}')
        else:
            plt.plot(data, color=colors[i])

    plt.ylim((-0.5, 1))
    plt.xlabel('Sorted eigenvalue index')
    plt.ylabel(r'$\log$ eigenvalue of $F$')
    plt.title(r'Evolvement of eigenvalues $s_j$ of $F$')
    plt.legend()
    plt.show()