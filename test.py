import os
import tensorflow as tf
import argparse
import experiments.experiment_utils as eu
import experiments.config as configs
import experiments.evaluation as eval 
from data_processing.cifar10 import get_cifar10
import pickle
import matplotlib.pyplot as plt
import experiments.visualisation.visualisation as vis


def evaluate(ds, model):
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    for img, lbl, in ds:
        results = model.predict(img)
        acc_metric.update_state(lbl, results)
    
    acc = acc_metric.result().numpy()
    return acc


def plot_eigenspace_results(path):
    with open(os.path.join(path, 'F_eigenval.pkl'), 'rb') as f:
        F_eigenval = pickle.load(f)
        with open(os.path.join(path, 'wp_eigenval.pkl'), 'rb') as f:
            wp_eigenval = pickle.load(f)
            with open(os.path.join(path, 'allignment.pkl'), 'rb') as f:
                allignment = pickle.load(f)
                with open(os.path.join(path, 'symmetry.pkl'), 'rb') as f:
                    symmetry = pickle.load(f)

                    fig, axs = plt.subplots(1, 4)
                    fig.set_size_inches(25, 5)
                    vis.eigenvalues_F(F_eigenval, axs[0])
                    vis.eigenvalues_Wp(wp_eigenval, axs[1])
                    vis.eigenspace_allignment(allignment, F_eigenval, axs[2])
                    vis.symmetry(symmetry, axs[3])

                    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name', 
        type=str, required=True,
        help='Specifies folder name of finetuned classifier.')
    parser.add_argument(
        '--only_vis', action='store_true',
        help='Do not run classifier on test set')
    args = parser.parse_args()
    results_path = os.path.join('results', args.name)
    classifier_path = os.path.join(results_path, 'classifier')

    if not args.only_vis:
        # Config is only needed for batch size (the same for all models).
        config = configs.get_byol()
        ds_test, _ = get_cifar10(
            batch_size=config.batch_size, split='test', include_labels=True)
        model = tf.keras.models.load_model(classifier_path, compile=False)

        print(f'Accuracy: {(evaluate(ds_test, model)*100):.2f}%')

    eigenspace_results_path = os.path.join(results_path, 'eigenspace_results')
    if os.path.exists(eigenspace_results_path):
        plot_eigenspace_results(eigenspace_results_path)


if __name__ == '__main__':
    main()
