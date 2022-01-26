import os
import tensorflow as tf
import argparse
import experiments.experiment_utils as eu
import experiments.config as configs
import experiments.evaluation as eval 
from data_processing.cifar10 import get_cifar10

MODELS = ['byol', 'simsiam', 'directpred', 'directcopy']

def get_config(args):
    if args.model == 'byol':
        if args.eigenspace:
            if args.symmetry:
                return configs.get_eigenspace_experiment_with_symmetry()
            return configs.get_eigenspace_experiment()
        if args.one_layer_predictor:
            return configs.get_byol_baseline()
        return configs.get_byol()
    elif args.model == 'simsiam':
        if args.eigenspace:
            if args.symmetry:
                return configs.get_simsiam_symmetric()
            return configs.get_simsiam()
        if args.one_layer_predictor:
            return configs.get_simsiam_baseline()
        return configs.get_simsiam()
    elif args.model == 'directpred':
        return configs.get_direct_pred()
    elif args.model == 'directcopy':
        return configs.get_direct_copy()

def evaluate(ds, model):
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    for img, lbl, in ds:
        results = model.predict(img)
        acc_metric.update_state(lbl, results)
    
    acc = acc_metric.result().numpy()
    return acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--name', 
        type=str, required=True,
        help='Specifies folder name of finetuned classifier.')
    args = parser.parse_args()
    classifier_path = os.path.join('results', args.name, 'classifier')

    # Config is only needed for batch size (the same for all models).
    config = configs.get_byol()
    ds_test, _ = get_cifar10(
        batch_size=config.batch_size, split='test', include_labels=True)
    model = tf.keras.models.load_model(classifier_path, compile=False)

    print(f'Accuracy: {(evaluate(ds_test, model)*100):.2f}%')


if __name__ == '__main__':
    main()
