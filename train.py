import os
import argparse
import experiments.experiment_utils as eu
import experiments.config as configs
import experiments.evaluation as eval 
from data_processing.cifar10 import get_cifar10

MODELS = ['byol', 'simsiam', 'directpred', 'directcopy']
RESULTS_DIR = 'results'


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

def make_dir_if_needed(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, choices=MODELS, default='byol', 
        help=f'Neural network name. Can be one of: {MODELS}')
    parser.add_argument(
        '--symmetry', action='store_true',
        help='Impose symmetry regularisation on predictor (Wp)')
    parser.add_argument(
        '--eigenspace', action='store_true',
        help='Track eigenspace evolvement.')
    parser.add_argument(
        '--one_layer_predictor', action='store_true',
        help='Make predictor consist of only one layer (only applicable to BYOL and SimSiam)')
    parser.add_argument(
        '--epochs_pretraining', type=int, default=101,
        help='Number of epochs for self-supervised pretraining')
    parser.add_argument(
        '--epochs_finetuning', type=int, default=50,
        help='Number of epochs for supervised fine tuning')
    parser.add_argument(
        '--name', 
        type=str, default='encoder',
        help='Specifies filename of pretrained encoder and folder name for finetuned classifier.')
    args = parser.parse_args()

    config = get_config(args)
    ds, _ = get_cifar10(batch_size=config.batch_size, split='train')

    results_path = make_dir_if_needed(os.path.join(RESULTS_DIR, args.name))
    encoder_path = os.path.join(results_path, 'pre_trained_encoder.h5')
    eigenspace_results_path = make_dir_if_needed(os.path.join(results_path, 'eigenspace_results'))
    classifier_path = make_dir_if_needed(os.path.join(results_path, 'classifier'))

    print('=== Self-supervised pretraining ===')
    experiment = eu.Experiment(config=config)
    experiment.train(
        ds, 
        saved_encoder_path=encoder_path, 
        eigenspace_results_path=eigenspace_results_path,
        epochs=args.epochs_pretraining)

    print('\n\n=== Supervised fine-tuning ===')
    ev = eval.Evaluation(encoder_path, config)
    ds, num_examples = get_cifar10(
        batch_size=config.batch_size, split='train', include_labels=True)
    ds_test, _ = get_cifar10(
        batch_size=config.batch_size, split='test', include_labels=True)
    
    ev.train(
        ds, ds_test, num_examples, 
        batch_size=config.batch_size, 
        epochs=args.epochs_finetuning,
        saved_model_path=classifier_path)


if __name__ == '__main__':
    main()
