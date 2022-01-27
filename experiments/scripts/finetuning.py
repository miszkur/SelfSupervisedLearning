import argparse
import tensorflow as tf
import sys
sys.path.append('../')
sys.path.append('../..')
import experiment_utils as eu
import config as conf
import importlib
from data_processing.cifar10 import get_cifar10
import experiments.evaluation as eval 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument(
        '--path_to_save_encoder', type=str, default='saved_model/simsiam_no_decay_lr_adjusted.h5')
    args = parser.parse_args()
    config = conf.get_simsiam_symmetric_predictor_decay_lr_adjusted()
    ev = eval.Evaluation(args.path_to_save_encoder, config)
    ds, num_examples = get_cifar10(
        batch_size=config.batch_size, split='train', include_labels=True)
    ds_test, _ = get_cifar10(
        batch_size=config.batch_size, split='test', include_labels=True)

    ev.train(
        ds, ds_test, num_examples, 
        batch_size=config.batch_size, 
        epochs=args.epochs)

if __name__=='__main__':
    main()