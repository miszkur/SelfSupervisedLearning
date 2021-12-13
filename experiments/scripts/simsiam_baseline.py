import argparse
import tensorflow as tf
import sys
sys.path.append('../')
sys.path.append('../..')
import experiment_utils as eu
import config as conf
import importlib
from data_processing.cifar10 import get_cifar10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=101)
    parser.add_argument(
        '--path_to_save_encoder', type=str, default='saved_model/simsiam_baseline.h5')
    args = parser.parse_args()
    config = conf.get_simsiam_baseline()
    ds, _ = get_cifar10(batch_size=config.batch_size, split='train')
    experiment = eu.Experiment(config)
    experiment.train(
        ds, 
        saved_encoder_path=args.path_to_save_encoder, 
        epochs=args.epochs)

if __name__ == '__main__':
    main()