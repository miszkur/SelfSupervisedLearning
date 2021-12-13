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
    epochs = 101
    config = conf.get_simsiam_pred()
    ds, _ = get_cifar10(batch_size=config.batch_size, split='train')
    experiment = eu.Experiment(config)
    experiment.train(
        ds, 
        saved_encoder_path='saved_model/simsiam_pred.h5', 
        epochs=epochs)

    print('--- first experiment done ---')

    config = conf.get_simsiam_baseline()
    ds, _ = get_cifar10(batch_size=config.batch_size, split='train')
    experiment = eu.Experiment(config)
    experiment.train(
        ds, 
        saved_encoder_path='saved_model/simsiam_baseline.h5', 
        epochs=epochs)

    config = conf.get_deeper_projection()
    ds, _ = get_cifar10(batch_size=config.batch_size, split='train')
    experiment = eu.Experiment(config)
    experiment.train(
        ds, 
        saved_encoder_path='saved_model/deeper_projection_encoder.h5', 
        saved_projection_head_path='saved_model/deeper_projection_projector.h5',
        epochs=epochs)
    

    

if __name__ == '__main__':
    main()