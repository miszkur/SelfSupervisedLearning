import tensorflow as tf
import sys
sys.path.append('/experiments')
sys.path.append('/models')
import experiments.experiment_utils as eu
import data_processing.stl10 as dsp
import experiments.config as conf
import tensorflow_datasets as tfds
from data_processing.cifar10 import get_cifar10
import experiments.evaluation as eval


def main():
    ds, _ = get_cifar10(batch_size=128, split='train')

    config = conf.get_simsiam_symmetric_no_decay_lr_adjusted()
    experiment = eu.Experiment(config=config)
    name = config.name + ".h5"
    experiment.train(ds, name, epochs=2)
    print(config.name + " is done ...")

    config = conf.get_simsiam_predictor_decay_lr_adjusted()
    experiment = eu.Experiment(config=config)
    name = config.name + ".h5"
    experiment.train(ds, name, epochs=2)
    print(config.name + " is done ...")

    config = conf.get_simsiam_no_decay_lr_adjusted()
    experiment = eu.Experiment(config=config)
    name = config.name + ".h5"
    experiment.train(ds, name, epochs=2)
    print(config.name + " is done ...")

    config = conf.get_byol_symmetric_predictor_decay_lr_adjusted()
    experiment = eu.Experiment(config=config)
    name = config.name + ".h5"
    experiment.train(ds, name, epochs=2)
    print(config.name + " is done ...")

    config = conf.get_byol_symmetric_no_decay_lr_adjusted()
    experiment = eu.Experiment(config=config)
    name = config.name + ".h5"
    experiment.train(ds, name, epochs=2)
    print(config.name + " is done ...")

if __name__=='__main__':
    main()