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
    # ds, _ = get_cifar10(batch_size=128, split='train')
    config = conf.get_simsiam_symmetric_predictor_decay()
    # experiment = eu.Experiment(config=config)
    # experiment.train(ds, 'my_test.h5', epochs=101)
    # print('finished pre-training ...')
    
    print('start finetuning ...')
    ev = eval.Evaluation('100my_test.h5', config)
    ds, num_examples = get_cifar10(batch_size=config.batch_size, split='train', include_labels=True)
    ds_test, _ = get_cifar10(batch_size=config.batch_size, split='test', include_labels=True)
    ev.train(ds, ds_test, num_examples, batch_size=config.batch_size, epochs=50)

if __name__=='__main__':
    main()