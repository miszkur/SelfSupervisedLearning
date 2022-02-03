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
import pickle



def main():
    ds, num_examples = get_cifar10(
        batch_size=128, split='train', include_labels=True)
    ds_test, _ = get_cifar10(
        batch_size=128, split='test', include_labels=True)
    results = dict()

    # config = conf.get_byol_symmetric_predictor_decay_lr_adjusted()
    # name = config.name + ".h5"
    # ev = eval.Evaluation(name, config)
    # result = ev.train(
    #     ds, ds_test, num_examples, 
    #     batch_size=config.batch_size, 
    #     epochs=50)
    # results.update({name: result})
    # result_file = open("results.pkl", "wb")
    # pickle.dump(results, result_file)
    # result_file.close()
    # print(config.name + " is done ...")

    # config = conf.get_byol_symmetric_no_decay_lr_adjusted()
    # name = config.name + ".h5"
    # ev = eval.Evaluation(name, config)
    # result = ev.train(
    #     ds, ds_test, num_examples, 
    #     batch_size=config.batch_size, 
    #     epochs=50)
    # results.update({name: result})
    # result_file = open("results.pkl", "wb")
    # pickle.dump(results, result_file)
    # result_file.close()
    # print(config.name + " is done ...")

    # config = conf.get_simsiam_symmetric_no_decay_lr_adjusted()
    # name = config.name + ".h5"
    # ev = eval.Evaluation(name, config)
    # result = ev.train(
    #     ds, ds_test, num_examples, 
    #     batch_size=config.batch_size, 
    #     epochs=50)
    # results.update({name: result})
    # result_file = open("results.pkl", "wb")
    # pickle.dump(results, result_file)
    # result_file.close()
    # print(config.name + " is done ...")

    # config = conf.get_simsiam_predictor_decay_lr_adjusted()
    # name = config.name + ".h5"
    # ev = eval.Evaluation(name, config)
    # result = ev.train(
    #     ds, ds_test, num_examples, 
    #     batch_size=config.batch_size, 
    #     epochs=50)
    # results.update({name: result})
    # result_file = open("results.pkl", "wb")
    # pickle.dump(results, result_file)
    # result_file.close()
    # print(config.name + " is done ...")

    # config = conf.get_simsiam_no_decay_lr_adjusted()
    # name = config.name + ".h5"
    # ev = eval.Evaluation(name, config)
    # result = ev.train(
    #     ds, ds_test, num_examples, 
    #     batch_size=config.batch_size, 
    #     epochs=50)
    # results.update({name: result})
    # result_file = open("results.pkl", "wb")
    # pickle.dump(results, result_file)
    # result_file.close()
    # print(config.name + " is done ...")

    # config = conf.get_byol_predictor_decay_lr_adjusted()
    # name = config.name + ".h5"
    # ev = eval.Evaluation(name, config)
    # result = ev.train(
    #     ds, ds_test, num_examples, 
    #     batch_size=config.batch_size, 
    #     epochs=50)
    # results.update({name: result})
    # result_file = open("results2.pkl", "wb")
    # pickle.dump(results, result_file)
    # result_file.close()
    # print(config.name + " is done ...")

    config = conf.get_byol_no_decay_lr_adjusted()
    name = config.name + ".h5"
    ev = eval.Evaluation(name, config)
    result = ev.train(
        ds, ds_test, num_examples, 
        batch_size=config.batch_size, 
        epochs=50)
    results.update({name: result})
    result_file = open("results3.pkl", "wb")
    pickle.dump(results, result_file)
    result_file.close()
    print(config.name + " is done ...")


if __name__=='__main__':
    main()