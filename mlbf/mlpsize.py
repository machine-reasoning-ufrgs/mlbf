import datetime
import math
import os
import fire

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

import dataset
import numpy as np
import pandas as pd


def mlpsize(*inputs, solver='unigen', output='out.csv', cvfolds=5,
            max_neurons=512, mlp_activation='relu', metric='accuracy', save_dataset=True):
    """
    Tests various hidden-layer sizes until it goes perfect on a given formula, or formulas.
    (much code's been shamelessly copied from main.py
    TODO: modularize the code)

    :param inputs: path to (multiple) dataset (.pkl.gz) or CNF (Dimacs) files (see https://people.sc.fsu.edu/~jburkardt/data/cnf/cnf.html)
    :param solver: name of the SAT solver to sample satisfying samples
    :param output: path to output file
    :param cvfolds: number of folds for cross-validation
    :param max_neurons: max # neurons
    :param mlp_activation: MLP's activation function
    :param save_dataset: if True, saves the dataset as 'input.cnf.pkl.gz'
    :param metric: score metric
    :return:
    """

    write_header(output)

    # useful if output already exists for avoiding repeating the same formula
    df = pd.read_csv(output)

    for formula in inputs:
        print(f'Running for {formula}')
        # checks whether this formula has already been tested
        if df['formula'].str.contains(os.path.basename(formula)).any():
            print(f'Formula {formula} already executed, skipping...')
            continue

        start = datetime.datetime.now()

        data_x, data_y = dataset.get_dataset(formula, solver)

        if len(data_x) < 100:
            print(f'{inputs} has {len(data_x)} instances, which is less than 100 (too few to learn). Aborting.')
            continue  # goes to the next input

        # goes from 1, 2, 4, ..., max_neurons until it finds the optimal accuracy
        for i in range(int(math.log2(max_neurons)) + 1):
            num_neurons = 2 ** i  # the next power of 2
            learner = MLPClassifier(hidden_layer_sizes=(num_neurons,), activation=mlp_activation)
            scores = cross_validate(learner, data_x, data_y, cv=cvfolds, scoring=metric, n_jobs=1)

            with open(output, 'a') as outstream:
                # gathers accuracy and precision by running the model and writes it to output
                print(num_neurons, scores)
                score, std = np.mean(scores['test_score']), np.std(scores['test_score'])
                finish = datetime.datetime.now()
                outstream.write(f'{os.path.basename(formula)},{solver},{mlp_activation},{num_neurons},{cvfolds},{metric},{score},{std},{start},{finish}\n')
                if score == 1 and std == 0:
                    print(f'Perfect {metric} for {formula} with {num_neurons} neurons in 1 hidden layer')
                    break


def mlpsize_list(inputs, solver, output, cvfolds,
            max_neurons, mlp_activation, metric, save_dataset):
    """
    Just a Pool.starmap-friendly proxy for mlpsize which receives inputs as a list
    and 'explodes' it
    """
    if len(inputs) == 0:
        print(f'No inputs! Skipping {output}...')
        return

    return mlpsize(*inputs, solver=solver, output=output, cvfolds=cvfolds,
            max_neurons=max_neurons, mlp_activation=mlp_activation, metric=metric,
            save_dataset=save_dataset)


def write_header(output):
    """
    Creates the header in the output file if it does not exist
    :param output: path to the output file
    :return:
    """
    if output is not None and not os.path.exists(output):
        with open(output, 'w') as out:
            out.write('formula,sampler,activation,#neurons,cvfolds,metric,mean,std,start,finish\n')


def vsphase_parallel(basedir, simultaneous, activation, var_sizes=range(10, 101, 10)):
    """
    Attempts to run multiple mlpsize experiments (on the phase transition instances).
    Some weird error arised on reading a dataset
    :param basedir: where the phase instances are located
    :param simultaneous: how many tasks to run in parallel
    :param activation: activation function to test (relu, tanh or logistic)
    :param var_sizes: which variable sizes to test
    """
    from multiprocessing import Pool
    import pathlib
    import glob

    #activations = ['relu', 'sigmoid']
    #var_sizes = range(10, 101, 10)  # [10,20,...,100]
    directories, outfiles = [], []

    basedir = basedir.rstrip('/')  # removes trailing '/' if there is one
    for v in var_sizes:
        dirs_on_v = list(glob.glob(f'{basedir}/v{v}/*/'))
        directories += [os.path.normpath(d) for d in dirs_on_v]
        outfiles += [f'{basedir}/neurons_{activation}_v{v}_{pathlib.PurePath(d).name}.csv' for d in dirs_on_v]

    #print(outfiles, len(outfiles))

    #param_list = []
    #for a in activations:
    param_list = [(list(glob.glob(f'{d}/*.pkl.gz')), 'unigen', o, 5, 512, activation, 'accuracy', True) for d, o in zip(directories, outfiles)]

    #print(param_list, len(param_list))
    with Pool(int(simultaneous)) as p:
        p.starmap(mlpsize_list, param_list)


def vsphase(basedir, activation, var_sizes=range(10, 101, 10)):
    """
   Runs mlpsize experiments on the phase transition instances.
   :param basedir: where the phase instances are located
   :param activation: activation function to test (relu, tanh or logistic)
   :param var_sizes: which variable sizes to test
   """
    import pathlib
    import glob

    basedir = basedir.rstrip('/')  # removes trailing '/' if there is one
    for v in var_sizes:

        for instance_dir in glob.glob(f'{basedir}/v{v}/*/'):
            instance_dir = os.path.normpath(instance_dir)  # normalizes the path, e.g. removing redundant /
            outfile = f'{basedir}/neurons_{activation}_v{v}_{pathlib.PurePath(instance_dir).name}.csv'
            mlpsize_list(glob.glob(f'{instance_dir}/*.pkl.gz'), 'unigen', outfile, 5, 512, activation, 'accuracy', True)


if __name__ == '__main__':
    fire.Fire()
