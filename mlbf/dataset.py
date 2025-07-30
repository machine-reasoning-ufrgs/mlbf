import os
import re

import glob
from math import log, log10

import fire
import random
import numpy as np
import pandas as pd
from pysat.formula import CNF

from positives import PySATSampler, UnigenSampler, Unigen3Sampler
import negatives as negative_sampler


def prepare_dataset(positives, negatives):
    """
    Creates the dataset from the positive and negative samples.
    Adds the labels, concatenates, shuffles, creates a
    dataframe and separate into inputs and labels
    :param positives:list
    :param negatives:list
    :return: pandas.DataFrame object
    """
    print(f'Preparing dataset.')

    # appends the labels (1 to sat samples, 0 to unsat samples)
    for p in positives:
        p.append(1)
    for n in negatives:
        n.append(0)

    # concats the two lists and shuffles
    all_data = positives + negatives
    # random.seed(2) # uncomment to debug (otherwise each shuffle will give a different array)
    random.shuffle(all_data)

    # column names = [x1, x2, ..., xn, f] (each x_i is a variable and f is the label)
    input_names = [f'x{i}' for i in range(1, len(all_data[0]))]
    df = pd.DataFrame(all_data, columns=input_names + ['f'])
    if any(df.duplicated(input_names)):
        print('ERROR: there are duplicate inputs in the dataset. Returning empty.')
        return [], []

    # replaces negated by 0 and asserted by 1
    df.mask(df < 0, 0, inplace=True)
    df.mask(df > 0, 1, inplace=True)

    if df.isin([np.nan, np.inf, -np.inf]).values.any():
        print("ERROR: there are invalid samples in the dataset. Returning empty.")
        return pd.DataFrame()

    return df


def dataset_exists(cnf):
    """
    Tells whether a dataset for the given CNF file exists
    :param cnf:
    :return:
    """
    return len(glob.glob(f'{cnf}_*.pkl.gz')) > 0


def get_dataset(path, solver='unigen', num_positives=500, num_negatives=500, save_dataset=True, overwrite=False):
    """
    Returns an existing dataset for the given formula if one exists.
    Otherwise, generates a new dataset with the solver and specified parameters
    """
    if dataset_exists(path) or path.endswith('.pkl.gz'):
        return read_dataset(path)
    else:
        return generate_dataset(path, solver, num_positives, num_negatives, save_dataset, overwrite)


def read_dataset(path):
    """
    Attempts to retrieve a dataset from a given file.
    The file can be either a .pkl.gz or .cnf.
    In case of a .cnf, it tries to find the corresponding .pkl.gz
    If more than one dataset exists (i.e. multiple files
    match the pattern {cnf}_*.pkl.gz), one is returned at random.
    :param path:
    :return:tuple of 2 pandas dataframes
    """
    if path.endswith('.pkl.gz'):
        return get_xy_data(pd.read_pickle(path, compression='gzip'))

    if not dataset_exists(path):
        raise ValueError(f"There is no dataset for {path}")

    dataset_files = glob.glob(f'{path}_*.pkl.gz')
    df = pd.read_pickle(random.choice(dataset_files), compression='gzip')

    # breaks into input features & label
    return get_xy_data(df)


def get_xy_data(dataframe):
    """
    Breaks the dataframe into features and label
    and returns these two dataframes
    :param dataframe:
    :return: tuple of dataframes
    """
    # breaks into input features & label
    data_x = dataframe.drop('f', axis=1)
    data_y = dataframe['f']

    return data_x, data_y


def generate_dataset(cnf, solver='unigen', num_positives=500, num_negatives=500, save_dataset=True, overwrite=False):
    """
    Generates a dataset out of a CNF boolean formula
    :param cnf: path to the boolean formula in DIMACS CNF format
    :param solver: unigen or the name of a PySAT solver
    :param num_positives: number of positive samples
    :param num_negatives: number of negative samples
    :param save_dataset: if True, saves the dataset as cnf_solver_pos_neg.pkl.gz, where pos & neg are the actual number of samples
    :param overwrite: if True, overwrites an existing datset
    :return: two dataframes with the input data & labels (X and y in ML library parlance)
    """

    sampler = {
        'unigen': UnigenSampler(),
        'unigen3': Unigen3Sampler()
    }

    positive_sampler = sampler.get(solver, PySATSampler(solver_name=solver))
    positives = positive_sampler.sample(cnf, num_positives)

    # checks the validity of the samples
    if len(positives) == 0:
        print(f"WARNING: No positives samples for {cnf}. Returning empty dataset...")
        return [], []

    negatives = negative_sampler.uniformly_negative_samples(cnf, num_negatives)
    if len(negatives) == 0:
        print("WARNING: No negative samples for {cnf}. Returning empty dataset...")
        return [], []
    df = prepare_dataset(positives, negatives)

    print(f'{len(df)} instances generated for {cnf}')
    if save_dataset:
        dataset_output = f'{cnf}_{solver}_{len(positives)}_{len(negatives)}.pkl.gz'
        if not overwrite and os.path.exists(dataset_output):
            print(f'Output "{dataset_output}" already exists, will not save.')
        else:
            print(f'Saving dataset to {dataset_output}.')
            df.to_pickle(dataset_output, compression='gzip')

    # breaks into input features & label
    data_x = df.drop('f', axis=1)
    data_y = df['f']

    return data_x, data_y


def cli_generate_dataset(*cnf, solver='unigen', failsafe_solver='Glucose3', num_positives=0, num_negatives=0,
                         save_dataset=True, overwrite=False, proportion=None, check_existing=False):
    """
    This function just calls 'generate_dataset' but does not return data for cleaner use with fire.Fire
    :param cnf: path to the boolean formula in DIMACS CNF format
    :param solver: unigen or the name of a PySAT solver
    :param failsafe_solver: if the main solver fails, use this to generate samples
    :param num_positives: number of positive samples
    :param num_negatives: number of negative samples
    :param save_dataset: if True, saves the dataset as cnf_solver_pos_neg.pkl.gz, where pos & neg are the actual number of samples
    :param overwrite: if True, overwrites an existing datset
    :param proportion: fraction of 2^#vars that will correspond to the dataset size
    (can use string 'quadratic' or 'loglike' for vars^2 or min(2^n,5000*2^(log(n/10)-1)), respectively
    :return:
    """
    dataset_function = get_dataset if check_existing else generate_dataset
    for formula in cnf:
        if proportion == 'quadratic':  # dataset size will be vars^2
            f = CNF(formula)
            num_positives = num_negatives = int((f.nv**2) / 2)
        elif proportion == 'loglike':   # dataset size grows log-scale with the number of possible assignments
            f = CNF(formula)
            num_positives = num_negatives = int(min(2**f.nv,  5000*2**(log10(f.nv)-1))) // 2
        dataset_function(formula, solver, num_positives, num_negatives, save_dataset, overwrite)
        # checks if the main solver has failed, if so, use the failsafe solver
        if len(glob.glob(f'{formula}*.pkl.gz')) == 0:
            print(f'WARNING: {solver} did not sample for {formula}. Using {failsafe_solver}.')
            dataset_function(formula, failsafe_solver, num_positives, num_negatives, save_dataset, overwrite)


def recursive(basedir, solver='unigen', failsafe_solver='Glucose3', num_positives=5000,
                                     num_negatives=5000,
                                     save_dataset=True, overwrite=False, proportion=None):
    """
    Traverses the specified basedir recursively, generating a dataset for each formula
    in the subdirectories (TODO check if basedir is included)
    :return:
    """

    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk(basedir):
        print(f'{root}')
        if any([re.search('\.cnf', f) is not None for f in files]):  # checks whether there are cnf files in the current dir
            print(f'.cnf files found on {root}. Generating datasets...')

            # prepends current dir to each file (which are only the file names)
            file_list = [os.path.join(root, f) for f in files]

            cli_generate_dataset(
                *file_list, solver=solver, failsafe_solver=failsafe_solver,
                num_positives=num_positives, num_negatives=num_negatives,
                save_dataset=save_dataset, overwrite=overwrite, proportion=proportion,
                check_existing=True
            )


if __name__ == '__main__':
    fire.Fire()


#  generate dataset for phase transition:
#  for d in instances/phase/v*/*/; do echo $d; srun --resv-ports  --nodes 1 --ntasks=1 -c 16 python mlbf/dataset.py $d/*.cnf; done
# loglike in shared:
# for d in instances/phase/vZZ/*/; do echo $d; srun --resv-ports  --nodes 1 --ntasks=1 -c 32 python mlbf/dataset.py $d/*.cnf --proportion loglike; done

