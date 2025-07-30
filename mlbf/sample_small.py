import os
import glob

import dataset
import fire


def sample_small(basedir, positives, negatives, solver='Glucose3', *var_sizes):
    """
    Traverses the basedir, looking for instances without their corresponding dataset.
    Generates the dataset for these, using pysat-based sampler
    :param basedir: directory where the 'phase' dataset is located
    :param positives: number of positive samples to generate
    :param negatives: number of negative samples to generate
    :param solver: sat solver used to enumerate instances
    :param var_sizes: will look for formulas with the number of variables in this list
    :return:
    """
    if len(var_sizes) == 0:
        var_sizes = range(10, 101, 10)

    for v in var_sizes:
        for cnf in glob.glob(f'{basedir}/v{v}/*/sat*.cnf'):
            if len(glob.glob(f'{cnf}*.pkl.gz')) == 0:
                print(f'Sampling for {cnf}')
                dataset.generate_dataset(cnf, solver, num_positives=positives, num_negatives=negatives)


if __name__ == '__main__':
    fire.Fire(sample_small)

# usage example:
# python mlbf/sample_small.py instances/phase --positives 5000 --negatives 5000 --solver Glucose3 10 20 30

# 20k fix:
# python mlbf/sample_small.py instances/phase_20k-loglike/ --positives 10000 --negatives 10000 --solver Glucose3 NUMBER
