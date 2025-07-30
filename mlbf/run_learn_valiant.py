import os
import fire
import tarfile
import datetime
from multiprocessing import Pool, cpu_count
from functools import partial

import learn_valiant


def run_learn(instances, output='out.csv', extraction_path='/tmp/learn_instances',
              n_cores=cpu_count(), cvfolds=5, arity=3):
    """
    Extracts all files in a tar.gz file and runs the experiment for each one of them
    :param n_cores: number of cpu cores to use with parallel computation
    :param instances: .tar.gz file containing the .cnf instances
    :param cvfolds: number of folds for cross-validation
    :param output: path to write results to (csv format)
    :param extraction_path: point to extract the cnf.pkl.gz instances
    :param arity: arity of CNF formulas
    :return:
    """

    if n_cores > cpu_count():
        n_cores = cpu_count()

    os.makedirs(extraction_path, exist_ok=True)

    # extracts the instances
    print(f'Extracting contents of {instances} to {extraction_path}')
    with tarfile.open(instances) as tf:
        tf.extractall(extraction_path)

    # run each instance in the extraction point (finds all files there recursively)
    data_files = [os.path.join(root, file) for root, dirs, files in os.walk(extraction_path) for file in files]

    print(f'{len(data_files)} files are at {extraction_path}.')

    start = datetime.datetime.now()

    with Pool(processes=n_cores) as pool:
        print(f'Running with {n_cores} cpu cores. Parent process id {os.getppid()}.')
        pool.map(partial(learn_valiant.evaluate, output=output, cvfolds=cvfolds,
                         cnf_arity=arity), data_files)
        pool.close()

        print()  # just a newline
    print(f"Finished all {len(data_files)} instances in {(datetime.datetime.now() - start).total_seconds():8.2f}s. ",
          f"Extracted instances and datasets are at {extraction_path}.")


if __name__ == '__main__':
    fire.Fire(run_learn)
