import os

import fire
import main
import tarfile
from multiprocessing import Pool, cpu_count
from functools import partial
import datetime


def run(instances, output='out.csv', extraction_point='/tmp/satinstances',
        cvfolds=5, model='MLP', n_cores=cpu_count(),
        mlp_layers=[200, 100], mlp_activation='relu',
        solver='unigen', save_dataset=False):
    """
    Extracts all files in a tar.gz file and runs the experiment for each one of them
    :param solver: name of the SAT solver to find the satisfying samples
    :param instances: .tar.gz file containing the .cnf instances
    :param cvfolds: number of folds for cross-validation
    :param model: learner (MLP, DecisionTree or RF for random forest)
    :param mlp_layers: list with #neurons in each hidden layer (from command line, pass it without spaces e.g. [200,100,50])
    :param mlp_activation: MLP's activation function
    :param output: path to write results to (csv format)
    :param extraction_point: point to extract the cnf instances
    :param n_cores: number of cpu cores to use with parallel computation
    :param save_dataset: whether to save the dataset generated from the cnf files
    :return:
    """

    if n_cores > cpu_count():
        n_cores = cpu_count()

    # creates the extraction point if it does not exist
    os.makedirs(extraction_point, exist_ok=True)

    # extracts the instances
    print(f'Extracting contents of {instances} to {extraction_point}')
    with tarfile.open(instances) as tf:
        tf.extractall(extraction_point)

    # run each instance in the extraction point (finds all files there recursively)
    data_files = [os.path.join(root, file) for root, dirs, files in os.walk(extraction_point) for file in files]

    print(f'{len(data_files)} files are at {extraction_point}.')

    start = datetime.datetime.now()

    with Pool(processes=n_cores) as pool:
        print(f'Running with {n_cores} cpu cores. Parent process id {os.getppid()}.')
        pool.map(partial(main.evaluate, output=output,
                         cvfolds=cvfolds, model=model, mlp_layers=mlp_layers,
                         mlp_activation=mlp_activation, solver=solver,
                         save_dataset=save_dataset), data_files)
        pool.close()

        print()  # just a newline
    print(f"Finished all {len(data_files)} instances in {(datetime.datetime.now() - start).total_seconds():8.2f}s. ",
          f"Extracted instances and datasets are at {extraction_point}.")
    # shutil.rmtree(extraction_point)
    # print('Done')


if __name__ == '__main__':
    fire.Fire(run)
