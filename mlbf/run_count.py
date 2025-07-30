import subprocess

import fire
import os
from multiprocessing import Pool, cpu_count
from datetime import datetime
from functools import partial
import pickle
import gzip
from sklearn.model_selection import train_test_split
import re


def write_header(output):
    """
    Creates the header in the output file if it does not exist
    :param output: path to the output file
    :return:
    """
    if output is not None and not os.path.exists(output):
        with open(output, 'w') as out:
            out.writelines(['dataset,vars,clauses,positives_train,negatives_train,',
                            'positives_test,negatives_test,count\n'])


def sat_model_count(cnf_file, counter_path):
    """
    Get the number of total solutions using external model count

    :param cnf_file: cnf file used to calculate the total number of solutions
    :param counter_path: path to the binary model count
    :return:
    """
    cmd_count = (f'{counter_path} -cs 2000 -t 5000  -seed 1000 -pol polaritycache -LSO 5000 -delta 0.05 '
                 f'-p -maxdec 5000000 500 -m 1 {cnf_file}')

    output_data = subprocess.run(cmd_count.split(' '), capture_output=True, text=True)

    return int(re.findall(r"s mc (\d+)", output_data.stdout)[0])


def process_count(dataset_file, output_file, counter_path):
    print(f'Working on file {os.path.basename(dataset_file)} with process {os.getpid()}.')
    cnf_file = dataset_file.split('_unigen')[0]

    write_header(output_file)

    with open(output_file, 'a') as out_f:
        with gzip.open(dataset_file, 'rb') as f:
            data = pickle.load(f)
            data_x, data_y = data.iloc[:, :-1], data.iloc[:, [-1]]

            X_train, X_test, y_train, y_test = train_test_split(
                data_x, data_y, test_size=0.25, stratify=data_y, shuffle=True,
                random_state=202205)

            f_vars = data_x.shape[1]
            f_clauses = int(re.match(r".*c(\d+)\.cnf.*", dataset_file).group(1))
            positives_train = sum(y_train['f'] == 1)
            negatives_train = sum(y_train['f'] == 0)
            positives_test = sum(y_test['f'] == 1)
            negatives_test = sum(y_test['f'] == 0)
            model_count = sat_model_count(cnf_file, counter_path=counter_path)

            out_f.writelines([
                f'{os.path.basename(cnf_file)},{f_vars},{f_clauses},{positives_train},{negatives_train},',
                f'{positives_test},{negatives_test},{model_count}\n']
            )

            out_f.flush()
            os.fsync(out_f.fileno())


def run_count(cnf_path, output='out.csv', n_cores=cpu_count(),
              counter_path=os.path.join("ganak", "ganak")):
    """
    Run model count and calculate other counts to explore information about
    CNF formulas

    :param counter_path: path to the model count binary file
    :param cnf_path: point to input folder with the cnf.pkl.gz instances
    :param output: path to write results to (csv format)
    :param n_cores: number of cpu cores to use with parallel computation
    :return:
    """
    if n_cores > cpu_count():
        n_cores = cpu_count()

    all_files = [os.path.join(root, file) for root, dirs, files in os.walk(cnf_path) for file in files]

    data_files = [file for file in all_files if file.endswith('.pkl.gz')]
    cnf_files = [file for file in all_files if file.endswith('.cnf')]

    print(f'Processing {len(cnf_files)} cnf files and {len(data_files)} data files from {cnf_path}.')

    if os.path.exists(output):
        os.remove(output)

    start = datetime.now()

    with Pool(processes=n_cores) as pool:
        print(f'Running with {n_cores} cpu cores. Parent process id {os.getppid()}.')
        pool.map(partial(process_count, output_file=output, counter_path=counter_path), data_files)
        pool.close()

        print()  # just a newline
    print(f"Finished all {len(data_files)} instances in {(datetime.now() - start).total_seconds():8.2f}s. ",
          f"Extracted instances and datasets are at {cnf_path}.")


if __name__ == '__main__':
    fire.Fire(run_count)
