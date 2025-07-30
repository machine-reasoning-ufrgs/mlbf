import datetime
import os
import fire

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score

import dataset
import numpy as np


def evaluate(cnf, solver='unigen', output='out.csv', cvfolds=5, model='MLP',
             mlp_layers=[200, 100], mlp_activation='relu', save_dataset=True):
    """
    Runs the prototype, executing the following steps:

    Receives a boolean formula specified in CNF format,
    Generates many sat and unsat samples for the formula,
    Trains a classifier on this dataset,
    Writes performance metrics to the standard output

    :param cnf: path to the boolean formula in CNF (Dimacs) format (see https://people.sc.fsu.edu/~jburkardt/data/cnf/cnf.html)
    :param solver: name of the SAT solver to find the satisfying samples
    :param output: path to output file
    :param cvfolds: number of folds for cross-validation
    :param model: learner (MLP or DecisionTree)
    :param mlp_layers: list with #neurons in each hidden layer (from command line, pass it without spaces e.g. [200,100,50])
    :param mlp_activation: MLP's activation function
    :param save_dataset: if True, saves the dataset as 'input.cnf.pkl.gz'
    :return:
    """

    print(f'Working on file {os.path.basename(cnf)} with process {os.getpid()}.')

    start = datetime.datetime.now()

    data_x, data_y = dataset.get_dataset(cnf, solver)

    if len(data_x) < 100:
        print(f'{cnf} has {len(data_x)} instances, which is less than 100 (too few to learn). Aborting.')
        return

    # change class_weight to accomodate for imbalanced dataset+
    learner = DecisionTreeClassifier(criterion='entropy', )
    if model == 'MLP':
        learner = MLPClassifier(hidden_layer_sizes=mlp_layers, activation=mlp_activation)
    if model == 'RF':
        learner = RandomForestClassifier()

    write_header(output, cvfolds > 1)

    with open(output, 'a') as outstream:
        # gathers accuracy and precision by running the model and writes it to output
        # print(scores)

        model_str = model if model != 'MLP' else f'{model}_{mlp_activation}_{"-".join([str(x) for x in mlp_layers])}'

        if cvfolds == 1:
            X_train, X_test, y_train, y_test = train_test_split(
                data_x, data_y, test_size=0.25, stratify=data_y, shuffle=True,
                random_state=202205)

            model_fit = learner.fit(X_train, y_train)
            acc, f1 = accuracy_score(y_test, model_fit.predict(X_test)), f1_score(y_test, model_fit.predict(X_test))

            finish = datetime.datetime.now()
            outstream.write(
                f'{os.path.basename(cnf)},{solver},{model_str},{acc},{f1},{start},{finish}\n')
        else:
            scoring = ['accuracy', 'f1_macro']
            scores = cross_validate(learner, data_x, data_y, cv=cvfolds, scoring=scoring)

            acc, f1 = np.mean(scores['test_accuracy']), np.mean(
                scores['test_f1_macro'])  # run_model(learner, data_x, data_y, splitter)
            std_acc, std_f1 = np.std(scores['test_accuracy']), np.std(scores['test_f1_macro'])
            finish = datetime.datetime.now()

            outstream.write(
                f'{os.path.basename(cnf)},{solver},{model_str},{cvfolds},{acc},{std_acc},{f1},{std_f1},{start},{finish}\n')


def write_header(output, k_fold=False):
    """
    Creates the header in the output file if it does not exist
    :param output: path to the output file
    :param k_fold: header for split data or k-fold cv
    :return:
    """
    if output is not None and not os.path.exists(output):
        with open(output, 'w') as out:
            if k_fold:
                out.write('dataset,sampler,learner,cvfolds,mean_acc,std_acc,mean_f1_macro,std_f1,start,finish\n')
            else:
                out.write('dataset,sampler,learner,mean_acc,mean_f1_macro,start,finish\n')


def run_model(model, data_x, data_y, splitter):
    """
    Runs a machine learning model in the specified dataset.
    For each train/test split on the dataset, it outputs the number
    of samples, precision and accuracy to the standard output.

    :param model: a learning algorithm that implements fit and predict methods
    :param data_x: instances of the dataset
    :param data_y: labels of the dataset
    :param splitter: an object that implements split to partition the dataset (useful for cross-validation, for example)
    :return:
    """

    print(f'Running the classifier {type(model)}')

    # prints the header
    print('#instances\tprec\tacc')

    accuracies = []
    precisions = []

    # trains the model for each split (fold)
    for train_index, test_index in splitter.split(data_x, data_y):
        # splits the data frames in test and train
        train_x, train_y = data_x.iloc[train_index], data_y.iloc[train_index]
        test_x, test_y = data_x.iloc[test_index], data_y.iloc[test_index]

        model.fit(train_x, train_y)

        # obtains the predictions on the test set
        predictions = model.predict(test_x)

        # calculates and reports some metrics
        acc = accuracy_score(predictions, test_y)
        prec = precision_score(predictions, test_y, average='macro')

        accuracies.append(acc)
        precisions.append(prec)
        print('{}\t\t{:.3f}\t{:.3f}'.format(len(test_y), prec, acc))

    return np.mean(accuracies), np.mean(precisions)


if __name__ == '__main__':
    fire.Fire(evaluate)
    print("Done")
