import itertools
import os.path
import pickle
import numpy as np
import gzip
import fire
import datetime

from tqdm import tqdm
from sklearn import model_selection
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score


def write_header(output, k_fold=False):
    """
    Creates the header in the output file if it does not exist
    :param k_fold: header for split data or k-fold cv
    :param output: path to the output file
    :return:
    """
    if output is not None and not os.path.exists(output):
        with open(output, 'w') as out:
            if k_fold:
                out.write('dataset,sampler,learner,cvfolds,mean_acc,std_acc,mean_f1_macro,std_f1,start,finish\n')
            else:
                out.write('dataset,sampler,learner,cvfolds,mean_acc,mean_f1_macro,start,finish\n')


def evaluate_clause(assignment, clause):  # assignment is a -1/+1 string
    for i in clause:
        if (2 * assignment[abs(i) - 1] - 1) * i > 0:  # the literal is satisfied
            return 1
    return 0


def check_clause(clause, x, positive_index):
    for index in positive_index:
        row = x.loc[index]
        # print('encounter a positive assignment at index '+repr(index))
        if evaluate_clause(row, clause=clause) == 0:
            return False
    return True


class LearnByValiant:
    def __init__(self, k=3):
        self.model = []
        self.k = k  # k is the arity of the learnt CNF formula

    def fit(self, x, y):
        n = len(x.columns)
        positive_index = [index for index, _ in y.iterrows() if y.loc[index]['f'] == 1]
        print(f'Ratio of positive assignments {len(positive_index)} / {len(x)} for formulas with {n} variables.')

        clauses = self.possible_clauses(n)
        only_positives = x.loc[positive_index].to_numpy()

        self.model = [c for c in tqdm(clauses, desc=' Learning: ', ascii=True)
                      if not (np.apply_along_axis(evaluate_clause, 1, only_positives, clause=c) == 0).any()]

        print(f'Learned a {self.k}-CNF formula by Valiant\'s algorithm with {len(self.model)} clauses.')

        return self

    def get_params(self, deep=False):
        return {'k': self.k, 'clauses': self.model}

    def print_model(self):
        print(f'Printing model with {len(self.model)} clauses')
        for clause in self.model:
            print(clause)

    def predict(self, x):
        predict_ans = []
        for index, row in tqdm(x.iterrows(), desc=' Predict: ', ascii=True, total=x.shape[0]):
            flag = 1
            for clause in self.model:
                if evaluate_clause(row, clause=clause) == 0:
                    predict_ans.append(0)
                    flag = 0
                    break
            if flag == 1:
                predict_ans.append(1)
        return predict_ans

    def possible_clauses(self, n):
        combs = itertools.combinations(range(1, n + 1), self.k)
        allClauses = []
        signs = list(itertools.product([-1, 1], repeat=self.k))
        for positiveClause in combs:
            for signTuple in signs:
                allClauses.append([positiveClause[i] * signTuple[i] for i in range(self.k)])
        return allClauses


def evaluate(dataset, output='out.csv', cvfolds=5, cnf_arity=3):
    """
    Learn Boolean functions by CNF using Valiant's Algorithm

    :param dataset: path to the mlbf output dataset 'output.cnf.pkl.gz'
    :param output: path to output file
    :param cvfolds: number of folds for cross-validation
    :param cnf_arity: arity of CNF formula
    :param debug: show model fit updates
    :return:
    """

    print(f'Working on file {os.path.basename(dataset)} with process {os.getpid()}.')

    start = datetime.datetime.now()

    learner = LearnByValiant(cnf_arity)

    write_header(output, cvfolds > 1)

    with open(output, 'a') as out_data:
        with gzip.open(dataset, 'rb') as f:
            data = pickle.load(f)
            data_x, data_y = data.iloc[:, :-1], data.iloc[:, [-1]]

            if cvfolds == 1:
                X_train, X_test, y_train, y_test = train_test_split(
                    data_x, data_y, test_size=0.25, stratify=data_y, shuffle=True,
                    random_state=202205)

                model_fit = learner.fit(X_train, y_train)
                model_predict = model_fit.predict(X_test)
                acc, f1 = accuracy_score(y_test, model_predict), f1_score(y_test, model_predict)

                finish = datetime.datetime.now()

                model_str = os.path.basename(dataset).split('_unigen')[0]

                out_data.write(
                    f'{model_str},,Valiant,{cvfolds},{acc},{f1},{start},{finish}\n'
                )
            else:
                scoring = ['accuracy', 'f1_macro']
                kf = model_selection.KFold(n_splits=cvfolds, random_state=None)

                scores = cross_validate(learner, data_x, data_y, cv=kf, scoring=scoring)
                acc, f1 = np.mean(scores['test_accuracy']), np.mean(scores['test_f1_macro'])
                std_acc, std_f1 = np.std(scores['test_accuracy']), np.std(scores['test_f1_macro'])

                finish = datetime.datetime.now()

                model_str = os.path.basename(dataset).split('_unigen')[0]

                out_data.write(
                    f'{model_str},,Valiant,{cvfolds},{acc},{std_acc},{f1},{std_f1},{start},{finish}\n'
                )

            out_data.flush()
            os.fsync(out_data.fileno())
            print(f'acc = {acc}, f1 = {f1}')


if __name__ == '__main__':
    fire.Fire(evaluate)
    print("Done")
