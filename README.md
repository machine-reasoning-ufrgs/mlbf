# MLBF
Machine Learning on Boolean Formulas

Companion code of the paper "Understanding Boolean Function Learnability on Deep Neural Networks"

Tested on Ubuntu Linux 18.04.

## Installation

You need python 3.8 and the following libraries (commands to install assume a conda environment):

* scikit-learn & pandas (`conda install -c anaconda scikit-learn pandas`)
* fire (`conda install -c conda-forge fire`)
* pysat (`pip install python-sat[pblib,aiger]`)


## Execution

- Replicating Section 4 experiments:
 `python mlbf/main.py *.cnf --output=out.csv`

This will generate a dataset, run 5-fold cross validation of a 2-hidden layer MLP (200 and 100 neurons, respectively) for each `.cnf` file, writing the statistics on `out.csv`. If the dataset was already generated, it will be used. Run `python mlbf/main.py -- --help` for additional options.
The resulting `.csv` contains a formula per line, with the associated statistics. 
See the companion `data` folder for the instances.
 

- Replicating Section 5 experiments:
`python mlbf/mlpsize.py mlpsize *.cnf --output=out.csv`

This will generate a dataset and test how many neurons in a single-hidden-layer MLP are required for perfect accuracy on 5-fold CV  for each `.cnf` file, writing the statistics on `out.csv`. 
If the dataset was already generated, it will be used. Run `python mlbf/mlpsize.py -- --help` for additional options.
The resulting `.csv` contains a formula per attempted number of neurons (from 1 to 512) per line, with the associated statistics.
Formulas whose lines stop before 512 neurons have been learned with less and not tested further. 
See the companion `data` folder for the instances.

Note: you might want to run `python mlbf/sample_small.py *.cnf` to make sure that all formulas have the companion dataset 
(the script will generate the dataset for those formulas that Unigen2 missed).

