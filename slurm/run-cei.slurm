#!/bin/bash
#SBATCH --job-name=mlbf-valiant
#SBATCH --mail-type=END,FAIL                    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=mnicolau@inf.ufrgs.br       # Where to send mail
#SBATCH --partition=cei
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

DIR_TASK="$HOME/learnability"
DIR_NODE="${SCRATCH}learnability"
DIR_RES="$DIR_NODE/outputs"

mkdir -p $DIR_RES

cp -r $DIR_TASK/* $DIR_NODE

echo "Date                      = $(date)"
echo "Hostname                  = $(hostname -s)"
echo "Working Directory         = $(pwd)"
echo "Task Directory            = $DIR_TASK"
echo "Node Directory            = $DIR_NODE"
echo "Results Directory         = $DIR_RES"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

echo "Setting HOME to $DIR_NODE/mlbf"
cd $DIR_NODE/mlbf

#srun python mlbf/run_instances.py instances/valiant_mlbf/v100c450.tar.gz --cvfolds=1 --output=$DIR_RES/cei_v100c450_cv1.csv
#srun python mlbf/run_instances.py instances/valiant_mlbf/v100c431.tar.gz --cvfolds=1 --output=$DIR_RES/cei_v100c431_cv1.csv
#srun python mlbf/run_instances.py instances/valiant_mlbf/v100c420.tar.gz --cvfolds=1 --output=$DIR_RES/cei_v100c420_cv1.csv
#srun python mlbf/run_instances.py instances/valiant_mlbf/v100c400.tar.gz --cvfolds=1 --output=$DIR_RES/cei_v100c400_cv1.csv
#srun python mlbf/run_instances.py instances/valiant_mlbf/v100c350.tar.gz --cvfolds=1 --output=$DIR_RES/cei_v100c350_cv1.csv

srun python mlbf/run_learn_valiant.py instances/valiant_mlbf/v100c450.tar.gz --cvfolds=1 --output=$DIR_RES/cei_v100c450_cv1_valiant.csv
#srun python mlbf/run_learn_valiant.py instances/valiant_mlbf/v100c431.tar.gz --cvfolds=1 --output=$DIR_RES/cei_v100c431_cv1_valiant.csv
#srun python mlbf/run_learn_valiant.py instances/valiant_mlbf/v100c420.tar.gz --cvfolds=1 --output=$DIR_RES/cei_v100c420_cv1_valiant.csv
#srun python mlbf/run_learn_valiant.py instances/valiant_mlbf/v100c400.tar.gz --cvfolds=1 --output=$DIR_RES/cei_v100c400_cv1_valiant.csv
#srun python mlbf/run_learn_valiant.py instances/valiant_mlbf/v100c350.tar.gz --cvfolds=1 --output=$DIR_RES/cei_v100c350_cv1_valiant.csv

echo "Listing files at $DIR_RES"

ls $DIR_RES

echo "Copying files from $DIR_RES to $DIR_TASK/data"

cp -r $DIR_RES/* $DIR_TASK/data

echo "Done."

date
