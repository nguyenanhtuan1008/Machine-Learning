# Machine-Learning
Machine Learning

## Setup:
conda create -n ml
conda activate ml

##1.1 Gather the data and build cross-validation

- Download this dataset: Categorical Feature Encoding Challenge from https://www.kaggle.com/c/cat-in-the-dat/data then save it into input folder

- Check csv file: 
    head train.csv

- python .\src\create_folds.py

- python .\src\train.py
Result:
0.7437763648654564

##1.2 Building an inference for the machine learning framework


## Ref:
run.bat in window 
set TRAINING_DATA=input\train_folds.csv
set FOLD=0
python -m src.train1

and run.sh linux: set ==> export