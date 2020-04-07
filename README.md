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
run.bat
    set TRAINING_DATA=input\train_folds.csv
    set TEST_DATA=input\test.csv

    set MODEL=%1

    set FOLD=0 
    python -m src.train
    set FOLD=1 
    python -m src.train
    set FOLD=2 
    python -m src.train
    set FOLD=3 
    python -m src.train
    set FOLD=4 
    python -m src.train

    python -m src.predict

.\run.bat randomforest

## 2. Cross Validation Framework

- dataset for binary_classification and multi class classification same as above
- dataset for single_col_classification is https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

## 3. Handling categorical features in ML problems

- Dataset : https://www.kaggle.com/c/cat-in-the-dat-ii

- count number rows csv:
    wc -l ../input/train_cat.csv (linux only)

## Ref:
run.bat in window 
set TRAINING_DATA=input\train_folds.csv
set FOLD=0
python -m src.train1

and run.sh linux: set ==> export