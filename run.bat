set TRAINING_DATA=input/train_folds.csv
set FOLD=0
set MODEL=$1

python -m src/train.py
pause