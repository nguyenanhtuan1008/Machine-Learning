set TRAINING_DATA=input\train_folds.csv
set FOLD=0
set MODEL=%1

python -m src.train1

REM for linux
REM export TRAINING_DATA=input\train_folds.csv
REM export FOLD=0
REM export MODEL=$1

REM python -m src.train1