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