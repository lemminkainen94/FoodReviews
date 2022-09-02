import os

os.system(
    "python src/model_training.py data/data_train.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128"
)
os.system(
    "python src/model_training.py data/data_train.tsv --lr 5e-4 --batch_size 32 --final_dropout 0.2 --max_len 128"
)
os.system(
    "python src/model_training.py data/data_train.tsv --lr 5e-3 --batch_size 32 --final_dropout 0.2 --max_len 128"
)

os.system(
    "python src/model_training.py data/data_train.tsv --lr 5e-5 --batch_size 64 --final_dropout 0.2 --max_len 128"
)
os.system(
    "python src/model_training.py data/data_train.tsv --lr 5e-5 --batch_size 128 --final_dropout 0.2 --max_len 128"
)

os.system(
    "python src/model_training.py data/data_train.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.5 --max_len 128"
)

os.system(
    "python src/model_training.py data/data_train.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 32"
)
os.system(
    "python src/model_training.py data/data_train.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 256"
)

os.system(
    "python src/model_training.py data/data_train.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --model_name yangheng/deberta-v3-base-absa-v1.1"
)
os.system(
    "python src/model_training.py data/data_train.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --model_name google/electra-small-discriminator"
)
