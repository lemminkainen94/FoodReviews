import os

os.system(
    "python src/model_training.py data/data_full.tsv --epochs 30 --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --model_name yangheng/deberta-v3-base-absa-v1.1"
)
# os.system("python src/model_training.py data/data_train_occ.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --model_name yangheng/deberta-v3-base-absa-v1.1")
