import os


os.system("python src/model_training.py data/data_train.py --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128")