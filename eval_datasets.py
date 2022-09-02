import os

os.system(
    "python src/model_training.py data/data_test_occ.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --test --model_name yangheng/deberta-v3-base-absa-v1.1 --in_model models/data_train_deberta-v3-base-absa-v1.1_32_5e-05.pt"
)
os.system(
    "python src/model_training.py data/data_test_occ.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --test --model_name yangheng/deberta-v3-base-absa-v1.1 --in_model models/data_train_occ_deberta-v3-base-absa-v1.1_32_5e-05.pt"
)
os.system(
    "python src/model_training.py data/data_test_occ.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --test --model_name yangheng/deberta-v3-base-absa-v1.1 --in_model models/transformer_deberta-v3-base-absa-v1.1_32_5e-05.pt"
)

os.system(
    "python src/model_training.py data/data_test.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --test --model_name yangheng/deberta-v3-base-absa-v1.1 --in_model models/data_train_deberta-v3-base-absa-v1.1_32_5e-05.pt"
)
os.system(
    "python src/model_training.py data/data_test.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --test --model_name yangheng/deberta-v3-base-absa-v1.1 --in_model models/data_train_occ_deberta-v3-base-absa-v1.1_32_5e-05.pt"
)
os.system(
    "python src/model_training.py data/data_test.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --test --model_name yangheng/deberta-v3-base-absa-v1.1 --in_model models/transformer_deberta-v3-base-absa-v1.1_32_5e-05.pt"
)

os.system(
    "python src/model_training.py data/food_reviews_test.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --test --model_name yangheng/deberta-v3-base-absa-v1.1 --in_model models/data_train_deberta-v3-base-absa-v1.1_32_5e-05.pt"
)
os.system(
    "python src/model_training.py data/food_reviews_test.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --test --model_name yangheng/deberta-v3-base-absa-v1.1 --in_model models/data_train_occ_deberta-v3-base-absa-v1.1_32_5e-05.pt"
)
os.system(
    "python src/model_training.py data/food_reviews_test.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --test --model_name yangheng/deberta-v3-base-absa-v1.1 --in_model models/transformer_deberta-v3-base-absa-v1.1_32_5e-05.pt"
)

os.system(
    "python src/model_training.py data/food_reviews_occ.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --test --model_name yangheng/deberta-v3-base-absa-v1.1 --in_model models/data_train_deberta-v3-base-absa-v1.1_32_5e-05.pt"
)
os.system(
    "python src/model_training.py data/food_reviews_occ.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --test --model_name yangheng/deberta-v3-base-absa-v1.1 --in_model models/data_train_occ_deberta-v3-base-absa-v1.1_32_5e-05.pt"
)
os.system(
    "python src/model_training.py data/food_reviews_occ.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --test --model_name yangheng/deberta-v3-base-absa-v1.1 --in_model models/transformer_deberta-v3-base-absa-v1.1_32_5e-05.pt"
)


# 30 epochs best model: deberta
os.system(
    "python src/model_training.py data/data_test_occ.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --test --model_name yangheng/deberta-v3-base-absa-v1.1 --in_model models/data_full_deberta-v3-base-absa-v1.1_32_5e-05.pt"
)
os.system(
    "python src/model_training.py data/data_test.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --test --model_name yangheng/deberta-v3-base-absa-v1.1 --in_model models/data_full_deberta-v3-base-absa-v1.1_32_5e-05.pt"
)
os.system(
    "python src/model_training.py data/food_reviews_test.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --test --model_name yangheng/deberta-v3-base-absa-v1.1 --in_model models/data_full_deberta-v3-base-absa-v1.1_32_5e-05.pt"
)
os.system(
    "python src/model_training.py data/food_reviews_occ.tsv --lr 5e-5 --batch_size 32 --final_dropout 0.2 --max_len 128 --test --model_name yangheng/deberta-v3-base-absa-v1.1 --in_model models/data_full_deberta-v3-base-absa-v1.1_32_5e-05.pt"
)
