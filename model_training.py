import argparse
import os
import random
import warnings
from collections import defaultdict
from time import time

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from transformers import AdamW, AutoTokenizer

from .src.data_loader import create_data_loader
from .src.eval import eval_model, get_predictions
from .src.train import train_epoch
from .src.transformer_model import TransformerTBSA

REMOTE_TRACKING_IP = os.getenv("REMOTE_IP", "localhost")
MLFLOW_TRACKING_URI = f"http://{REMOTE_TRACKING_IP}:5000"

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def run_epoch(args, epoch):
    print(f"Epoch {epoch + 1}/{args.epochs}")
    print("-" * 10)
    train_acc, train_loss, train_avg_losses, avg_accs = train_epoch(args)

    print(f"Train loss {train_loss} accuracy {train_acc}")

    val_f1, val_loss = eval_model(args)

    print(f"Val   loss {val_loss} f1 macro {val_f1}")
    print()

    args.history["train_acc"].append(train_acc)
    args.history["train_loss"] += train_avg_losses
    args.history["val_f1"].append(val_f1)
    args.history["val_loss"].append(val_loss)

    if val_f1 > args.best_f1:
        mlflow.log_metric("f1_macro_val", val_f1)
        if args.out_model is None:
            name = f"{args.data.split('/')[-1][:-4]}_{args.model_name.split('/')[-1]}_{args.batch_size}_{args.lr}"
            args.out_model = f"models/{name}.pt"
        torch.save(args.model.state_dict(), args.out_model)
        args.best_f1 = val_f1
        print("NEW BEST F1:", args.best_f1)
        print("NEW BEST ARGS: ", args.batch_size, args.lr, args.final_dropout)


def parse_args():
    parser = argparse.ArgumentParser("FoodReviews")
    parser.add_argument("data", type=str, help="path to data file")
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Huggingface Transformers pretrained model. distilbert-base-uncased by default",
    )
    parser.add_argument(
        "--in_model",
        default=None,
        help="Path to the model weights to load for training/eval. \n"
        + "Must conform with the chosen architecture",
    )
    parser.add_argument(
        "--out_model",
        default=None,
        help="Name of the trained model (will be saved with that name). Used for training only",
    )
    parser.add_argument(
        "--max_len",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--acc_steps", type=int, default=1, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="number of training epochs"
    )
    parser.add_argument(
        "--final_dropout",
        type=float,
        default=0.2,
        help="droput rate of the final layer",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="data loader batch size"
    )
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument(
        "--test", default=False, action="store_true", help="Whether on test mode"
    )

    return parser.parse_args()


def get_model_params(args):
    return {
        "seed": args.seed,
        "acc_steps": args.acc_steps,
        "data_path": args.data,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "max_len": args.max_len,
        "lr": args.lr,
        "final_dropout": args.final_dropout,
    }


def model_train(args):
    args.loss_fn = nn.CrossEntropyLoss().to(args.device)
    args.model = TransformerTBSA(args)
    args.model.to(args.device)
    args.optimizer = AdamW(args.model.parameters(), lr=args.lr, correct_bias=False)
    args.history = defaultdict(list)

    with mlflow.start_run():
        mlflow.set_tag("model", f"{args.model_name}_{args.epochs}")
        mlflow.log_params(get_model_params(args))

        for epoch in range(args.epochs):
            run_epoch(args, epoch)


def model_eval(args, df):
    args.model = TransformerTBSA(args)
    if args.in_model:
        args.model.load_state_dict(torch.load(args.in_model))
    args.model.to(args.device)

    args.test_dl = create_data_loader(df, args)

    with mlflow.start_run():
        y_review_texts, y_pred, y_test = get_predictions(args)
        acc = accuracy_score(y_test, y_pred)
        f1_mac = f1_score(y_test, y_pred, average="macro")
        mlflow.set_tag("model", f"{args.model_name}_{args.epochs}")
        mlflow.set_tag("model_path", f"{args.in_model}")
        mlflow.log_params(get_model_params(args))
        mlflow.log_metric("f1_macro", f1_mac)
        mlflow.log_metric("accuracy", acc)
        print(acc)
        print(f1_mac)
        print(confusion_matrix(y_test, y_pred))


def main():
    args = parse_args()

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sep = "\t" if args.data.endswith("tsv") else ","
    df = pd.read_csv(args.data, sep=sep)
    df.sentiment += 1
    df["target_text"] = df.apply(lambda x: x.text + " [SEP] " + x.target, axis=1)

    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    ### Model Evaluation
    if args.test:
        args.shuffle = False
        EXPERIMENT_NAME = f"tbsa_eval_{args.data.split('/')[-1][:-4]}"
        mlflow.set_experiment(EXPERIMENT_NAME)
        model_eval(args, df)
    ### Model Training, with some hyperparam tuning
    else:
        args.shuffle = True
        TBSA_EXPERIMENT_NAME = f"tbsa_train_{args.data.split('/')[-1][:-4]}"
        mlflow.set_experiment(TBSA_EXPERIMENT_NAME)
        df_train, df_eval = train_test_split(df, test_size=0.2, random_state=args.seed)

        print(df_train.shape, df_eval.shape)
        args.best_f1 = 0

        args.train_size = len(df_train)
        args.eval_size = len(df_eval)
        args.train_dl = create_data_loader(df_train, args)
        args.eval_dl = create_data_loader(df_eval, args)
        print(args)
        model_train(args)


if __name__ == "__main__":
    main()
