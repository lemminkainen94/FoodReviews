"""
Uses hyperopt to search for the best model, then logs the best model. Builds a Prefect Deployment
"""
import argparse
import os
import pickle
import time
import warnings
from datetime import timedelta

import mlflow
import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB


def mlflow_setup():
    REMOTE_TRACKING_IP = os.getenv("REMOTE_IP", "localhost")
    MLFLOW_TRACKING_URI = f"http://{REMOTE_TRACKING_IP}:5000"

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    return client


def save_preprocessors(count_vect, tfidf):
    with open("preprocessors.pkl", "wb") as file:
        pickle.dump((count_vect, tfidf), file)


def transform_data(df_train, df_test, vectorizer_params):
    count_vect = CountVectorizer(**vectorizer_params)
    word_idx = count_vect.fit_transform(df_train.target_text.values)

    tfidf_transformer = TfidfTransformer().fit(word_idx)

    x_train = tfidf_transformer.transform(word_idx)
    x_test = tfidf_transformer.transform(
        count_vect.transform(df_test.target_text.values)
    )

    return (
        x_train,
        x_test,
        df_train.sentiment,
        df_test.sentiment,
        count_vect,
        tfidf_transformer,
    )


def sgd_hyperopt_search(x_train, y_train):
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "sgd")
            mlflow.log_params(params)
            sgd = SGDClassifier(max_iter=1000, **params)
            kfold = StratifiedKFold(n_splits=5, shuffle=True)
            f1_mac = cross_val_score(
                sgd, x_train, y_train, cv=kfold, scoring="f1_macro", verbose=False
            ).mean()
            mlflow.log_metric("f1_macro", f1_mac)

        return {"loss": -f1_mac, "status": STATUS_OK}

    loss = ["hinge", "log", "perceptron", "modified_huber"]
    penalty = ["l1", "l2"]

    search_space = {
        "loss": hp.choice("loss", loss),
        "alpha": hp.loguniform("alpha", -8, -1),
        "penalty": hp.choice("penalty", penalty),
        "tol": hp.loguniform("tol", -4, 0),
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=1000,
        trials=Trials(),
    )

    best_result["loss"] = loss[best_result["loss"]]
    best_result["penalty"] = penalty[best_result["penalty"]]
    print(best_result)


def retrieve_best_params(exp_name, client):
    experiment = client.get_experiment_by_name(exp_name)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.f1_macro DESC"],
    )
    params = best_run[0].data.params
    params["alpha"] = float(params["alpha"])
    params["tol"] = float(params["tol"])
    return params


def train_best(X, y, clf, model_name, exp_name, client):
    mlflow.set_tag("model", f"{model_name}_best")
    clf.fit(X, y)

    with open(f"{model_name}_best.pkl", "wb") as f:
        pickle.dump(clf, f)

    mlflow.log_artifact("preprocessors.pkl", artifact_path="model")
    mlflow.log_artifact(f"{model_name}_best.pkl", artifact_path="model")

    experiment_id = client.get_experiment_by_name(exp_name).experiment_id
    all_runs = client.search_runs(experiment_ids=experiment_id)
    for mlflow_run in all_runs:
        client.delete_run(mlflow_run.info.run_id)


def eval(X, y, clf):
    predicted = clf.predict(X)
    f1_mac = f1_score(y, predicted, average="macro")
    acc = accuracy_score(y, predicted)
    mlflow.log_metric("f1_macro", f1_mac)
    mlflow.log_metric("accuracy", acc)


def parse_args():
    parser = argparse.ArgumentParser("FoodReviews")
    parser.add_argument(
        "--data_train",
        type=str,
        default="./data/data_train.tsv",
        help="path to train data file",
    )
    parser.add_argument(
        "--data_test",
        type=str,
        default="./data/data_test.tsv",
        help="path to test data file",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="NB",
        help="Sklearn classifier. Can be one of NB(default), SGD",
    )

    return parser.parse_args()


def main():
    client = mlflow_setup()

    args = parse_args()
    df_train = pd.read_csv(args.data_train, sep="\t")
    df_test = pd.read_csv(args.data_test, sep="\t")
    df_train["target_text"] = df_train.apply(
        lambda x: x.text + " [SEP] " + x.target, axis=1
    )
    df_test["target_text"] = df_test.apply(
        lambda x: x.text + " [SEP] " + x.target, axis=1
    )
    vectorizer_params = dict(ngram_range=(1, 2), max_df=0.8)

    TBSA_EXPERIMENT_NAME = f"tbsa_train_{args.data_train.split('/')[-1][:-4]}"
    EXPERIMENT_NAME = f"tbsa_eval_{args.data_test.split('/')[-1][:-4]}"

    mlflow.set_experiment(TBSA_EXPERIMENT_NAME)

    x_train, x_test, y_train, y_test, cvect, tfidf_trans = transform_data(
        df_train, df_test, vectorizer_params
    )
    save_preprocessors(cvect, tfidf_trans)
    clf = MultinomialNB()
    params = {}
    if args.model_name == "SGD":
        sgd_hyperopt_search(x_train, y_train)
        params = retrieve_best_params(TBSA_EXPERIMENT_NAME, client)
        clf = SGDClassifier(max_iter=1000, **params)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        train_best(x_train, y_train, clf, args.model_name, TBSA_EXPERIMENT_NAME, client)
        mlflow.log_params(params)

        mlflow.sklearn.autolog()
        eval(x_test, y_test, clf)


if __name__ == "__main__":
    main()
