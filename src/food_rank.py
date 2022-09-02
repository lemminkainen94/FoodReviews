import json
import os
from datetime import datetime
from types import SimpleNamespace

import torch
from flask import Flask, request
from scipy.stats import beta
from transformers import AutoTokenizer

from .data_loader import create_data_loader
from .eval import pred
from .food_reviews_db_conn import FoodReviewsDBConn
from .transformer_model import TransformerTBSA


class FoodRank:
    def __init__(self, creds):
        self.food_db_conn = FoodReviewsDBConn(creds)
        self._load_model()

    def get_rest_rank(self, content, prior=2, top_res=10):
        if "mode" in content and content["mode"] == "conservative":
            prior = 5
        if "top_res" in content:
            top_res = int(top_res)

        df = self.food_db_conn.food_query_db(content["food"], content["city"])

        return json.dumps(
            self._rank_rests(df, content["food"], top_res=top_res, prior=prior)
        )

    def _load_model(self):
        with open(
            os.getenv("BEST_MODEL_ARGS", "./models/best_model_args.json")
        ) as arg_f:
            self.args = json.load(arg_f)

        self.args = SimpleNamespace(**self.args)
        self.args.shuffle = False
        self.args.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

        self.args.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)

        self.args.model = TransformerTBSA(self.args)
        if self.args.in_model:
            self.args.model.load_state_dict(torch.load(self.args.in_model))
        self.args.model.to(self.args.device)

    def _rank_rests(self, name_df, meal, prior=2, top_res=10, method="model"):
        rests = []

        for rest in name_df.name.unique():
            rests.append(
                self._get_rank(name_df, rest, meal, method=method, prior=prior)
            )

        rests.sort(key=lambda x: x["score"], reverse=True)
        return rests[:top_res]

    def _get_rank(self, name_df, rest, meal, prior=2, method="model"):
        rest_df = name_df[name_df.name == rest]
        city = rest_df.city.iloc[0]
        (
            stat_id,
            pos_count,
            neg_count,
            neutral_count,
            last_update,
        ) = self.food_db_conn.query_place_meal_stats(rest, meal, city)
        if last_update:
            rest_df = rest_df[rest_df.created > last_update]

        self._get_preds(rest_df, meal, method)

        pos_count += len(rest_df[rest_df.sentiment == 1])
        neg_count += len(rest_df[rest_df.sentiment == -1])
        neutral_count = len(rest_df[rest_df.sentiment == 0])

        self.food_db_conn.update_stats(
            {
                "id": stat_id,
                "restaurant": rest,
                "meal": meal,
                "city": city,
                "pos_count": pos_count,
                "neg_count": neg_count,
                "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "neutral_count": neutral_count,
            }
        )

        # sample from beta distr to get a score
        # for each restaurant with at least one review mentioning the food
        return {
            "name": rest,
            "total reviews": pos_count + neg_count + neutral_count,
            "% positive": f"{pos_count / (pos_count + neg_count + neutral_count):.0%}",
            "score": beta.rvs(prior + pos_count, prior + neg_count),
        }

    def _get_preds(self, name_df, meal, method="model"):
        if method == "star":
            name_df["sentiment"] = self._star_preds(name_df)
        else:
            name_df["sentiment"] = self._model_preds(name_df, meal)

    def _star_preds(self, name_df):
        return name_df.stars.apply(self._get_sentiment)

    def _get_sentiment(self, stars):
        if stars < 3:
            return -1
        if stars > 3:
            return 1
        return 0

    def _model_preds(self, df, meal):
        df["target_text"] = df.apply(lambda x: x.review_text + " [SEP] " + meal, axis=1)
        self.args.dl = create_data_loader(df, self.args)
        return pred(self.args)

    def close_conn(self):
        self.food_db_conn.close()
