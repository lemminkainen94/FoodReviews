import json

from flask import Flask, request

from src.food_rank import FoodRank

with open("../db_con_creds.json") as f:
    creds = json.load(f)

app = Flask("food-reviews")


@app.route("/foodReviews", methods=["POST"])
def get_top_places():
    content = request.json

    return app.config["ranker"].get_rest_rank(content)


if __name__ == "__main__":
    ranker = FoodRank(creds)
    app.config["ranker"] = ranker
    app.run(debug=True, host="0.0.0.0", port=9696)
    ranker.close_conn()
