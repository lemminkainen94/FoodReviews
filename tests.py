import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import psycopg2
import pytest

from .src.food_rank import FoodRank
from .src.food_reviews_db_conn import FoodReviewsDBConn

with open("../db_con_creds.json") as f:
    creds = json.load(f)

connection = psycopg2.connect(
    user=creds["user"],
    password=creds["pwd"],
    host=creds["host"],
    database=creds["db"],
    port=creds["port"],
)

food = FoodRank(creds)


@pytest.fixture
def get_test_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["testaurant1", "testaurant1", "testaurant2", "testaurant2"],
            "city": ["testville"] * 4,
            "category": [""] * 4,
            "stars": [1, 5, 3, 5],
            "published_at": ["2022-11-11"] * 4,
            "review_text": [
                "disgusting testburger",
                "the best testburger ever",
                "i had testburger",
                "splendid testburger!",
            ],
            "review_id": [1, 2, 3, 4],
            "created": ["2022-11-11"] * 4,
        }
    )


def test_food_query_db():
    df = food.food_db_conn.food_query_db("nothing", "void")
    assert list(df.columns) == [
        "id",
        "name",
        "city",
        "category",
        "stars",
        "published_at",
        "review_text",
        "review_id",
        "created",
    ]


def test_query_place_meal_stats():
    stats = food.food_db_conn.query_place_meal_stats("nothing", "void", "void")
    assert stats[1:4] == (0, 0, 0)


def test_load_model():
    new_food = FoodRank(creds)
    assert "TransformerTBSA" in str(new_food.args.model)
    assert "PreTrainedTokenizer" in str(new_food.args.tokenizer)
    assert new_food.args.model_name == new_food.args.tokenizer.name_or_path


@patch.object(FoodReviewsDBConn, "update_stats")
def test_rank_rests(mock_update_stats, get_test_df):
    np.random.seed(42)
    rest_rank = food._rank_rests(get_test_df, "testburger")
    assert mock_update_stats.call_count == 2
    assert rest_rank[0]["score"] > rest_rank[1]["score"]
    assert rest_rank[0]["name"] == "testaurant2"


@patch.object(FoodReviewsDBConn, "update_stats")
def test_get_rank(mock_update_stats, get_test_df):
    np.random.seed(42)
    res = food._get_rank(get_test_df, "testaurant1", "testburger")
    assert res == {
        "% positive": "50%",
        "name": "testaurant1",
        "score": 0.5928137969881028,
        "total reviews": 2,
    }
    mock_update_stats.assert_called_once()


def test_get_preds_star(get_test_df):
    test_df = get_test_df
    food._get_preds(test_df, "testburger", "star")
    preds = list(test_df["sentiment"])
    assert preds == [-1, 1, 0, 1]


def test_get_preds_model(get_test_df):
    test_df = get_test_df
    food._get_preds(test_df, "testburger", "model")
    preds = list(test_df["sentiment"])
    assert preds == [-1, 1, 0, 1]


def test_database(get_test_df):
    """
    database integration test;
    checks all the db functions step by step:
    1. update the database based on get_test_df fixture
    2. call food_query_db to extract the data added in step one
    3. call query_place_meal_stat to verify the function fetches correct stats
    4. delete the data added in step 1
    """
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO food_reviews
            (id, name, city, category, stars, published_at, review_text, review_id, created)
        VALUES
        (999999996, 'testaurant1', 'testville', '', 1, '2022-11-11', 'disgusting testburger', 1, '2022-11-11'),
        (999999997, 'testaurant1', 'testville', '', 5, '2022-11-11', 'the best testburger ever', 2, '2022-11-11'),
        (999999998, 'testaurant2', 'testville', '', 3, '2022-11-11', 'i had testburger', 3, '2022-11-11'),
        (999999999, 'testaurant2', 'testville', '', 5, '2022-11-11', 'splendid testburger!', 4, '2022-11-11')
    """
    )
    connection.commit()
    cursor.close()

    np.random.seed(42)
    rest_rank = food._rank_rests(get_test_df, "testburger")

    name_df = food.food_db_conn.food_query_db("testburger", "testville")
    assert (
        (
            name_df[["name", "city", "review_text"]]
            == get_test_df[["name", "city", "review_text"]]
        )
        .all()
        .all()
    )

    stats = food.food_db_conn.query_place_meal_stats(
        "testaurant1", "testburger", "testville"
    )
    assert stats[1:4] == (1, 1, 0)

    cursor = connection.cursor()
    cursor.execute("DELETE FROM place_meal_stats WHERE city = 'testville'")
    connection.commit()
    cursor.execute("DELETE FROM food_reviews WHERE city = 'testville'")
    connection.commit()
    cursor.close()
