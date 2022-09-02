import pandas as pd
import psycopg2

REVIEW_SQL = """
    SELECT * FROM food_reviews
    WHERE review_text LIKE %(food)s AND city = %(city)s
"""

STAT_SQL = """
    SELECT 
        id, restaurant, meal, city, positive_count,
        negative_count, created, neutral_count
    FROM place_meal_stats
    WHERE meal = %(meal)s AND restaurant = %(restaurant)s AND city = %(city)s
"""

INSERT_STAT_SQL = """
    INSERT INTO place_meal_stats(
        restaurant, meal, city, positive_count,
        negative_count, created, neutral_count
    ) VALUES (
        %(restaurant)s, %(meal)s, %(city)s, %(pos_count)s,
        %(neg_count)s, %(created)s, %(neutral_count)s
    )
"""

UPDATE_STAT_SQL = """
    UPDATE place_meal_stats
    SET restaurant = %(restaurant)s, 
        meal = %(meal)s,  
        city = %(city)s, 
        positive_count = %(pos_count)s,
        negative_count = %(neg_count)s, 
        created = %(created)s, 
        neutral_count = %(neutral_count)s
    WHERE id = %(id)s
"""


class FoodReviewsDBConn:
    """
    this class handles a connection to the food reviews database
    and provides functions for retrieving and updating it
    """

    def __init__(self, creds):
        """
        creds - dictionary with credentials
        must contain the following fields: user, pwd, host, db, port
        """
        self.connection = psycopg2.connect(
            user=creds["user"],
            password=creds["pwd"],
            host=creds["host"],
            database=creds["db"],
            port=creds["port"],
        )

    def food_query_db(self, food, city):
        cur = self.connection.cursor()
        cur.execute(REVIEW_SQL, {"food": f"%{food}%", "city": city})

        cols = [
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
        df = pd.DataFrame(cur.fetchall(), columns=cols)
        df.published_at = df.published_at.apply(pd.Timestamp)
        df.created = df.created.apply(pd.Timestamp)
        df.review_text = df.review_text.fillna("").apply(lambda x: x.lower())
        print(df.shape)

        cur.close()

        return df

    def query_place_meal_stats(self, place, meal, city):
        cur = self.connection.cursor()
        cur.execute(STAT_SQL, {"meal": meal, "restaurant": place, "city": city})

        cols = [
            "id",
            "restaurant",
            "meal",
            "city",
            "positive_count",
            "negative_count",
            "created",
            "neutral_count",
        ]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
        df.created = df.created.apply(pd.Timestamp)

        cur.close()

        if len(df) == 0:
            return -1, 0, 0, 0, None

        return (
            int(df["id"].iloc[0]),
            int(df.positive_count.iloc[0]),
            int(df.negative_count.iloc[0]),
            int(df.neutral_count.iloc[0]),
            df.created.iloc[0],
        )

    def update_stats(self, row):
        cur = self.connection.cursor()
        if row["id"] == -1:
            cur.execute(INSERT_STAT_SQL, row)
        else:
            cur.execute(UPDATE_STAT_SQL, row)
        self.connection.commit()
        cur.close()

    def close(self):
        self.connection.close()
