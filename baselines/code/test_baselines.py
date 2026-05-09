import os
import unittest

import numpy as np
import pandas as pd

from pf import ProbabilisticFiltering, filter_dataset, load_ratings, load_item_genres


class TestReferenceBaselines(unittest.TestCase):
    def test_filter_contract_shape(self):
        here = os.path.dirname(os.path.abspath(__file__))
        ratings_path = os.path.join(here, "..", "..", "..", "ml-100k", "u.data")

        df = load_ratings(ratings_path)
        df_f = filter_dataset(df)

        self.assertEqual(df_f["user_id"].nunique(), 497)
        self.assertEqual(df_f["item_id"].nunique(), 903)
        self.assertEqual(len(df_f), 79432)

    def test_pf_neighbourhood_excludes_self(self):
        # Tiny synthetic dataset, just to validate neighbourhood bookkeeping.
        df_train = pd.DataFrame(
            [
                {"user_id": 1, "item_id": 10, "rating": 5},
                {"user_id": 1, "item_id": 11, "rating": 4},
                {"user_id": 2, "item_id": 10, "rating": 5},
                {"user_id": 3, "item_id": 10, "rating": 1},
            ]
        )
        item_genres = pd.DataFrame(
            [[0] * 19, [0] * 19],
            index=[10, 11],
            columns=[
                "unknown",
                "Action",
                "Adventure",
                "Animation",
                "Children's",
                "Comedy",
                "Crime",
                "Documentary",
                "Drama",
                "Fantasy",
                "Film-Noir",
                "Horror",
                "Musical",
                "Mystery",
                "Romance",
                "Sci-Fi",
                "Thriller",
                "War",
                "Western",
            ],
        )

        model = ProbabilisticFiltering(theta=0.0)
        model.fit(df_train, item_genres)
        neighs = model._neighbourhoods_for_k(2)  # pylint: disable=protected-access

        # Each user's neighbourhood should never include itself.
        for ui, nbr in enumerate(neighs):
            self.assertTrue(np.all(nbr != ui))
            self.assertLessEqual(len(nbr), 2)

    def test_pf_returns_none_when_no_neighbour_rated_item(self):
        # Build a situation where u1's neighbour exists (u2), but u2 didn't rate item 11.
        df_train = pd.DataFrame(
            [
                {"user_id": 1, "item_id": 10, "rating": 5},
                {"user_id": 1, "item_id": 11, "rating": 4},
                {"user_id": 1, "item_id": 12, "rating": 1},
                {"user_id": 2, "item_id": 10, "rating": 5},
                {"user_id": 2, "item_id": 12, "rating": 2},
            ]
        )
        item_genres = pd.DataFrame(
            [[0] * 19, [0] * 19, [0] * 19],
            index=[10, 11, 12],
            columns=[
                "unknown",
                "Action",
                "Adventure",
                "Animation",
                "Children's",
                "Comedy",
                "Crime",
                "Documentary",
                "Drama",
                "Fantasy",
                "Film-Noir",
                "Horror",
                "Musical",
                "Mystery",
                "Romance",
                "Sci-Fi",
                "Thriller",
                "War",
                "Western",
            ],
        )

        model = ProbabilisticFiltering(theta=0.0)
        model.fit(df_train, item_genres)

        # Predict for user 2 on item 11: neighbour u1 rated item 11 -> should be a number.
        p_ok = model.predict(2, 11, k=1)
        self.assertIsNotNone(p_ok)

        # Predict for user 1 on item 11: neighbour u2 did NOT rate item 11.
        # With the Eq. (19) "term set to zero" interpretation used in pf.py,
        # there are no rated neighbours contributing to the summation -> no prediction.
        p_none = model.predict(1, 11, k=1)
        self.assertIsNone(p_none)


if __name__ == "__main__":
    unittest.main()
