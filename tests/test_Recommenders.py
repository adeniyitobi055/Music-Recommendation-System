#!/usr/bin/python3
"""
test_Recommenders module
"""
import unittest
import recommender
import numpy as np
import pandas


class TestItemSimilarityRecommender(unittest.TestCase):
    def setUp(self):
        """ Set up the necessary data for testing """
        self.recommender = ItemSimilarityRecommender
        self.recommender.train_data = pandas.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3],
            'item_id': ['A', 'B', 'A', 'C', 'B', 'C']
        })
        self.recommender.user_id = 'user_id'
        self.recommender.item_id = 'item_id'

    def test_get_user_items(self):
        """ Test case for a user with interactions """
        user_items = self.recommender.get_user_items(1)
        expected_items = ['A', 'B']
        self.assertEqual(user_items, expected_items)

        """ Test case for a user with no interactions """
        user_items = self.recommender.get_user_items(4)
        expected_items = []
        self.assertEqual(user_items, expected_items)

        """ Test case for a non-existent user """
        user_items = self.recommender.get_user_items(5)
        expected_items = []
        self.assertEqual(user_items, expected_items)


if __name__ == '__main__':
    unittest.main()
