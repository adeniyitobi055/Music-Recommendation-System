#!/usr/bin/python3
"""
test_Recommenders module
"""
import unittest
from unittest import mock
from unittest.mock import patch

import numpy as np
import pandas

from recommender import ItemSimilarityRecommender


class TestItemSimilarityRecommender(unittest.TestCase):
    def setUp(self):
        """ Set up the necessary data for testing """
        self.recommender = ItemSimilarityRecommender()
        self.recommender.train_data = pandas.DataFrame({
            'user_id': [1, 1, 2, 2, 3, 3],
            'item_id': [100, 106, 100, 109, 106, 109]
        })
        self.recommender.user_id = 'user_id'
        self.recommender.item_id = 'item_id'

    def test_get_user_items(self):
        """ Test case for a user with interactions """
        # Define test data
        user_items = self.recommender.get_user_items(1)
        expected_items = [100, 106]

        """
        Perform assertions to check if the expected output
        matches the expected output.
        """
        self.assertEqual(user_items, expected_items)

        """ Test case for a user with no interactions """
        user_items = self.recommender.get_user_items(4)
        expected_items = []
        self.assertEqual(user_items, expected_items)

        """ Test case for a non-existent user """
        user_items = self.recommender.get_user_items(5)
        expected_items = []
        self.assertEqual(user_items, expected_items)
    
    def test_get_items_users(self):
        """ Test case for an item with interaction """
        item_users = self.recommender.get_item_users(109)
        expected_users = {2, 3}
        self.assertEqual(item_users, expected_users,\
            "Retrieved item users do not match the expected users")

        """ Test case for an item with no interactions """
        item_users = self.recommender.get_item_users(101)
        expected_users = set()
        self.assertEqual(item_users, expected_users)

        """ Test case for a non-existent item """
        item_users = self.recommender.get_item_users(102)
        expected_users = set()
        self.assertEqual(item_users, expected_users)
    
    def test_get_all_items_train_data(self):
        """ Test case for all items in train data """
        # Define test data
        expected_items = [100, 106, 109]
        actual_items = self.recommender.get_all_items_train_data()

        """
        Perform assertion to check if the expected output
        matches the actual output.
        """
        self.assertListEqual(actual_items, expected_items,\
            "Retrieved items do not match the expected items")
    
    def test_construct_cooccurence_matrix(self):
        """ Test case for cooccurence matrix """
        # Define test data
        user_songs = ["song_1", "song_2", "song_3"]
        all_songs = ["song_1", "song_2", "song_3", "song_4", "song_5"]

        # Call the method being tested
        cooccurrence_matrix = self.recommender.\
            construct_cooccurence_matrix(user_songs, all_songs)
        
        """
        Perform assertions to check if the expected output
        matches the actual output.
        """
        self.assertEqual(cooccurrence_matrix.shape[0], len(user_songs),\
            "Number of rows in the cooccurrence matrix is incorrect")
        self.assertEqual(cooccurrence_matrix.shape[1], len(all_songs),\
            "Number of columns in the cooccurrence matrix is incorrect")
        self.assertIsInstance(cooccurrence_matrix, np.matrix,\
            "Cooccurrence matrix is not a type of np.matrix")
    
    def test_generate_top_recommendations(self):
        """ Testcase for top recommendation to user """
        # Define test data
        user = "user_1"
        cooccurrence_matrix = np.matrix([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5],\
            [0.3, 0.2, 0.5]])
        all_songs = ["song_1", "song_2", "song_3"]
        user_songs = ["song_1"]

        # Call the method being tested
        df_recommendations = self.recommender.generate_top_recommendations(user,\
            cooccurrence_matrix, all_songs, user_songs)
        """
        Perform assertions to check the properties of the
        recommendation dataframe.
        """
        self.assertIsInstance(df_recommendations, pandas.DataFrame,\
            "Recommendations are not of type pandas.DataFramr")
        self.assertEqual(df_recommendations.shape[0], 2,\
            "Number of recommendations is incorrect")
        self.assertEqual(df_recommendations.shape[1], 4,\
            "Number of columns in the recommendations dataframe is incorrect")
        self.assertEqual(df_recommendations.iloc[0]["user_id"], user,\
            "Incorrect user ID in recommendations")
        
        # Check if the dataframe contains the expected songs
        expected_songs = ["song_2", "song_3"]
        for song in expected_songs:
            self.assertTrue(song in df_recommendations["song"].values,\
                f"{song} is missing from recommendation")
    
    def test_create(self):
        """ Test case for create, initializing recommender with train_data """
        # Define test data
        train_data = ["song_1", "song_2", "song_3"]
        user_id = [1, 1, 2, 2, 3, 3],
        item_id = [100, 106, 100, 109, 106, 109]

        # Call the method being tested
        self.recommender.create(train_data, user_id, item_id)

        # Perform assertions to check the properties of the recommender object
        self.assertEqual(self.recommender.train_data, train_data,\
            "Train data is not set correctly")
        self.assertEqual(self.recommender.user_id, user_id,\
            "User ID is not set correctly")
        self.assertEqual(self.recommender.item_id, item_id,\
            "Item ID is not set correctly")
    
    @patch('builtins.print')
    def test_get_similar_items(self, mock_print):
        """
        Test case for getting similar items from train_data
        """
        # Prepare test data
        item_list = ['item_1', 'item_2', 'item_3']

        # Mock the method calls with in get_similar_items
        self.recommender.get_all_items_train_data = lambda: ['item_1', 'item_2',\
                                                             'item_3', 'item_4']
        self.recommender.construct_cooccurence_matrix = lambda user_songs, all_songs:\
            [['similarity_1', 'similarity_2'], ['similarity_3', 'similarity_4']]
        self.recommender.generate_top_recommendations = lambda user, cooccurence_matrix,\
            all_songs, user_songs: 'recommendations'
        
        # Call the function to test
        result = self.recommender.get_similar_items(item_list)

        # Assertions
        self.assertEqual(result, 'recommendations')
        self.assertEqual(mock_print.call_args[0][0], "No. of unique songs in the training set: 4")


if __name__ == '__main__':
    unittest.main()
