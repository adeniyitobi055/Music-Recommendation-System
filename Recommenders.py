import numpy as np
import pandas


class item_similarity_recommender_py():
    def __init__(self):
        """Initialization"""
        self.train_data = None #rep training data used to build the recommender system
        self.user_id = None #refers to the users
        self.item_id = None #refers to items in the recommender system
        self.cooccurence_matrix = None #similarities between items based on user preferences
        self.songs_dict = None #maps items IDs to their corresponding names
        self.rev_songs_dict = None #maps item names to their corresponding IDs
        self.item_similarity_recommendations = None #stores recommendations generated

    def get_user_items(self, user):
        """
        Get item IDs that a specific user has iteracted with in the
        training data
        """
        user_data = self.train_data[self.train_data[self.user_id] == user]
        user_items = list(user_data[self.item_id].unique())

        return user_items

    def get_item_users(self, item):
        """
        Retrieve the users who have interacted with a 
        specific item in the training data
        """
        item_data = self.train_data[self.train_data[self.item_id] == item]
        item_users = set(item_data[self.user_id].unique())

        return item_users

    def get_all_items_train_data(self):
        """
        Retrieves all unique items present in the training data
        """
        all_items = list(self.train_data[self.item_id].unique())

        return all_items

    def construct_cooccurence_matrix(self, user_songs, all_songs):
        """
        Build a co-occurrence matrix based on the user-item
        interactions in the training data
        """
        user_songs_users = []

        for i in range(0, len(user_songs)):
            user_songs_users.append(self.get_item_users(user_songs[i]))

        cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs),
                                                len(all_songs))), float)

        for i in range(0, len(all_songs)):
            songs_i_data = self.train_data[self.train_data\
                    [self.item_id] == all_songs[i]]
            users_i = set(song_i_data[self.user_id].unique())

            for j in range(0, len(user_songs)):
                users_j = user_songs_users[j]
                users_intersection = users_i.intersection(users_j)

                if len(users_intersection) != 0:
                    users_union = users_i.union(users.j)
                    cooccurence_matrix[j, i] =\
                            float(len(users_intersection)) / float(len(users_union))
                else:
                    cooccurence_matrix[j, i] = 0

        return cooccurence_matrix

    def generate_top_recommendations(self, user, cooccurence_matrix, all_songs, user_songs):
        """
        Generates top recommenations for a given user based on
        the co-occurence matrix and user-item interactions
        """
        non_zero_count = np.count_nonzero(cooccurence_matrix)
        print("Non zero values in cooccurence_matrix: %d" % non_zero_count)

        user_sim_scores = cooccurence_matrix.sum(axis=0)\
                / float(cooccurence_matrix.shape[0])
        user_sim_scores = np.array(user_sim_scores)[0].tolist()

        sort_index = sorted(((e, i) for i, e in enumerate(list\
                (user_sim_scores))), reverse=True)
        columns = ['user_id', 'song', 'score', 'rank']
        df = pandas.DataFrame(columns=columns)

        rank = 1
        for i in range(0, len(sort_index)):
            if ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]]\
                    not in user_songs and rank <= 10:
                    df.loc[len(df)] = [user, all_songs[sort_index[i][1]],\
                            sort_index[i][0], rank]
                    rank = rank + 1

        if df.shape[0] == 0:
            print("The current user has no songs for training the\
                    item similarity-based recommendation model.")
            return -1
        else:
            global df5
            df5 = ['NAME'] + df['song'].tolist()
            return df

    def create(self, train_data, user_id, item_id):
        """
        Initializes the recommender system with the training data
        and the user and item IDs.
        """
        self.train_data = train_data
        self.user_id = user_id
        self.item_id = item_id

    def recommend(self, user):
        """
        Recommend items for a given user
        """
        user_songs = self.get_user_items(user)
        print("No. of unique songs for the user: %d" % len(user_songs))

        all_songs = self.get_all_items_train_data()
        print("No. of unique songs in the training set: %d" % len(all_songs))

        cooccurence_matrix = self.construct_cooccurence_matrix(user_songs, all_songs)

        user = ""
        df_recommendations = self.generate_top_recommendations\
                (user, coocurence_matrix, all_songs, user_songs)

        return df_recommendations
