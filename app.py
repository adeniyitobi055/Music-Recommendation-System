#!/usr/bin/python3
"""
app module
"""
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import recommender as recommender
global s1, data
s1=0
data=0
rec_song=0


def model1(s1):
    # Load the files that'll be used to train the recommender system
    song_df_1 = pd.read_csv('user_data_2.csv')
    song_df_1.head()
    song_df_2 = pd.read_csv('metadata_file.csv')
    song_df_2.head()
    song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(['song_id']), on='song_id', how='left')
    song_df.head()
    print(len(song_df_1), len(song_df_2))
    len(song_df)
    song_df['song'] = song_df['title'] + ' - ' + song_df['artist_name']
    song_df.head()
    song_df = song_df.head(20000)
    song_grouped = song_df.groupby(['song']).agg({'listen_count':'count'}).reset_index()
    song_grouped.head()
    grouped_sum = song_grouped['listen_count'].sum()
    song_grouped['percentage'] = (song_grouped['listen_count'] / grouped_sum) * 100
    song_grouped.sort_values(['listen_count', 'song'], ascending=[0,1])
    ir = recommender.ItemSimilarityRecommender()
    ir.create(song_df, 'user_id', 'song')
    user_items = ir.get_user_items(song_df['user_id'][10])
    for user_item in user_items:
        print(user_item)
    

    # DataFrame 3
    df3 = song_df[song_df['title'].str.contains(str(s1), na=False)][['song']]
    # Print list
    list1 = df3.values.tolist()
    if len(list1) > 0:
        t1 = list1[0][0]
    else:
        t1 = "No similar items found"

    ir.get_similar_items([str(t1)])
    global rec_song
    rec_song = recommender.df5


app = Flask(__name__)


@app.route('/')
def index_page():
    """
    Returns index page to the user
    """
    return render_template('index.html')
    
@app.route('/', methods=['POST'])
def search():
    """
    Allows user to search
    """
    text = request.form['searchsong']
    s1 = text
    model1(s1)
    print(rec_song)
    
    return render_template('recommend.html', data=rec_song)


if __name__ == "__main__":
    app.run(debug=True)